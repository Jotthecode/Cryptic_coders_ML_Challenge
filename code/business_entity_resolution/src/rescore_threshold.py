"""
ML Challenge 2026 - Business Entity Resolution
Ultra-fast rescoring tool for candidate matches.
Allows adjusting classification threshold (e.g. from 0.50 to 0.70) in ~2 minutes
without re-running expensive candidate indexing.
"""

import argparse
import gc
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import FastCountryBlocker
from train_matching import extract_pair_features
from utils import clean_address, clean_domain, clean_text, extract_address_numbers


def rescore_pipeline(
    input_matches: str = "output/matching_results.tsv",
    input_candidates: str = "output/candidate_pairs.tsv",
    output_matches: str = "output/matching_results_p70.tsv",
    test_dir: str = "dataset/test",
    cache_dir: str = "tmp_cache",
    model_path: str = "models/matching_model.pkl",
    threshold: float = 0.70,
    batch_size: int = 50000,
):
    print("=" * 70)
    print(f"  HIGH-PRECISION RESCORING TO THRESHOLD {threshold:.2f}")
    print("=" * 70)
    t_start = time.time()

    # Load model
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
        model = payload["model"] if isinstance(payload, dict) else payload

    # Read existing matches (threshold 0.50) and candidate ranks
    print(f"Reading existing candidate matches from {input_matches} and {input_candidates}...")
    s1_matches = {}
    total_50_matches = 0
    with open(input_matches, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1 and parts[1]:
                m_list = parts[1].split(",")
                s1_matches[parts[0]] = m_list
                total_50_matches += len(m_list)

    print(f"Total S1 entities with matches at 0.50: {len(s1_matches):,}")
    print(f"Total candidate matches to rescore: {total_50_matches:,}")

    # Read candidate ranks for these entities
    s1_cand_ranks = {}
    with open(input_candidates, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            if s1_id in s1_matches:
                if len(parts) > 1 and parts[1]:
                    cands = parts[1].split(",")
                    s1_cand_ranks[s1_id] = {cid: idx + 1 for idx, cid in enumerate(cands)}
                else:
                    s1_cand_ranks[s1_id] = {}

    # Load S1 test records
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading Source 1 records from {s1_path}...")
    s1_df = pl.read_csv(s1_path, separator="\t")
    s1_df = s1_df.with_columns(
        pl.col("country").fill_null("UNKNOWN").str.strip_chars().str.to_uppercase().alias("c_norm")
    )

    all_s1_ids = s1_df["entity_id"].to_list()
    countries = s1_df["c_norm"].unique().to_list()

    p70_matches: Dict[str, List[str]] = defaultdict(list)

    for country in countries:
        country_s1 = s1_df.filter(pl.col("c_norm") == country)
        parquet_path = os.path.join(cache_dir, f"targets_{country}.parquet")
        if not os.path.exists(parquet_path):
            continue

        print(f"\n[{country}] Loading targets from {parquet_path}...")
        t0 = time.time()
        targets_df = pl.read_parquet(parquet_path)
        print(f"[{country}] Loaded {len(targets_df):,} targets in {time.time()-t0:.2f}s.")

        # Build fast lookup for target records
        target_ids = targets_df["entity_id"].to_list()
        raw_names = targets_df["business_name"].to_list()
        raw_addrs = targets_df["business_address"].to_list()
        del targets_df
        gc.collect()

        target_map = {
            target_ids[i]: (raw_names[i], raw_addrs[i]) for i in range(len(target_ids))
        }

        # S1 records for this country that have matches to rescore
        s1_country_recs = {}
        for r in country_s1.iter_rows(named=True):
            sid = r["entity_id"]
            if sid in s1_matches:
                s1_country_recs[sid] = (r.get("business_name") or "", r.get("business_address") or "")

        print(f"[{country}] Extracting features for {len(s1_country_recs):,} S1 entities to rescore...")
        t0 = time.time()

        pairs_to_score = []
        pair_keys = []

        for sid, (s1_n, s1_a) in s1_country_recs.items():
            s1_n_cl = clean_text(s1_n, remove_legal=True)
            s1_n_ns = clean_domain(s1_n).replace(" ", "")
            s1_a_cl = clean_address(s1_a)
            s1_nums = extract_address_numbers(s1_a)

            rank_map = s1_cand_ranks.get(sid, {})

            for tid in s1_matches[sid]:
                t_rec = target_map.get(tid)
                if not t_rec:
                    continue
                tn, ta = t_rec
                b_rank = rank_map.get(tid, 1)

                feat = extract_pair_features(
                    s1_name_cl=s1_n_cl,
                    s1_name_ns=s1_n_ns,
                    s1_addr_cl=s1_a_cl,
                    s1_nums=s1_nums,
                    t_name_cl=clean_text(tn, remove_legal=True),
                    t_name_ns=clean_domain(tn).replace(" ", ""),
                    t_addr_cl=clean_address(ta),
                    t_nums=extract_address_numbers(ta) if s1_nums else set(),
                    target_id=tid,
                    b_score=10.0 / b_rank,  # high fidelity rank-based IDF proxy
                    b_rank=b_rank,
                )
                pairs_to_score.append(feat)
                pair_keys.append((sid, tid))

                if len(pairs_to_score) >= batch_size:
                    X_batch = np.array(pairs_to_score, dtype=np.float32)
                    probs = model.predict_proba(X_batch)[:, 1]
                    for (s_id, t_id), prob in zip(pair_keys, probs):
                        if prob >= threshold:
                            p70_matches[s_id].append(t_id)
                    pairs_to_score.clear()
                    pair_keys.clear()

        if pairs_to_score:
            X_batch = np.array(pairs_to_score, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]
            for (s_id, t_id), prob in zip(pair_keys, probs):
                if prob >= threshold:
                    p70_matches[s_id].append(t_id)
            pairs_to_score.clear()
            pair_keys.clear()

        print(f"[{country}] Completed in {time.time()-t0:.2f}s.")
        del target_map, target_ids, raw_names, raw_addrs, s1_country_recs
        gc.collect()

    # Write out matching_results_p70.tsv
    print(f"\nWriting {output_matches} in exact test_source1.tsv order...")
    t0 = time.time()
    total_retained_matches = 0
    retained_entities = 0

    with open(output_matches, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matches = p70_matches.get(sid, [])
            if matches:
                total_retained_matches += len(matches)
                retained_entities += 1
                f_out.write(f"{sid}\t{','.join(matches)}\n")
            else:
                f_out.write(f"{sid}\t\n")

    print(f"Saved {len(all_s1_ids):,} rows in {time.time()-t0:.2f}s.")
    print("=" * 70)
    print("RESCORING COMPLETE!")
    print(f"  Total S1 entities: {len(all_s1_ids):,}")
    print(f"  Matches at threshold 0.50: {total_50_matches:,} across {len(s1_matches):,} entities")
    print(f"  Matches at threshold {threshold:.2f}: {total_retained_matches:,} across {retained_entities:,} entities")
    print(f"  Borderline noisy matches pruned: {total_50_matches - total_retained_matches:,}")
    print(f"  Execution time: {time.time()-t_start:.2f}s")
    print(f"  Output saved to: {output_matches}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast threshold rescorer.")
    parser.add_argument("--threshold", type=float, default=0.70, help="New threshold (e.g. 0.70)")
    parser.add_argument("--output", default="output/matching_results_p70.tsv", help="Output TSV path")
    args = parser.parse_args()

    rescore_pipeline(threshold=args.threshold, output_matches=args.output)
