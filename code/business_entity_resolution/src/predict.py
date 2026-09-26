"""
ML Challenge 2026 - Business Entity Resolution
Production inference script generating candidate_pairs.tsv and matching_results.tsv.
Key Upgrades:
  1. High-Recall Composite-Key Blocker (max_candidates = 20)
  2. 20 RapidFuzz + Address Structure Features
  3. Strict 1-to-1 Target Uniqueness (Greedy Maximum-Weight Bipartite Assignment)
  4. Singleton Margin Gating (Recovers valid non-singletons while protecting true singletons)
  5. Low RAM footprint (< 1.5 GB peak)
  6. Strict Candidate Subset Constraint
  7. Exact row-for-row alignment with test_source1.tsv
"""

import argparse
import gc
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
from tqdm import tqdm

from blocking import FastCountryBlocker
from train_matching import FEATURE_NAMES, extract_pair_features
from utils import clean_address, clean_domain, clean_text, extract_address_numbers, normalize_country


def ensure_target_parquets(test_dir: str, cache_dir: str = "tmp_cache") -> List[str]:
    """Ensure S2 + S3 test targets are partitioned by country as parquet files."""
    os.makedirs(cache_dir, exist_ok=True)
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    existing_parquets = [f for f in os.listdir(cache_dir) if f.startswith("targets_") and f.endswith(".parquet")]
    if existing_parquets:
        countries = [f.replace("targets_", "").replace(".parquet", "") for f in existing_parquets]
        print(f"Found cached country target parquets for: {countries}")
        return countries

    print(f"Building country-partitioned target cache in {cache_dir}...")
    t0 = time.time()
    s2 = pl.read_csv(s2_path, separator="\t")
    s3 = pl.read_csv(s3_path, separator="\t")
    targets = pl.concat([s2, s3], how="vertical").unique(subset=["entity_id"])
    del s2, s3
    gc.collect()

    targets = targets.with_columns(
        pl.col("country").fill_null("UNKNOWN").str.strip_chars().str.to_uppercase().alias("c_norm")
    )

    countries = targets["c_norm"].unique().to_list()
    for c in countries:
        sub = targets.filter(pl.col("c_norm") == c)
        out_path = os.path.join(cache_dir, f"targets_{c}.parquet")
        sub.write_parquet(out_path)
        print(f"  Cached {c}: {len(sub):,} targets -> {out_path}")

    del targets
    gc.collect()
    print(f"Target partitioning completed in {time.time()-t0:.2f}s.")
    return countries


def run_country_inference(
    country: str,
    s1_df: pl.DataFrame,
    cache_dir: str,
    model,
    threshold: float,
    results_map: Dict[str, Tuple[str, str]],
    batch_size: int = 50000,
    max_candidates: int = 20,
    singleton_gate_p: float = 0.28,
):
    """
    Run high-recall blocking, scoring, competitive 1-to-1 assignment, and singleton gating.
    Populates results_map[s1_id] = (cand_str, match_str).
    """
    country_upper = country.strip().upper()
    parquet_path = os.path.join(cache_dir, f"targets_{country_upper}.parquet")

    if not os.path.exists(parquet_path):
        print(f"WARNING: No targets found for country {country_upper}. All entities will be singletons.")
        for r in s1_df.iter_rows(named=True):
            results_map[r["entity_id"]] = ("", "")
        return

    print(f"\n[{country_upper}] Loading targets from {parquet_path}...")
    t0 = time.time()
    targets_df = pl.read_parquet(parquet_path)
    N_targets = len(targets_df)
    target_ids = targets_df["entity_id"].to_list()
    raw_names = targets_df["business_name"].to_list()
    raw_addrs = targets_df["business_address"].to_list()
    del targets_df
    gc.collect()
    print(f"[{country_upper}] Loaded {N_targets:,} targets in {time.time()-t0:.2f}s.")

    # Build composite inverted index
    t0 = time.time()
    blocker = FastCountryBlocker(max_candidates=max_candidates)
    blocker.fit_targets(target_ids, raw_names, raw_addrs)
    print(f"[{country_upper}] Composite index built in {time.time()-t0:.2f}s with {len(blocker.token_index):,} tokens.")

    # Pre-normalize target records once to eliminate millions of redundant regexes
    t0 = time.time()
    t_cl_names = [clean_text(str(n).lower() if n else "", remove_legal=True) for n in raw_names]
    t_ns_names = [clean_domain(str(n).lower() if n else "").replace(" ", "") for n in raw_names]
    t_cl_addrs = [clean_address(str(a).lower() if a else "") for a in raw_addrs]
    t_nums_list = [extract_address_numbers(str(a).lower() if a else "") for a in raw_addrs]
    print(f"[{country_upper}] Target strings pre-normalized in {time.time()-t0:.2f}s.")

    total_s1 = len(s1_df)
    print(f"[{country_upper}] Scoring {total_s1:,} S1 entities in batches of {batch_size}...")
    t0 = time.time()

    s1_ids = s1_df["entity_id"].to_list()
    s1_names = s1_df["business_name"].to_list()
    s1_addrs = s1_df["business_address"].to_list()

    all_cands_map: Dict[str, List[str]] = {}
    candidate_pairs_pool: List[Tuple[float, str, str]] = []  # (prob, s1_id, target_id)
    s1_top_candidate: Dict[str, Tuple[str, float]] = {}      # s1_id -> (top_tid, top_prob)

    for start_idx in tqdm(range(0, total_s1, batch_size), desc=f"Processing {country_upper}"):
        end_idx = min(start_idx + batch_size, total_s1)

        batch_features = []
        batch_pair_keys = []

        for i in range(start_idx, end_idx):
            s1_id = s1_ids[i]
            s1_n = s1_names[i]
            s1_a = s1_addrs[i]

            s1_n_cl = clean_text(s1_n, remove_legal=True)
            s1_n_ns = clean_domain(s1_n).replace(" ", "")
            s1_a_cl = clean_address(s1_a)
            s1_nums = extract_address_numbers(s1_a)

            cands = blocker.query_entity(s1_n, s1_a, max_candidates=max_candidates)
            cand_id_list = [target_ids[tidx] for tidx, _, _ in cands]
            all_cands_map[s1_id] = cand_id_list

            for tidx, b_score, b_rank in cands:
                tid = target_ids[tidx]

                feat = extract_pair_features(
                    s1_name_cl=s1_n_cl,
                    s1_name_ns=s1_n_ns,
                    s1_addr_cl=s1_a_cl,
                    s1_nums=s1_nums,
                    t_name_cl=t_cl_names[tidx],
                    t_name_ns=t_ns_names[tidx],
                    t_addr_cl=t_cl_addrs[tidx],
                    t_nums=t_nums_list[tidx],
                    target_id=tid,
                    b_score=b_score,
                    b_rank=b_rank,
                )
                batch_features.append(feat)
                batch_pair_keys.append((s1_id, tid))

        # Model inference on batch
        if batch_features and model is not None:
            X_batch = np.array(batch_features, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]

            for (sid, tid), prob in zip(batch_pair_keys, probs):
                p = float(prob)
                # Track top candidate per S1 entity
                if sid not in s1_top_candidate or p > s1_top_candidate[sid][1]:
                    s1_top_candidate[sid] = (tid, p)

                # Keep candidates above probability threshold
                if p >= threshold:
                    candidate_pairs_pool.append((p, sid, tid))

    print(f"[{country_upper}] Scored {total_s1:,} entities in {time.time()-t0:.2f}s.")
    print(f"[{country_upper}] Total candidate match predictions before 1-to-1 assignment: {len(candidate_pairs_pool):,}")

    # 1-to-1 Competitive Assignment (Greedy Maximum-Weight Matching)
    # A target record can match at most ONE S1 entity in ground truth
    candidate_pairs_pool.sort(key=lambda x: x[0], reverse=True)
    assigned_targets: Set[str] = set()
    s1_assigned_matches: Dict[str, List[str]] = defaultdict(list)

    for p, sid, tid in candidate_pairs_pool:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            s1_assigned_matches[sid].append(tid)

    # Singleton Gating: for entities with 0 matches, recover high-confidence top candidate
    singleton_recovers = 0
    for sid in s1_ids:
        if not s1_assigned_matches.get(sid):
            if sid in s1_top_candidate:
                top_tid, top_p = s1_top_candidate[sid]
                if top_p >= singleton_gate_p and top_tid not in assigned_targets:
                    assigned_targets.add(top_tid)
                    s1_assigned_matches[sid].append(top_tid)
                    singleton_recovers += 1

    print(f"[{country_upper}] 1-to-1 assignment completed. Recovered {singleton_recovers:,} borderline non-singletons.")
    total_matched_s1 = sum(1 for sid in s1_ids if s1_assigned_matches.get(sid))
    total_matches = sum(len(m) for m in s1_assigned_matches.values())
    print(f"[{country_upper}] S1 entities with matches: {total_matched_s1:,} / {total_s1:,} ({total_matched_s1/total_s1:.2%}).")
    print(f"[{country_upper}] Total matches retained: {total_matches:,}.")

    # Store results
    for sid in s1_ids:
        cands = all_cands_map.get(sid, [])
        matches = s1_assigned_matches.get(sid, [])

        cand_str = ",".join(cands) if cands else ""
        match_str = ",".join(matches) if matches else ""

        results_map[sid] = (cand_str, match_str)

    # Free memory before next country
    del blocker, target_ids, raw_names, raw_addrs, all_cands_map, candidate_pairs_pool
    del t_cl_names, t_ns_names, t_cl_addrs, t_nums_list
    del s1_ids, s1_names, s1_addrs, s1_assigned_matches, assigned_targets, s1_top_candidate
    gc.collect()


def predict_pipeline(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "models/matching_model.pkl",
    cache_dir: str = "tmp_cache",
    threshold: float = None,
    sample_size: int = None,
):
    """Run full test inference generating matching_results.tsv and candidate_pairs.tsv."""
    os.makedirs(output_dir, exist_ok=True)
    cand_file_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_file_path = os.path.join(output_dir, "matching_results.tsv")

    print("=" * 70)
    print("       ML CHALLENGE 2026: UPGRADED INFERENCE PIPELINE")
    print("=" * 70)
    print(f"Test Directory    : {test_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Model Path        : {model_path}")

    # Load trained model and threshold
    model = None
    default_th = 0.45
    if os.path.exists(model_path):
        print(f"Loading trained matching model from: {model_path}")
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
            if isinstance(payload, dict) and "model" in payload:
                model = payload["model"]
                default_th = payload.get("threshold", 0.45)
                print(f"Loaded model with tuned optimal threshold: {default_th:.2f}")
            else:
                model = payload
    else:
        print(f"WARNING: Model file {model_path} not found.")

    final_threshold = threshold if threshold is not None else default_th
    print(f"Active Classification Threshold: {final_threshold:.2f}")

    # Ensure target parquets are ready
    ensure_target_parquets(test_dir=test_dir, cache_dir=cache_dir)

    # Scan test S1 to discover countries dynamically (open-set support)
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading test Source 1 records from {s1_path}...")

    s1_df_all = pl.read_csv(s1_path, separator="\t")
    if sample_size:
        print(f"Running in fast sample mode (first {sample_size:,} entities)...")
        s1_df_all = s1_df_all[:sample_size]

    # Normalize country column
    s1_df_all = s1_df_all.with_columns(
        pl.col("country").fill_null("UNKNOWN").str.strip_chars().str.to_uppercase().alias("country_norm")
    )

    distinct_countries = s1_df_all["country_norm"].unique().to_list()
    print(f"Total Test S1 Entities: {len(s1_df_all):,}")
    print(f"Discovered Countries ({len(distinct_countries)}): {distinct_countries}")

    results_map: Dict[str, Tuple[str, str]] = {}

    # Process each country partition
    for country in distinct_countries:
        country_s1 = s1_df_all.filter(pl.col("country_norm") == country)
        run_country_inference(
            country=country,
            s1_df=country_s1,
            cache_dir=cache_dir,
            model=model,
            threshold=final_threshold,
            results_map=results_map,
            batch_size=50000,
            max_candidates=20,
        )

    # Write out deliverables in exact test_source1.tsv ordering
    print("\nWriting out final submission TSVs in exact test_source1 order...")
    all_s1_ids = s1_df_all["entity_id"].to_list()

    t0 = time.time()
    with open(cand_file_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands, _ = results_map.get(sid, ("", ""))
            f_cand.write(f"{sid}\t{cands}\n")
    print(f"Saved {cand_file_path} in {time.time()-t0:.2f}s.")

    t0 = time.time()
    with open(match_file_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            _, matches = results_map.get(sid, ("", ""))
            f_match.write(f"{sid}\t{matches}\n")
    print(f"Saved {match_file_path} in {time.time()-t0:.2f}s.")

    print("=" * 70)
    print("UPGRADED INFERENCE PIPELINE COMPLETE!")
    print(f"Candidate pairs saved to : {cand_file_path}")
    print(f"Matching results saved to: {match_file_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Set Inference Pipeline.")
    parser.add_argument("--test-dir", default="dataset/test", help="Test directory path")
    parser.add_argument("--output-dir", default="output", help="Output directory path")
    parser.add_argument("--model-path", default="models/matching_model.pkl", help="Model path")
    parser.add_argument("--threshold", type=float, default=None, help="Custom probability threshold")
    parser.add_argument("--sample-size", type=int, default=None, help="Run on small sample for testing")
    args = parser.parse_args()

    predict_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        sample_size=args.sample_size,
    )
