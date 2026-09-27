#!/usr/bin/env python3
"""
ML Challenge 2026 - High-Precision Candidate Rescorer
Author: Jot Ajmani (Team: QuantumQuants)

Applies:
  1. Source-aware asymmetric thresholding (tau_S2 = 0.72, tau_S3 = 0.82)
  2. Street number conflict penalty (eliminates same-street franchise false merges)
  3. Strict 1-to-1 maximum-weight competitive bipartite assignment
  4. Natural singleton preservation (no artificial low-confidence gating)
"""

import argparse
import collections
import gc
import os
import pickle
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from utils import (
    clean_address,
    clean_domain,
    clean_text,
    extract_address_numbers,
    normalize_country,
)


def extract_pair_features_fast(
    s1_cl: str,
    s1_ns: str,
    s1_addr: str,
    s1_nums: Set[str],
    t_cl: str,
    t_ns: str,
    t_addr: str,
    t_nums: Set[str],
    target_id: str,
    b_score: float = 5.0,
    b_rank: int = 1,
) -> List[float]:
    f_name_ratio = fuzz.ratio(s1_cl, t_cl) / 100.0 if (s1_cl and t_cl) else 0.0
    f_name_partial = fuzz.partial_ratio(s1_cl, t_cl) / 100.0 if (s1_cl and t_cl) else 0.0
    f_name_sort = fuzz.token_sort_ratio(s1_cl, t_cl) / 100.0 if (s1_cl and t_cl) else 0.0
    f_name_set = fuzz.token_set_ratio(s1_cl, t_cl) / 100.0 if (s1_cl and t_cl) else 0.0

    dom_exact = 1.0 if (s1_ns and s1_ns == t_ns) else 0.0
    dom_contains = 1.0 if (s1_ns and t_ns and (s1_ns in t_ns or t_ns in s1_ns)) else 0.0
    exact_name = 1.0 if (s1_cl and s1_cl == t_cl) else 0.0

    w1 = s1_cl.split()[0] if s1_cl else ""
    w2 = t_cl.split()[0] if t_cl else ""
    first_word_match = 1.0 if (w1 and w1 == w2) else 0.0

    has_a1 = 1.0 if len(s1_addr) > 0 else 0.0
    has_a2 = 1.0 if len(t_addr) > 0 else 0.0
    f_addr_ratio = (fuzz.ratio(s1_addr, t_addr) / 100.0) if (has_a1 and has_a2) else 0.0
    f_addr_set = (fuzz.token_set_ratio(s1_addr, t_addr) / 100.0) if (has_a1 and has_a2) else 0.0
    exact_addr = 1.0 if (has_a1 and has_a2 and s1_addr == t_addr) else 0.0

    num_jaccard = (len(s1_nums & t_nums) / len(s1_nums | t_nums)) if (s1_nums and t_nums) else 0.0

    aw1 = set([w for w in s1_addr.split() if len(w) >= 3])
    aw2 = set([w for w in t_addr.split() if len(w) >= 3])
    addr_word_jaccard = (len(aw1 & aw2) / len(aw1 | aw2)) if (aw1 and aw2) else 0.0

    l1 = len(s1_cl)
    l2 = len(t_cl)
    len_ratio = min(l1, l2) / max(l1, l2, 1)

    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0

    return [
        f_name_ratio,
        f_name_partial,
        f_name_sort,
        f_name_set,
        dom_exact,
        dom_contains,
        exact_name,
        first_word_match,
        f_addr_ratio,
        f_addr_set,
        exact_addr,
        num_jaccard,
        addr_word_jaccard,
        has_a1,
        has_a2,
        len_ratio,
        is_s2,
        is_s3,
        float(b_score),
        float(b_rank),
    ]


def run_rescoring(
    cand_file: str = "output/candidate_pairs.tsv",
    test_dir: str = "dataset/test",
    model_path: str = "models/matching_model.pkl",
    output_file: str = "output/matching_results.tsv",
    s2_thresh: float = 0.72,
    s3_thresh: float = 0.82,
    batch_size: int = 50000,
):
    print("=" * 70)
    print("   ML CHALLENGE 2026: HIGH-PRECISION ASYMMETRIC RESCORING")
    print("   Author: Jot Ajmani (Team: QuantumQuants)")
    print("=" * 70)
    print(f"Candidate file     : {cand_file}")
    print(f"Test directory     : {test_dir}")
    print(f"Model path         : {model_path}")
    print(f"Output file        : {output_file}")
    print(f"Calibrated Thresholds : S2 = {s2_thresh:.2f}, S3 = {s3_thresh:.2f}")

    # Load model
    print(f"\nLoading LightGBM model from {model_path}...")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]

    # Read S1 in exact order and partition by country
    print("\nReading test_source1.tsv...")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s1_df = pl.read_csv(s1_path, separator="\t")
    ordered_s1_ids = s1_df["entity_id"].to_list()
    total_s1 = len(ordered_s1_ids)

    # Read candidate pairs into dictionary
    print(f"Reading {cand_file}...")
    t0 = time.time()
    cand_df = pl.read_csv(cand_file, separator="\t")
    cand_map = {}
    for sid, c_str in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]):
        if c_str and c_str.strip():
            cand_map[sid] = [t.strip() for t in c_str.split(",") if t.strip()]
        else:
            cand_map[sid] = []
    print(f"Loaded candidates for {len(cand_map):,} entities in {time.time()-t0:.2f}s.")

    # Partition S1 entities by country
    country_s1_map = collections.defaultdict(list)
    for sid, name, addr, country in zip(
        s1_df["entity_id"],
        s1_df["business_name"].fill_null(""),
        s1_df["business_address"].fill_null(""),
        s1_df["country"].fill_null("UNKNOWN"),
    ):
        c_upper = normalize_country(country)
        country_s1_map[c_upper].append((sid, name, addr))

    del s1_df, cand_df
    gc.collect()

    all_matches_map = {}
    total_matches_all = 0

    # Process each country partition
    for country, s1_records in country_s1_map.items():
        c_t0 = time.time()
        print(f"\n[{country}] Processing {len(s1_records):,} S1 entities...")

        # Load cached target parquet
        parquet_path = os.path.join("tmp_cache", f"targets_{country}.parquet")
        if not os.path.exists(parquet_path):
            print(f"Warning: {parquet_path} not found! Skipping {country}...")
            continue

        print(f"[{country}] Loading targets from {parquet_path}...")
        t_df = pl.read_parquet(parquet_path)
        print(f"[{country}] Loaded {len(t_df):,} target records.")

        target_lookup = {}
        for tid, bn, ba in zip(
            t_df["entity_id"],
            t_df["business_name"].fill_null(""),
            t_df["business_address"].fill_null(""),
        ):
            target_lookup[tid] = (
                clean_text(bn, remove_legal=True),
                clean_domain(bn).replace(" ", ""),
                clean_address(ba),
                extract_address_numbers(ba),
            )
        del t_df
        gc.collect()
        print(f"[{country}] Target dictionary ready.")

        # Score candidate pairs
        country_pool = []
        batch_pairs = []
        batch_keys = []
        batch_extra = []

        total_pairs_scored = 0

        for sid, sn, sa in s1_records:
            cands = cand_map.get(sid, [])
            if not cands:
                continue

            s1_cl = clean_text(sn, remove_legal=True)
            s1_ns = clean_domain(sn).replace(" ", "")
            s1_addr = clean_address(sa)
            s1_nums = extract_address_numbers(sa)

            for rank_idx, tid in enumerate(cands, 1):
                tinfo = target_lookup.get(tid)
                if not tinfo:
                    continue

                feat = extract_pair_features_fast(
                    s1_cl=s1_cl,
                    s1_ns=s1_ns,
                    s1_addr=s1_addr,
                    s1_nums=s1_nums,
                    t_cl=tinfo[0],
                    t_ns=tinfo[1],
                    t_addr=tinfo[2],
                    t_nums=tinfo[3],
                    target_id=tid,
                    b_score=1.0 / rank_idx,
                    b_rank=rank_idx,
                )
                batch_pairs.append(feat)
                batch_keys.append((sid, tid))
                batch_extra.append((s1_nums, tinfo[3], s1_cl == tinfo[0]))

                if len(batch_pairs) >= batch_size:
                    X_b = np.array(batch_pairs, dtype=np.float32)
                    probs = model.predict_proba(X_b)[:, 1]
                    for (s_id, t_id), (s_num, t_num, exact_n), prob in zip(batch_keys, batch_extra, probs):
                        p = float(prob)
                        # Street number conflict check
                        if s_num and t_num and not (s_num & t_num) and not exact_n:
                            p *= 0.1
                        th = s2_thresh if t_id.startswith("S2-") else s3_thresh
                        if p >= th:
                            country_pool.append((p, s_id, t_id))
                    total_pairs_scored += len(batch_pairs)
                    batch_pairs = []
                    batch_keys = []
                    batch_extra = []

        # Flush remaining batch
        if batch_pairs:
            X_b = np.array(batch_pairs, dtype=np.float32)
            probs = model.predict_proba(X_b)[:, 1]
            for (s_id, t_id), (s_num, t_num, exact_n), prob in zip(batch_keys, batch_extra, probs):
                p = float(prob)
                if s_num and t_num and not (s_num & t_num) and not exact_n:
                    p *= 0.1
                th = s2_thresh if t_id.startswith("S2-") else s3_thresh
                if p >= th:
                    country_pool.append((p, s_id, t_id))
            total_pairs_scored += len(batch_pairs)

        print(f"[{country}] Scored {total_pairs_scored:,} candidate pairs in {time.time()-c_t0:.1f}s.")
        print(f"[{country}] Candidates meeting threshold: {len(country_pool):,}.")

        # 1-to-1 Greedy Competitive Bipartite Assignment
        country_pool.sort(key=lambda x: x[0], reverse=True)
        assigned_targets = set()
        c_matches = collections.defaultdict(list)

        for p, sid, tid in country_pool:
            if tid not in assigned_targets:
                assigned_targets.add(tid)
                c_matches[sid].append(tid)

        matched_s1 = sum(1 for sid, _, _ in s1_records if c_matches.get(sid))
        total_m = sum(len(m) for m in c_matches.values())
        print(f"[{country}] Matched S1: {matched_s1:,} / {len(s1_records):,} ({matched_s1/len(s1_records):.2%}).")
        print(f"[{country}] Total matches retained: {total_m:,} (avg {total_m/len(s1_records):.2f} per entity).")

        for sid, _, _ in s1_records:
            all_matches_map[sid] = ",".join(c_matches.get(sid, []))
        total_matches_all += total_m

        del target_lookup, country_pool, c_matches, assigned_targets
        gc.collect()

    # Stream out final matching_results.tsv
    print("\nWriting out final matching_results.tsv in exact test_source1 order...")
    t0 = time.time()
    n_empty = 0
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_s1_ids:
            m_str = all_matches_map.get(sid, "")
            if not m_str:
                n_empty += 1
            f.write(f"{sid}\t{m_str}\n")

    print(f"Saved {output_file} in {time.time()-t0:.2f}s.")
    print("=" * 70)
    print("FINAL SUMMARY:")
    print(f"  Total Entities      : {total_s1:,}")
    print(f"  Singletons (Empty)  : {n_empty:,} ({n_empty/total_s1:.2%})")
    print(f"  Entities with Matches: {total_s1 - n_empty:,} ({(total_s1 - n_empty)/total_s1:.2%})")
    print(f"  Total Links Retained: {total_matches_all:,} (avg {total_matches_all/total_s1:.2f} per entity)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Precision Candidate Rescorer")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Candidate pairs file")
    parser.add_argument("--test-dir", default="dataset/test", help="Test directory")
    parser.add_argument("--model-path", default="models/matching_model.pkl", help="Model path")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Output matching file")
    parser.add_argument("--s2-thresh", type=float, default=0.72, help="Threshold for Source 2")
    parser.add_argument("--s3-thresh", type=float, default=0.82, help="Threshold for Source 3")
    parser.add_argument("--batch-size", type=int, default=50000, help="Inference batch size")

    args = parser.parse_args()
    run_rescoring(
        cand_file=args.candidate,
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_file=args.output,
        s2_thresh=args.s2_thresh,
        s3_thresh=args.s3_thresh,
        batch_size=args.batch_size,
    )
