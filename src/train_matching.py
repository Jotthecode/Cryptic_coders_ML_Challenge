"""
ML Challenge 2026 - Business Entity Resolution
Pairwise matching feature extraction, model training, and threshold optimization module.
Uses RapidFuzz string similarities, address structure features, and LightGBM to optimize Macro F_0.5.
"""

import argparse
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from tqdm import tqdm

from blocking import FastCountryBlocker
from utils import (
    clean_address,
    clean_domain,
    clean_text,
    compute_f_beta,
    evaluate_f05_macro,
    extract_address_numbers,
    normalize_country,
)

FEATURE_NAMES = [
    "f_name_ratio",
    "f_name_partial",
    "f_name_sort",
    "f_name_set",
    "dom_exact",
    "dom_contains",
    "exact_name",
    "f_addr_ratio",
    "f_addr_set",
    "exact_addr",
    "num_jaccard",
    "has_addr1",
    "has_addr2",
    "len_ratio",
    "is_s2",
    "is_s3",
    "b_score",
    "b_rank",
]


def extract_pair_features(
    s1_name_cl: str,
    s1_name_ns: str,
    s1_addr_cl: str,
    s1_nums: Set[str],
    t_name_cl: str,
    t_name_ns: str,
    t_addr_cl: str,
    t_nums: Set[str],
    target_id: str,
    b_score: float,
    b_rank: int,
) -> List[float]:
    """Extract pairwise similarity and consistency features between S1 and a target candidate."""
    # 1. Name similarities
    f_name_ratio = fuzz.ratio(s1_name_cl, t_name_cl) / 100.0
    f_name_partial = fuzz.partial_ratio(s1_name_cl, t_name_cl) / 100.0
    f_name_sort = fuzz.token_sort_ratio(s1_name_cl, t_name_cl) / 100.0
    f_name_set = fuzz.token_set_ratio(s1_name_cl, t_name_cl) / 100.0

    # 2. Domain / Concatenated name match
    dom_exact = 1.0 if (s1_name_ns and s1_name_ns == t_name_ns) else 0.0
    dom_contains = (
        1.0
        if (s1_name_ns and t_name_ns and (s1_name_ns in t_name_ns or t_name_ns in s1_name_ns) and min(len(s1_name_ns), len(t_name_ns)) >= 5)
        else 0.0
    )
    exact_name = 1.0 if (s1_name_cl and s1_name_cl == t_name_cl) else 0.0

    # 3. Address similarities
    has_a1 = 1.0 if len(s1_addr_cl) > 0 else 0.0
    has_a2 = 1.0 if len(t_addr_cl) > 0 else 0.0
    f_addr_ratio = (fuzz.ratio(s1_addr_cl, t_addr_cl) / 100.0) if (has_a1 and has_a2) else 0.0
    f_addr_set = (fuzz.token_set_ratio(s1_addr_cl, t_addr_cl) / 100.0) if (has_a1 and has_a2) else 0.0
    exact_addr = 1.0 if (has_a1 and has_a2 and s1_addr_cl == t_addr_cl) else 0.0

    # 4. Number overlaps (PIN codes, house numbers)
    num_jaccard = (len(s1_nums & t_nums) / len(s1_nums | t_nums)) if (s1_nums and t_nums) else 0.0

    # 5. Length ratio
    l1 = len(s1_name_cl)
    l2 = len(t_name_cl)
    len_ratio = min(l1, l2) / max(l1, l2, 1)

    # 6. Source indicators
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
        f_addr_ratio,
        f_addr_set,
        exact_addr,
        num_jaccard,
        has_a1,
        has_a2,
        len_ratio,
        is_s2,
        is_s3,
        float(b_score),
        float(b_rank),
    ]


def train_pipeline(
    data_dir: str = "dataset",
    model_save_path: str = "models/matching_model.pkl",
    train_sample: int = 25000,
    val_sample: int = 5000,
):
    """
    End-to-end training and threshold tuning pipeline.
    Trains LightGBM classifier on candidate pairs and selects optimal threshold for Macro F_0.5.
    """
    print("=" * 70)
    print("  TRAINING ENTITY MATCHING MODEL (LIGHTGBM + RAPIDFUZZ)")
    print("=" * 70)

    train_s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    train_s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    train_s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")

    total_s1_needed = train_sample + val_sample
    print(f"Loading {total_s1_needed} S1 records from {train_s1_path}...")
    s1_df = pl.scan_csv(train_s1_path, separator="\t").limit(total_s1_needed).collect()

    s1_ids = set(s1_df["entity_id"])
    print(f"Loading ground truth for {len(s1_ids)} S1 records...")
    gt_df = pl.scan_csv(gt_path, separator="\t").filter(pl.col("source1_entity_id").is_in(s1_ids)).collect()

    gt_map: Dict[str, Set[str]] = {}
    matched_target_ids: Set[str] = set()
    for r in gt_df.iter_rows(named=True):
        m = r["matched_entity_ids"]
        cands = set(m.split(",")) if (m and str(m).strip()) else set()
        gt_map[r["source1_entity_id"]] = cands
        matched_target_ids.update(cands)

    # Split train and validation
    train_s1_df = s1_df[:train_sample]
    val_s1_df = s1_df[train_sample:]

    # Load targets: all matched targets + random sample of ~400k targets
    print(f"Loading target pool (matched targets + background targets)...")
    s2_df = pl.scan_csv(train_s2_path, separator="\t").filter(
        (pl.col("entity_id").is_in(matched_target_ids)) | (pl.int_range(0, pl.len()) < 250000)
    ).collect()
    s3_df = pl.scan_csv(train_s3_path, separator="\t").filter(
        (pl.col("entity_id").is_in(matched_target_ids)) | (pl.int_range(0, pl.len()) < 250000)
    ).collect()

    all_targets = pl.concat([s2_df, s3_df], how="vertical").unique(subset=["entity_id"])
    print(f"Target pool loaded: {len(all_targets)} records.")

    # Partition targets by country
    target_by_country = defaultdict(lambda: {"ids": [], "names": [], "addrs": []})
    target_lookup = {}

    for r in all_targets.iter_rows(named=True):
        t_id = r["entity_id"]
        c = normalize_country(r.get("country"))
        n_raw = r.get("business_name") or ""
        a_raw = r.get("business_address") or ""

        target_by_country[c]["ids"].append(t_id)
        target_by_country[c]["names"].append(n_raw)
        target_by_country[c]["addrs"].append(a_raw)

        n_cl = clean_text(n_raw, remove_legal=True)
        n_dom = clean_domain(n_raw)
        n_ns = n_dom.replace(" ", "")
        a_cl = clean_address(a_raw)
        nums = extract_address_numbers(a_raw)

        target_lookup[t_id] = {
            "name_cl": n_cl,
            "name_ns": n_ns,
            "addr_cl": a_cl,
            "nums": nums,
            "country": c,
        }

    # Build blocker for each country partition
    blockers = {}
    for c, data in target_by_country.items():
        print(f"Building blocker index for country: {c} ({len(data['ids'])} targets)...")
        b = FastCountryBlocker(max_candidates=12)
        b.fit_targets(data["ids"], data["names"], data["addrs"])
        blockers[c] = b

    # Generate training pairs
    print(f"Extracting pairwise training features from {len(train_s1_df)} S1 entities...")
    X_train = []
    y_train = []

    for r in tqdm(train_s1_df.iter_rows(named=True), total=len(train_s1_df), desc="Train pairs"):
        s1_id = r["entity_id"]
        c = normalize_country(r.get("country"))
        blocker = blockers.get(c)
        if not blocker:
            continue

        true_matches = gt_map.get(s1_id, set())
        s1_raw_name = r.get("business_name") or ""
        s1_raw_addr = r.get("business_address") or ""

        s1_cl = clean_text(s1_raw_name, remove_legal=True)
        s1_dom = clean_domain(s1_raw_name)
        s1_ns = s1_dom.replace(" ", "")
        s1_addr_cl = clean_address(s1_raw_addr)
        s1_nums = extract_address_numbers(s1_raw_addr)

        cands = blocker.query_entity(s1_raw_name, s1_raw_addr, max_candidates=12)
        for tid, b_score, b_rank in cands:
            t_info = target_lookup.get(tid)
            if not t_info:
                continue

            feat = extract_pair_features(
                s1_name_cl=s1_cl,
                s1_name_ns=s1_ns,
                s1_addr_cl=s1_addr_cl,
                s1_nums=s1_nums,
                t_name_cl=t_info["name_cl"],
                t_name_ns=t_info["name_ns"],
                t_addr_cl=t_info["addr_cl"],
                t_nums=t_info["nums"],
                target_id=tid,
                b_score=b_score,
                b_rank=b_rank,
            )
            label = 1 if tid in true_matches else 0
            X_train.append(feat)
            y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    print(f"Generated {len(X_train)} training pairs: {pos_count} positives, {neg_count} negatives.")

    # Train LightGBM model
    print("Fitting LightGBM classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"LightGBM fitted in {time.time() - t0:.2f}s.")

    # Validation evaluation & threshold tuning
    print(f"Evaluating and optimizing threshold on {len(val_s1_df)} validation S1 entities...")
    val_features = []
    val_index_pairs = []

    for r in tqdm(val_s1_df.iter_rows(named=True), total=len(val_s1_df), desc="Val pairs"):
        s1_id = r["entity_id"]
        c = normalize_country(r.get("country"))
        blocker = blockers.get(c)
        if not blocker:
            continue

        s1_raw_name = r.get("business_name") or ""
        s1_raw_addr = r.get("business_address") or ""

        s1_cl = clean_text(s1_raw_name, remove_legal=True)
        s1_dom = clean_domain(s1_raw_name)
        s1_ns = s1_dom.replace(" ", "")
        s1_addr_cl = clean_address(s1_raw_addr)
        s1_nums = extract_address_numbers(s1_raw_addr)

        cands = blocker.query_entity(s1_raw_name, s1_raw_addr, max_candidates=12)
        for tid, b_score, b_rank in cands:
            t_info = target_lookup.get(tid)
            if not t_info:
                continue

            feat = extract_pair_features(
                s1_name_cl=s1_cl,
                s1_name_ns=s1_ns,
                s1_addr_cl=s1_addr_cl,
                s1_nums=s1_nums,
                t_name_cl=t_info["name_cl"],
                t_name_ns=t_info["name_ns"],
                t_addr_cl=t_info["addr_cl"],
                t_nums=t_info["nums"],
                target_id=tid,
                b_score=b_score,
                b_rank=b_rank,
            )
            val_features.append(feat)
            val_index_pairs.append((s1_id, tid))

    val_X = np.array(val_features, dtype=np.float32)
    val_probs = model.predict_proba(val_X)[:, 1]

    s1_pred_map = defaultdict(list)
    for (s1_id, tid), prob in zip(val_index_pairs, val_probs):
        s1_pred_map[s1_id].append((tid, prob))

    thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    best_th = 0.65
    best_f05 = -1.0
    best_stats = {}

    for th in thresholds:
        preds_dict = {}
        gt_eval = {}
        for r in val_s1_df.iter_rows(named=True):
            s1_id = r["entity_id"]
            gt_eval[s1_id] = gt_map.get(s1_id, set())
            preds_dict[s1_id] = set([tid for tid, p in s1_pred_map.get(s1_id, []) if p >= th])

        metrics = evaluate_f05_macro(gt_eval, preds_dict)
        f05 = metrics["f05_macro"]
        prec = metrics["precision_macro"]
        rec = metrics["recall_macro"]
        print(f"  Threshold {th:.2f} -> Macro F0.5: {f05:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_th = th
            best_stats = metrics

    print("=" * 70)
    print(f"Optimal Threshold: {best_th:.2f}")
    print(f"Validation Macro F0.5 : {best_stats['f05_macro']:.4f}")
    print(f"Validation Precision  : {best_stats['precision_macro']:.4f}")
    print(f"Validation Recall     : {best_stats['recall_macro']:.4f}")
    print("=" * 70)

    # Save model and metadata
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    payload = {
        "model": model,
        "threshold": best_th,
        "feature_names": FEATURE_NAMES,
        "val_metrics": best_stats,
    }
    with open(model_save_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model and metadata successfully saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Matching Model.")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset root folder")
    parser.add_argument("--model-save-path", default="models/matching_model.pkl", help="Model output file path")
    parser.add_argument("--train-sample", type=int, default=25000, help="Number of S1 train entities")
    parser.add_argument("--val-sample", type=int, default=5000, help="Number of S1 validation entities")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        train_sample=args.train_sample,
        val_sample=args.val_sample,
    )
