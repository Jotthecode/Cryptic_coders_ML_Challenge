"""
ML Challenge 2026 - Business Entity Resolution
Exploratory Data Analysis (EDA) Script

Performs:
1. Dataset loading & shape inspection
2. Singletons vs. Multi-matches distribution analysis
3. Country distributions across training and test sources
4. Missing / null values analysis across all sources
5. Legal suffixes and common name tokens analysis
6. Address noise patterns & abbreviations
"""

import os
import re
import sys
from collections import Counter
import pandas as pd


def resolve_path(rel_path: str) -> str:
    """Find file from current working directory or relative project roots."""
    candidates = [
        rel_path,
        os.path.join("..", rel_path),
        os.path.join(os.path.dirname(__file__), "..", rel_path),
        os.path.join(os.path.dirname(__file__), "..", "..", "student_resource", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path


def main():
    print("=" * 70)
    print("      ML CHALLENGE 2026: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Load Data
    s1_path = resolve_path("dataset/train/train_source1.tsv")
    s2_path = resolve_path("dataset/train/train_source2.tsv")
    s3_path = resolve_path("dataset/train/train_source3.tsv")
    gt_path = resolve_path("dataset/train/train_ground_truth.tsv")

    print(f"\n[1] Loading training datasets...")
    print(f"  - Source 1: {s1_path}")
    print(f"  - Source 2: {s2_path}")
    print(f"  - Source 3: {s3_path}")
    print(f"  - Ground Truth: {gt_path}")

    train_s1 = pd.read_csv(s1_path, sep="\t")
    train_s2 = pd.read_csv(s2_path, sep="\t")
    train_s3 = pd.read_csv(s3_path, sep="\t")
    gt = pd.read_csv(gt_path, sep="\t")

    print(f"\n--- Record Counts ---")
    print(f"S1 count: {len(train_s1):,}")
    print(f"S2 count: {len(train_s2):,}")
    print(f"S3 count: {len(train_s3):,}")
    print(f"Ground Truth count: {len(gt):,}")

    # 2. Analyze Singletons vs. Multi-matches
    print(f"\n[2] Analyzing Singletons vs. Multi-matches...")
    gt["matched_count"] = gt["matched_entity_ids"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() != "" else 0
    )

    singletons = (gt["matched_count"] == 0).sum()
    print(f"Singletons (Entities with 0 matches): {singletons:,} / {len(gt):,} ({singletons/len(gt)*100:.2f}%)")
    print("\nMatch Count Distribution:")
    vc = gt["matched_count"].value_counts().sort_index()
    for count, n_entities in vc.items():
        pct = (n_entities / len(gt)) * 100
        print(f"  {count} matches: {n_entities:8,d} entities ({pct:6.2f}%)")

    # 3. Check Countries
    print(f"\n[3] Country Distribution...")
    print("Train Countries in S1:", train_s1["country"].value_counts().to_dict())
    print("Train Countries in S2:", train_s2["country"].value_counts().to_dict())
    print("Train Countries in S3:", train_s3["country"].value_counts().to_dict())

    # Check test country distribution
    test_s1_path = resolve_path("dataset/test/test_source1.tsv")
    if os.path.exists(test_s1_path):
        test_s1 = pd.read_csv(test_s1_path, sep="\t", usecols=["entity_id", "country"])
        print("Test Countries in S1 :", test_s1["country"].value_counts().to_dict())
        del test_s1

    # 4. Missing/Null Values Analysis
    print(f"\n[4] Missing / Null Values Analysis...")
    for name, df in [("Source 1", train_s1), ("Source 2", train_s2), ("Source 3", train_s3)]:
        nulls = df.isnull().sum()
        null_report = {col: f"{count} ({count/len(df)*100:.2f}%)" for col, count in nulls.items()}
        print(f"  {name} nulls: {null_report}")

    gt_nulls = gt["matched_entity_ids"].isnull().sum()
    print(f"  Ground Truth 'matched_entity_ids' nulls (explicit NaNs / singletons): {gt_nulls:,} ({gt_nulls/len(gt)*100:.2f}%)")

    # 5. Legal Suffixes & Abbreviations in Business Names
    print(f"\n[5] Legal Suffixes & Common Tokens in Business Names...")
    # Sample 100k names for fast token frequency analysis
    sample_names = train_s1["business_name"].dropna().sample(min(100000, len(train_s1)), random_state=42)
    token_counter = Counter()
    for name in sample_names:
        cleaned = re.sub(r"[^\w\s]", " ", str(name).lower())
        token_counter.update(cleaned.split())

    legal_terms = ["llc", "inc", "ltd", "corp", "corporation", "pvt", "limited", "co", "company", "services", "enterprises"]
    print("Frequency of top legal suffixes / entity words in Source 1 sample (100k):")
    for term in legal_terms:
        print(f"  - '{term}': {token_counter[term]:,} occurrences")

    print("\nTop 15 most frequent overall tokens in business names:")
    for tok, cnt in token_counter.most_common(15):
        print(f"  - {tok:15s}: {cnt:,}")

    # 6. Address Patterns & Common Abbreviations
    print(f"\n[6] Address Analysis...")
    sample_addrs = train_s1["business_address"].dropna().sample(min(100000, len(train_s1)), random_state=42)
    addr_token_counter = Counter()
    for addr in sample_addrs:
        cleaned = re.sub(r"[^\w\s]", " ", str(addr).lower())
        addr_token_counter.update(cleaned.split())

    addr_abbrevs = ["rd", "road", "st", "street", "ave", "avenue", "dr", "drive", "blvd", "lane", "ln", "ct", "hwy", "apt", "suite", "ste", "near", "opp"]
    print("Frequency of address abbreviation terms in Source 1 sample (100k):")
    for term in addr_abbrevs:
        print(f"  - '{term}': {addr_token_counter[term]:,} occurrences")

    # 7. Sample Matching Ground Truth Pairs
    print(f"\n[7] Sample Ground Truth Matches (Demonstrating Noise Patterns)...")
    # Find records with 1 or 2 matches
    multi_gt = gt[gt["matched_count"] > 0].head(3)
    s2_lookup = train_s2.set_index("entity_id").to_dict(orient="index")
    s3_lookup = train_s3.set_index("entity_id").to_dict(orient="index")
    s1_lookup = train_s1.set_index("entity_id").to_dict(orient="index")

    for _, row in multi_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"].split(",")
        s1_info = s1_lookup.get(s1_id, {})
        print(f"\n[Reference S1] ID: {s1_id}")
        print(f"  Name   : {s1_info.get('business_name')}")
        print(f"  Address: {s1_info.get('business_address')}")
        print(f"  Country: {s1_info.get('country')}")
        for mid in matched_ids:
            target_info = s2_lookup.get(mid) or s3_lookup.get(mid, {})
            print(f"  --> Match [{mid}]")
            print(f"      Name   : {target_info.get('business_name')}")
            print(f"      Address: {target_info.get('business_address')}")
            print(f"      Country: {target_info.get('country')}")

    print("\n" + "=" * 70)
    print("                      EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
