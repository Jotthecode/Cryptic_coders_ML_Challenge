# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** QuantumQuants  
**Team Members:** Jot Ajmani  
**Submission Date:** September 2026  

---

## 1. Executive Summary

This solution presents a scalable, high-precision Machine Learning pipeline for large-scale Business Entity Resolution across multiple heterogeneous, noisy data sources comprising over 24 million total records. By exploiting strict country-partitioning invariants (0 cross-country matches across 346,000+ evaluated pairs), our approach deploys an ultra-lean, multi-pass inverted-index blocker (combining frequency-capped name tokens, domain/URL normalizations, and address numeric anchors) to achieve a 99.74% candidate recall ceiling while pruning candidates to a tight budget of 12 per Source 1 entity. Downstream, a C++ accelerated RapidFuzz feature extraction engine feeds a LightGBM gradient boosted model directly tuned to maximize macro $F_{0.5}$ (weighting precision 2× over recall), achieving **0.8797 Macro $F_{0.5}$** with **93.11% precision** on holdout validation.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) on the 12.5M training records and 11.7M test records revealed crucial structural properties:
1. **Strict Country Boundaries:** Ground-truth analysis over 345,997 matched entity pairs revealed exactly **zero** cross-country matches. Partitioning by country strictly eliminates cross-country candidate comparisons with zero loss of recall.
2. **Open-Set Country Support:** While training records contain only `US` and `India`, the test set introduces `France` (~259k S1 records, ~1.43M target records). The pipeline dynamically discovers and processes country string labels case-insensitively without any hardcoded filtering.
3. **Noise Patterns:**
   - *Name Variations:* Heavy use of abbreviations and inconsistent legal suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`), website URLs as trade names (e.g. `maurewilliamscolombier.com`), character transpositions, and noise/garbled text.
   - *Address Variations:* Missing address fields (~3.3% of S2/S3 targets have `None` addresses), road/street abbreviations (`Rd` vs. `Road`), landmark-based Indian addresses, and rearranged components.
4. **Metric Mechanics ($F_{0.5}$ Macro):**
   - $F_{0.5}$ penalizes false positives twice as heavily as false negatives. Singletons (5.58% in train) score 1.0 when predicted empty, but drop to 0.0 on any false positive merge. Hence, conservative high-precision classification is essential.

### 2.2 Solution Strategy

**Approach Type:** Multi-Pass Inverted Index Blocking + Gradient Boosted Decision Tree (LightGBM) Pairwise Linkage with Macro $F_{0.5}$ Threshold Optimization.  
**Core Innovation:** 
1. *Zero-Copy Low-Memory Streaming:* By caching country targets into columnar Arrow/Parquet tables and indexing targets into lightweight integer arrays, the pipeline processes 11.7 million test records within $< 1.5$ GB peak RAM on an ordinary 16 GB machine.
2. *Domain and Numeric Anchoring:* Special handling for domain names and numeric tokens (PIN codes, street numbers) recovers records where names are corrupted but addresses match, or where addresses are missing but domain names match.

---

## 3. Candidate Generation (Blocking)

Candidate generation reduces the search space from $O(N \times M) \approx 1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairwise comparisons down to $\le 12$ candidates per Source 1 entity:

- **Blocking keys used:**
  1. *Significant Name Tokens:* Cleaned, legal-suffix-stripped words of length $\ge 3$.
  2. *Domain / Concatenated Strings:* Normalized URLs stripping protocols and extensions (`.com`, `.in`, `.fr`), concatenated into continuous strings to bridge domain trade names.
  3. *Address Numeric Anchors:* Regex-extracted numeric tokens (PIN codes, house numbers) indexed with country-level IDF weights.
  4. *Document Frequency Filtering:* Tokens appearing in $> 4,000$ targets are filtered out to prevent common corporate terms (e.g., "enterprises", "solutions", "holdings") from polluting posting lists.
- **Candidate pairs generated:** Average of 8.8 candidates per Source 1 entity (strictly capped at max 12 candidates).
- **How true matches were preserved:**
  - Evaluated on ground truth: the multi-pass blocker achieved a **99.74% recall ceiling**.
  - Dual anchor guarantee: if an address is missing, name tokens and domain keys capture the entity; if a name is corrupted, address numeric anchors bridge the match.

---

## 4. Matching Model

### Features Used (18 Total):
1. **Name Similarity Features:**
   - `f_name_ratio`: RapidFuzz Levenshtein ratio on cleaned names.
   - `f_name_partial`: Partial string ratio (handles sub-entity name containment).
   - `f_name_sort`: Token Sort Ratio (handles word order transpositions).
   - `f_name_set`: Token Set Ratio (robust against extra descriptive words).
   - `dom_exact` & `dom_contains`: Binary indicators for domain-name equivalence.
   - `exact_name`: Strict equality on cleaned legal-stripped names.
2. **Address Similarity Features:**
   - `f_addr_ratio`: Levenshtein ratio on normalized addresses.
   - `f_addr_set`: Token set ratio on normalized addresses.
   - `exact_addr`: Strict equality on cleaned address strings.
   - `num_jaccard`: Jaccard overlap of numeric tokens (PIN codes, house numbers).
   - `has_addr1` & `has_addr2`: Missingness indicators for address fields.
3. **Structural & Contextual Features:**
   - `len_ratio`: Ratio of character lengths.
   - `is_s2` & `is_s3`: Source partition indicator flags.
   - `b_score`: Inverted-index accumulated IDF score from blocking.
   - `b_rank`: Candidate rank (1-indexed) within the entity's candidate pool.

**Model Type:** LightGBM Classifier (200 estimators, learning rate 0.08, num_leaves 31, subsample 0.8).  
**Threshold Selection Method:** Direct grid search over validation split evaluating macro $F_{0.5}$. The optimal threshold was identified at **$T = 0.50$**, yielding maximum macro $F_{0.5}$ while maintaining $> 93\%$ precision.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.8797** on 6,000 validation entities.
- **Macro Precision:** **0.9311** (93.11%).
- **Macro Recall:** **0.7891** (78.91%).
- **Threshold Sensitivity Analysis:**
  | Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall |
  | :--- | :--- | :--- | :--- |
  | 0.40 | 0.8783 | 0.9281 | 0.7908 |
  | **0.50** | **0.8797** | **0.9311** | **0.7891** |
  | 0.60 | 0.8790 | 0.9323 | 0.7845 |
  | 0.70 | 0.8796 | 0.9357 | 0.7797 |
  | 0.80 | 0.8778 | 0.9379 | 0.7702 |

- **Common False Positives (Wrong Merges):**
  - Chain branches or franchises sharing identical corporate names and city names with differing local street numbers.
- **Common False Negatives (Missed Matches):**
  - Extreme abbreviation pairs (e.g. 3-letter acronyms without expansion) where both address and full name had non-overlapping tokens.

---

## 6. Conclusion

By combining strict country partitioning, an ultra-fast IDF-weighted multi-pass inverted index, and an optimized LightGBM matcher with C++ string similarities, our solution delivers state-of-the-art entity resolution performance. It satisfies all candidate set compactness criteria, guarantees strict candidate-subset constraints, seamlessly supports open-set countries, and executes end-to-end within minimal memory and runtime budgets.

---

## Appendix

### A. Code Artefacts
All runnable code is located under `code/business_entity_resolution/`:
- `src/utils.py`: Text normalization, legal suffix regexes, address cleaning, domain extraction, evaluation metrics.
- `src/blocking.py`: `FastCountryBlocker` inverted index engine with IDF scoring.
- `src/train_matching.py`: Feature extraction, LightGBM training, and threshold optimization.
- `src/predict.py`: Chunked streaming test set inference generating `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `requirements.txt`: Pinned dependencies (`polars`, `rapidfuzz`, `lightgbm`, `scikit-learn`, `pyarrow`).

Reproduce with:
```bash
python src/predict.py --test-dir dataset/test --output-dir output --model-path models/matching_model.pkl
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### B. Additional Results
- **Throughput:** ~1,000 S1 entities per second during feature extraction and inference.
- **Peak RAM:** Kept strictly $< 1.5$ GB across all partitions.
- **Candidate Size:** Mean candidates per entity $\approx 8.8$ (strict max 12).
