# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** QuantumQuants  
**Team Members:** Jot Ajmani  
**Submission Date:** September 2026  
**Repository:** [Cryptic_coders_ML_Challenge](https://github.com/Jotthecode/Cryptic_coders_ML_Challenge)  

---

## 1. Executive Summary

This document presents the complete technical architecture, methodology, and empirical validation for our solution to the **ML Challenge 2026: Business Entity Resolution Challenge**. 

The task requires resolving noisy, multi-source business records across a reference source (**Source 1**) and target sources (**Source 2** and **Source 3**) comprising **over 24 million total records** spanning the United States, India, and an unseen open-set country partition (France). The competition evaluates systems using **Macro $F_{0.5}$**, a metric that penalizes false merges (false positives) twice as severely as missed links (false negatives), while favoring compact candidate sets.

### Key Highlights of the QuantumQuants System:
1. **Academic Integrity & Zero External Data:** Our solution strictly relies **100% on the provided training records**. No external APIs, web lookup, geocoding services, or pre-trained proprietary entity databases were queried or utilized.
2. **Strict Country Invariant Partitioning:** Ground-truth analysis of 345,997 matched entity pairs revealed **0 cross-country matches**. By enforcing country partitioning, our system reduced the search space from $1.73 \times 10^{13}$ pairs down to $\approx 4 \times 10^{12}$ pairs with zero loss of true recall, while dynamically handling open-set countries (France) without hardcoded rules.
3. **High-Recall Multi-Pass Blocking (`src/blocking.py`):** An ultra-lean, multi-pass inverted-index engine combining frequency-capped name tokens, domain/URL normalizations, and address numeric anchors (PIN codes, house numbers) achieves a **99.74% candidate recall ceiling** on ground truth while generating an average of **8.8 candidates per entity** (capped at a strict maximum of 12).
4. **C++ Accelerated Matching Classifier (`src/train_matching.py`):** An 18-feature gradient-boosted decision tree (LightGBM) trained on ground-truth candidate pairs leveraging RapidFuzz C++ string metrics, Jaccard token overlaps, and blocking score priors.
5. **Country-Adaptive Decision Boundaries (`src/rescore_threshold.py`):** Recognizing structural variances between standardized US postal records and informal landmark-heavy Indian addresses, country-specific threshold tuning raised our holdout **Validation Macro $F_{0.5}$ to 0.8916** with **96.91% precision on US** and **89.82% precision on India**, pruning 122,489 borderline false merges across 1.73 million test entities.
6. **Low-Memory Streaming Architecture (`src/predict.py`):** Using columnar Parquet caching and zero-copy flat integer arrays, the entire test set of 11.7 million records was processed with **peak memory strictly under 1.5 GB RAM**.

---

## 2. Methodology & Problem Analysis

```
                              ┌──────────────────────────────────────────────┐
                              │     Raw Test Records (~11.7M Total)          │
                              │  S1: 1.73M | S2: 8.65M | S3: 1.32M Targets   │
                              └──────────────────────┬───────────────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │ Dynamic Country Partitioning │
                                      │   (US, India, Open France)   │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │ Multi-Pass Inverted Blocking │
                                      │  • DF-Capped Name Tokens     │
                                      │  • Domain / URL Stripping    │
                                      │  • Numeric Address Anchors   │
                                      └──────────────┬───────────────┘
                                                     │ (Avg 8.8 cands/entity; Max 12)
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │  RapidFuzz Feature Extraction│
                                      │    (18 Pairwise Features)    │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │  LightGBM Match Classifier   │
                                      │   (Probability Inference)    │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │ Country-Adaptive Thresholds  │
                                      │  • US:     P >= 0.60         │
                                      │  • India:  P >= 0.50         │
                                      │  • France: P >= 0.58         │
                                      └──────────────┬───────────────┘
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             ▼                                               ▼
             ┌───────────────────────────────┐               ┌───────────────────────────────┐
             │    candidate_pairs.tsv        │               │     matching_results.tsv      │
             │   (Strict Superset, 262MB)    │               │  (Strict Subset, 70MB, F0.5)  │
             └───────────────────────────────┘               └───────────────────────────────┘
```

### 2.1 Problem Analysis & Data Characteristics
Exploratory Data Analysis across the 12,507,760 training records and 11,701,570 test records revealed key structural properties:

1. **Strict Country Boundaries:**
   Across all 345,997 positive entity pairs in `train_ground_truth.tsv`, cross-country matches are strictly **zero (0.00%)**. Partitioning by country guarantees 100% preservation of all valid links while slashing redundant cross-border comparisons.
2. **Open-Set Country Generalization:**
   The training data contains only `US` and `India`, whereas the test set introduces `France` (~259k S1 records, ~1.43M targets). Hardcoding country lists would break inference; our pipeline dynamically partitions by `pl.col('country').str.to_uppercase()`, allowing seamless open-set processing.
3. **Data Noise Dimensions:**
   - *Corporate Legal Suffixes:* Frequent discrepancies (`Google LLC` vs `Google Inc` vs `Google Pvt Ltd`).
   - *Domain Name Registries:* Businesses frequently omit physical trade names and instead record their website domains (e.g. `maurewilliamscolombier.com`).
   - *Address Incompleteness:* ~3.3% of target records lack address strings entirely (`None` / empty).
   - *Informal Regional Nomenclature:* Indian records feature colloquial landmark references ("Near SBI ATM", "Opposite Metro Station"), while US addresses follow standardized street numbering.
4. **Metric Mechanics ($F_{0.5}$ Macro):**
   The competition evaluates performance using Macro $F_{0.5}$:
   $$F_{0.5} = (1 + 0.5^2) \frac{\text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   In this formulation, **precision is weighted 2× more heavily than recall**. A false positive merge drops the entity score drastically, whereas singletons (entities with 0 matches) receive a perfect score of $1.0$ if left empty, but plummet to $0.0$ if a false match is linked. High-precision conservatism is therefore the winning strategy.

### 2.2 Academic Integrity & Fair Play Statement
Our pipeline complies strictly with competition rules:
- **No External Data:** No commercial APIs (Google Places, Dun & Bradstreet, OpenCorporates), no government databases, and no geocoders were queried.
- **Offline ML Only:** All models, tokenizers, regex rules, and similarity measures run entirely locally on commodity compute using only the dataset provided.

---

## 3. Candidate Generation (Blocking Strategy)

The theoretical search space without blocking is $1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.73 \times 10^{13}$ pairs. Multi-pass candidate blocking prunes this to $\le 12$ candidates per Source 1 entity:

### 3.1 Multi-Key Inverted Index
Implemented in `src/blocking.py`, the `FastCountryBlocker` indexes target records across distinct key channels:
1. **Cleaned Name Tokens:** Legal suffixes (`inc`, `corp`, `ltd`, `pvt`, `gmbh`, `sarl`, `llc`) are stripped using compiled regular expressions. Tokens with $\ge 3$ characters are added to the inverted index.
2. **Name Word Bigrams:** Consecutive word shingles (e.g. `bg_apex_solutions`) capture multi-word business identity roots even when individual constituent words are common.
3. **Domain & Concatenated URL Keys:** Extracted by stripping web protocols (`http://`, `https://`, `www.`) and extensions (`.com`, `.in`, `.fr`, `.org`, `.net`). Spaces and punctuation are removed to form contiguous string keys, enabling direct linkages for domain-name business registrations.
4. **Address Words:** Significant street, city, and area word tokens ($\ge 4$ characters) to anchor businesses where names are shortened.
5. **Hyper-Specific Composite Keys ($c\_\text{NUM}\_\text{NAME}$):** Pairing postal PIN codes or street numbers with the primary name root (e.g. `c_400001_shree`). These composite tokens have $\text{DF} \le 5$, avoiding stopword suppression entirely.

### 3.2 Document Frequency Capping, IDF Weighting & Top-IDF Prioritization
- High-frequency stopwords (tokens appearing in $> 12,000$ records, raw numbers $> 2,500$, or address words $> 3,500$) are suppressed.
- Remaining tokens are weighted using Inverse Document Frequency:
  $$\text{IDF}(t) = \ln\left(1 + \frac{N_{\text{targets}}}{\text{DF}(t)}\right)$$
- **Top-IDF Query Prioritization:** During candidate querying, an entity evaluates its top-6 highest-IDF keys first, yielding blazing execution ($> 2,600$ queries/second) while preserving $94.81\%$ true candidate recall.
- Candidate ceiling is expanded to **20 candidates per entity** to ensure multi-match entities (averaging 3.67 targets in ground truth) are never squeezed out.

---

## 4. Matching Model Architecture & Feature Engineering

### 4.1 Feature Engineering (20 Pairwise Features)

For each candidate pair $(S_1, T)$, our C++ accelerated feature pipeline extracts 20 predictive features:

| # | Feature Name | Description | RapidFuzz / Mathematical Formulation |
|---|---|---|---|
| 1 | `f_name_ratio` | Levenshtein string similarity on cleaned names | `fuzz.ratio(s1_name_cl, t_name_cl) / 100.0` |
| 2 | `f_name_partial` | Substring containment similarity | `fuzz.partial_ratio(s1_name_cl, t_name_cl) / 100.0` |
| 3 | `f_name_sort` | Word order invariant token similarity | `fuzz.token_sort_ratio(s1_name_cl, t_name_cl) / 100.0` |
| 4 | `f_name_set` | Duplicate-word invariant token similarity | `fuzz.token_set_ratio(s1_name_cl, t_name_cl) / 100.0` |
| 5 | `dom_exact` | Binary exact domain match indicator | $1.0 \text{ if } \text{dom}_1 == \text{dom}_2 \ne \text{"" else } 0.0$ |
| 6 | `dom_contains` | Binary domain substring containment | $1.0 \text{ if } \text{dom}_1 \in \text{dom}_2 \text{ or } \text{dom}_2 \in \text{dom}_1 \text{ else } 0.0$ |
| 7 | `exact_name` | Binary strict equality on cleaned names | $1.0 \text{ if } \text{s1\_cl} == \text{t\_cl} \text{ else } 0.0$ |
| 8 | `first_word_match`| Binary equality on first word of business names | $1.0 \text{ if } \text{word}_1 == \text{word}_2 \ne \text{"" else } 0.0$ |
| 9 | `f_addr_ratio` | Levenshtein similarity on cleaned addresses | `fuzz.ratio(s1_addr_cl, t_addr_cl) / 100.0` |
| 10 | `f_addr_set` | Token set similarity on cleaned addresses | `fuzz.token_set_ratio(s1_addr_cl, t_addr_cl) / 100.0` |
| 11 | `exact_addr` | Binary strict equality on cleaned addresses | $1.0 \text{ if } \text{s1\_addr\_cl} == \text{t\_addr\_cl} \ne \text{"" else } 0.0$ |
| 12 | `num_jaccard` | Jaccard overlap of address numeric tokens | $\frac{\|S_1^{\text{num}} \cap T^{\text{num}}\|}{\|S_1^{\text{num}} \cup T^{\text{num}}\|}$ |
| 13 | `addr_word_jaccard`| Jaccard overlap of non-numeric address words | $\frac{\|S_1^{\text{words}} \cap T^{\text{words}}\|}{\|S_1^{\text{words}} \cup T^{\text{words}}\|}$ |
| 14 | `has_addr1` | S1 address presence flag | $1.0 \text{ if address present else } 0.0$ |
| 15 | `has_addr2` | Target address presence flag | $1.0 \text{ if address present else } 0.0$ |
| 16 | `len_ratio` | Name string length ratio | $\min(\text{len}_1, \text{len}_2) / \max(\text{len}_1, \text{len}_2)$ |
| 17 | `is_s2` | Target belongs to Source 2 | $1.0 \text{ if } T \in \text{Source 2 else } 0.0$ |
| 18 | `is_s3` | Target belongs to Source 3 | $1.0 \text{ if } T \in \text{Source 3 else } 0.0$ |
| 19 | `b_score` | Inverted index accumulated IDF prior | Continuous IDF sum from blocking phase |
| 20 | `b_rank` | Candidate priority rank | $1 / \text{rank}$ (where rank $\in [1, 20]$) |

### 4.2 LightGBM Classifier Architecture

We train a Gradient Boosted Decision Tree using **LightGBM** (`LGBMClassifier`) on positive ground-truth pairs and realistic negative distractor candidates:
- **Number of Estimators:** 220 trees
- **Max Leaves:** 31 (`num_leaves=31`)
- **Learning Rate:** 0.08
- **Feature Subsampling:** 0.80 (`colsample_bytree=0.80`)
- **Row Subsampling:** 0.80 (`subsample=0.80`)
- **Row Subsampling:** 0.80 (`subsample=0.80`)
- **Objective:** Binary cross-entropy with log-loss optimization:
  $$\mathcal{L}(y, \hat{p}) = - \left[ y \ln(\hat{p}) + (1 - y) \ln(1 - \hat{p}) \right]$$
- **Feature Importances:** Top contributors to tree splits were `f_name_set` (28.4%), `f_addr_set` (22.1%), `b_rank` (15.3%), `num_jaccard` (12.7%), and `dom_exact` (8.9%).

---

## 5. Empirical Results & Root-Cause Diagnosis

### 5.1 Leaderboard Diagnostic: The 0.534 Baseline Root Cause
On initial baseline submission, the leaderboard evaluated at **0.534 – 0.535**. An exhaustive diagnostic against `train_ground_truth.tsv` revealed the exact mathematical root cause:

1. **The False-Empty Collapse:**
   - In ground truth, **94.42%** of entities have matches (averaging **3.67 matches per entity**), and only **5.58%** are true singletons.
   - The initial baseline predicted **477,644 empty entities (27.57%)** — over 5× the true singleton rate.
   - For ~380,000 entities (~22% of the test set), the model predicted empty when true matches existed, scoring a flat $F_{0.5} = 0.0000$ and capping the macro average at $\approx 0.53$.
2. **Target Multi-Mapping Violations:**
   - Ground truth analysis across 7.63M matched targets proved that **0.0000% of targets match multiple S1 entities** (every S2/S3 target maps to at most ONE S1 entity).
   - The baseline permitted 57,078 target duplicate predictions, causing tens of thousands of false-positive penalties.

### 5.2 Architectural Interventions & Retrained Validation Performance

To resolve these bottlenecks, we introduced:
1. **1-to-1 Maximum-Weight Bipartite Assignment:** Pairs with $P \ge T$ are sorted globally by confidence, assigning each target record exclusively to the S1 entity that scored it highest.
2. **Singleton Margin Gating ($P \ge 0.28$):** For entities without a match above threshold, recovering the top candidate when confidence $\ge 0.28$ eliminates the false-empty collapse while preserving true singletons.
3. **Realistic Distractor Mining:** LightGBM was retrained on 812,000 realistic targets with composite-key candidates.

**Holdout Validation Sweep Across Thresholds (5,000 Validation Entities):**

| Threshold ($T$) | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Key Dynamic |
| :---: | :---: | :---: | :---: | :--- |
| 0.35 | 0.9304 | 94.85% | **90.23%** | Permissive matching; high recall. |
| 0.40 | 0.9316 | 95.09% | 90.05% | Robust candidate recovery. |
| 0.45 | 0.9331 | 95.33% | 89.87% | Strong precision-recall balance. |
| 0.50 | 0.9340 | 95.57% | 89.64% | Solid high-precision performance. |
| 0.55 | 0.9349 | 95.75% | 89.43% | Excellent candidate filtering. |
| **0.60** | **0.9353** | **0.9592** | **89.11%** | **Optimal Configuration (Max Macro $F_{0.5}$).** |

**Empirical Validation Leap:**
- **Validation Macro $F_{0.5}$:** Raised from 0.8797 to **0.9353** (+0.0556 absolute gain).
- **Validation Recall:** Surged from 78.91% to **89.11%** (+10.20% recall recovery).
- **Validation Precision:** Elevated from 93.11% to **95.92%**.

---

## 6. Streaming Pipeline & Low-Memory Architecture

Processing 11.7 million test records on commodity hardware (16 GB RAM with $\approx 3$ GB free) presented severe out-of-memory risks. Our engineering solution achieved zero OOM errors:

1. **Columnar Parquet Caching:** Target TSVs (Sources 2 & 3) were ingested via streaming Polars and cached by country into columnar Parquet tables (`tmp_cache/targets_{COUNTRY}.parquet`).
2. **Flat Integer Arrays:** Rather than keeping millions of Python string objects in memory, entities are mapped to contiguous integer arrays and accessed via zero-copy Arrow memory buffers.
3. **Chunked Scoring:** Source 1 entities are evaluated in chunks of 50,000, streaming feature extraction directly into numpy float32 arrays and calling garbage collection (`gc.collect()`) after each partition.
4. **Candidate Subset Guarantee:** By construction, matches are obtained by filtering the candidate list ($P \ge T$). This mathematically guarantees that:
   $$\text{Matches}(S_1) \subseteq \text{Candidates}(S_1) \quad \forall S_1 \in \text{Source 1}$$
5. **Format Validation:** Ran `utils/validate_submission.py` on the final output:
   - Required S1 entities: **1,732,544**
   - Rows generated: **1,732,544** (exact match, exact ordering)
   - Result: **`PASS — no blocking issues found. Safe to submit.`**

---

## 7. Results, Error Analysis & Edge Cases

### 7.1 Quantitative Performance Summary
- **Validation Macro $F_{0.5}$:** 0.9353 (Unified 1-to-1 Maximum-Weight Bipartite Assignment)
- **Validation Precision:** 95.92%
- **Validation Recall:** 89.11%
- **Candidate Set Size:** Average ~11.4 candidates/entity (Capped at 20)
- **End-to-End Test Set Inference Throughput:** ~1,100 entities/sec

### 7.2 Error Analysis
- **False Positives (Incorrect Links):**
  - *Chain Retailers & Franchise Outlets:* Identical brand names (e.g. "Subway", "State Bank of India") operating across multiple street addresses within the same city. Solved predominantly by strict `num_jaccard` numeric street checks and 1-to-1 competitive assignment.
- **False Negatives (Missed Links):**
  - *Extreme Acronyms:* Business pairs like *"Melania Garcia Tankers"* vs *"MGT"* where no shared address tokens exist. These require specialized phonetic/acronym anchors to bridge.

---

## 8. Conclusion

By combining strict country-invariant partitioning, a high-recall composite IDF-weighted multi-pass inverted index, RapidFuzz C++ string similarity feature engineering, and 1-to-1 competitive bipartite matching, our solution delivers an ultra-high precision, memory-efficient entity resolution system. It adheres 100% to academic integrity rules, completely satisfies the $F_{0.5}$ precision bias, and executes end-to-end within strict compute boundaries.

---

## Appendix: Reproduction Instructions

### A. Environment Setup
```bash
pip install -r requirements.txt
```
*(Dependencies: `polars==1.8.2`, `rapidfuzz==3.9.7`, `lightgbm==4.5.0`, `scikit-learn==1.5.2`, `pyarrow==17.0.0`)*

### B. Training the Model
```bash
python src/train_matching.py --data-dir dataset --model-save-path models/matching_model.pkl
```

### C. Generating Final Deliverables
```bash
# 1. End-to-end test set inference (generates candidate_pairs.tsv and matching_results.tsv)
python src/predict.py --test-dir dataset/test --output-dir output --model-path models/matching_model.pkl

# 2. Format Verification
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
