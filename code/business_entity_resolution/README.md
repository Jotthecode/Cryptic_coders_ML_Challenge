# ML Challenge 2026: Business Entity Resolution Pipeline

High-performance, scalable entity resolution pipeline built for large-scale multi-source record linkage across millions of noisy business records.

## 🚀 System Architecture

1. **Country Partitioning & Open-Set Handling:**
   - Evaluates records strictly partitioned by country (0 cross-country matches across ground truth), drastically slashing the search space with zero recall loss.
   - Dynamically discovers country labels (`US`, `India`, `France`, and open-set labels).
2. **Lean Multi-Pass Candidate Blocking (`src/blocking.py`):**
   - Multi-key inverted index on normalized business name tokens, domain/website tokens, and address numeric anchors (PIN codes, house numbers).
   - Frequency-capped IDF weighting to prune uninformative generic corporate suffix tokens.
   - Generates compact candidate sets (top 12 candidates per S1 entity) maximizing candidate reduction ratio.
3. **High-Precision Matching Model (`src/train_matching.py`):**
   - C++ accelerated string similarities via `rapidfuzz` (Token Sort Ratio, Token Set Ratio, Levenshtein Ratio, Partial Ratio).
   - LightGBM Gradient Boosted Decision Trees trained on ground-truth candidate pairs.
   - Threshold tuning directly optimizing macro $F_{0.5}$ (weights precision 2× over recall).
4. **Streaming Inference Pipeline (`src/predict.py`):**
   - Chunked disk streaming keeping peak memory $< 1.5$ GB across 11.7 million test records.
   - Ensures predicted matches are a strict mathematical subset of candidate pairs.
   - Exactly preserves row ordering of `test_source1.tsv`.

---

## 🛠️ Reproduction Instructions

### 1. Environment Installation

Ensure Python 3.8+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis (EDA)

Inspect dataset statistics, country distributions, missingness, and match distributions:

```bash
python src/eda.py
```

### 3. Model Training & Threshold Optimization

Train the LightGBM classifier on candidate pairs from the training split and optimize the classification threshold for macro $F_{0.5}$:

```bash
python src/train_matching.py --data-dir dataset --model-save-path models/matching_model.pkl --train-sample 30000 --val-sample 6000
```

### 4. Test Set Inference

Generate `matching_results.tsv` and `candidate_pairs.tsv` in the `output/` folder:

```bash
python src/predict.py --test-dir dataset/test --output-dir output --model-path models/matching_model.pkl
```

### 5. Format & Constraint Validation

Validate compliance with competition format rules:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### 6. (Optional) Country-Adaptive High-Precision Rescoring

Dynamically fine-tune decision boundaries by region (e.g. US=0.60, India=0.50, France=0.58) to eliminate false merges while protecting recall:

```bash
python src/rescore_threshold.py --us 0.60 --india 0.50 --france 0.58 --output output/matching_results_adaptive.tsv
```
