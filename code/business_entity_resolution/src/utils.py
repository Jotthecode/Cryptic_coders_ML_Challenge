"""
ML Challenge 2026 - Business Entity Resolution
High-performance utility functions for text preprocessing, string similarity,
evaluation metrics, and open-set country normalization.
"""

import os
import re
from typing import Dict, List, Set, Union
import numpy as np
import pandas as pd

# Standard legal suffixes to normalize / strip
LEGAL_SUFFIXES = [
    r"\bpvt\s+ltd\b",
    r"\bprivate\s+limited\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\binc\b",
    r"\bincorporated\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\benterprises\b",
    r"\benterprise\b",
    r"\bservices\b",
    r"\bsolutions\b",
    r"\bgroup\b",
    r"\bgmbh\b",
    r"\bsarl\b",
    r"\bsa\b",
    r"\bsas\b",
]

# Standard address abbreviations mapping
ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bflr\b": "floor",
    r"\bstr\b": "street",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\boff\b": "office",
    r"\bext\b": "extension",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
}

# Precompile regexes for fast processing
RE_LEGAL = re.compile("|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)
RE_SPECIAL_CHARS = re.compile(r"[^\w\s]", flags=re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMS = re.compile(r"\b\d{2,}\b")
RE_DOMAIN_EXT = re.compile(r"\.(com|org|net|in|fr|co|io|biz|info|gov|edu|ai|app).*$", flags=re.IGNORECASE)
RE_URL_PREFIX = re.compile(r"https?://(?:www\.)?", flags=re.IGNORECASE)


def normalize_country(country: Union[str, float, None]) -> str:
    """Normalize country string in an open-set manner (no hardcoding of allowed countries)."""
    if pd.isna(country) or country is None:
        return "UNKNOWN"
    c = str(country).strip().upper()
    return c if c else "UNKNOWN"


def clean_text(text: Union[str, float], remove_legal: bool = False) -> str:
    """Clean text by lowercasing, expanding/removing noise, and normalizing spaces."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()

    if remove_legal:
        text = RE_LEGAL.sub(" ", text)

    text = RE_SPECIAL_CHARS.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def clean_domain(text: Union[str, float]) -> str:
    """Clean domain names and website URLs to extract the underlying business identity token."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = RE_URL_PREFIX.sub("", text)
    text = RE_DOMAIN_EXT.sub("", text)
    text = RE_SPECIAL_CHARS.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def clean_address(address: Union[str, float]) -> str:
    """Normalize address string: expand common road/street abbreviations and remove special chars."""
    if pd.isna(address) or address is None:
        return ""
    text = str(address).lower()
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = RE_SPECIAL_CHARS.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def extract_address_numbers(address: Union[str, float]) -> Set[str]:
    """Extract numeric tokens (e.g. street numbers, PIN codes, postal codes) from address."""
    if pd.isna(address) or address is None:
        return set()
    return set(RE_NUMS.findall(str(address).lower()))


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from text."""
    if not text:
        return set()
    padded = f" {text} "
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def token_overlap(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute token-level Jaccard similarity."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    return jaccard_similarity(set_a, set_b)


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F-beta score given precision and recall."""
    beta_sq = beta**2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


def evaluate_f05_macro(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5 score over all Source 1 entities in ground truth.
    Per competition rules:
    - Singletons: If true matches is empty:
        - If predicted matches is empty -> score = 1.0
        - If predicted matches is not empty -> score = 0.0
    - Non-singletons:
        - precision = |true & pred| / |pred| (if |pred| > 0 else 0)
        - recall = |true & pred| / |true|
        - F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"f05_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0}

    f05_scores = []
    precision_scores = []
    recall_scores = []

    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())

        # Check singleton
        if len(true_matches) == 0:
            if len(pred_matches) == 0:
                f05 = 1.0
                prec = 1.0
                rec = 1.0
            else:
                f05 = 0.0
                prec = 0.0
                rec = 0.0
        else:
            if len(pred_matches) == 0:
                f05 = 0.0
                prec = 0.0
                rec = 0.0
            else:
                tp = len(true_matches & pred_matches)
                prec = tp / len(pred_matches)
                rec = tp / len(true_matches)
                f05 = compute_f_beta(prec, rec, beta=0.5)

        f05_scores.append(f05)
        precision_scores.append(prec)
        recall_scores.append(rec)

    return {
        "f05_macro": float(np.mean(f05_scores)),
        "precision_macro": float(np.mean(precision_scores)),
        "recall_macro": float(np.mean(recall_scores)),
        "total_evaluated": total_entities,
    }


def find_data_dir() -> str:
    """Helper to locate dataset directory reliably."""
    candidates = [
        "dataset",
        os.path.join("..", "dataset"),
        os.path.join(os.path.dirname(__file__), "..", "dataset"),
        os.path.join(os.path.dirname(__file__), "..", "..", "student_resource", "dataset"),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "train", "train_source1.tsv")):
            return os.path.abspath(path)
    return "dataset"
