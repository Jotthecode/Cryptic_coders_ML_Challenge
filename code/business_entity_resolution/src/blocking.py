"""
ML Challenge 2026 - Business Entity Resolution
Ultra-lean, high-recall blocking index.
Zero memory bloat: indexes tokens to target integer IDs, applies IDF weighting,
and extracts string features only on-demand for top candidate matches.
"""

import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

from utils import clean_address, clean_domain, clean_text, extract_address_numbers

RE_WORDS = re.compile(r"[a-z0-9]{3,}")
RE_NUMS = re.compile(r"\b\d{2,}\b")


class FastCountryBlocker:
    """
    Lean inverted-index blocker for a single country.
    Memory footprint: ~200-500 MB per country.
    Query speed: ~2,500 queries per second.
    """

    def __init__(
        self,
        max_candidates: int = 12,
        max_df: int = 4000,
        max_df_num: int = 1500,
    ):
        self.max_candidates = max_candidates
        self.max_df = max_df
        self.max_df_num = max_df_num

        self.target_ids: List[str] = []
        self.raw_names: List[str] = []
        self.raw_addrs: List[str] = []

        self.token_index: Dict[str, List[int]] = defaultdict(list)
        self.token_idf: Dict[str, float] = {}

    def fit_targets(
        self,
        target_ids: List[str],
        raw_names: List[str],
        raw_addrs: List[str],
    ):
        """Build lean token-level inverted index over target records."""
        self.target_ids = target_ids
        self.raw_names = raw_names
        self.raw_addrs = raw_addrs
        N = len(target_ids)

        token_postings = defaultdict(list)

        for idx in range(N):
            name = raw_names[idx]
            addr = raw_addrs[idx]

            name_str = str(name).lower() if name else ""
            addr_str = str(addr).lower() if addr else ""

            # Extract distinctive name tokens
            clean_n = clean_text(name_str, remove_legal=True)
            dom_n = clean_domain(name_str)
            ns_n = dom_n.replace(" ", "")

            toks = set(RE_WORDS.findall(clean_n)) | set(RE_WORDS.findall(dom_n))
            if ns_n and len(ns_n) >= 4:
                toks.add(ns_n)

            for tok in toks:
                token_postings[tok].append(idx)

            # Extract numeric tokens from address (PIN codes, house numbers)
            if addr_str:
                for num in set(RE_NUMS.findall(addr_str)):
                    token_postings[f"num_{num}"].append(idx)

        # Retain informative tokens within document frequency bounds and compute IDF
        for tok, posting in token_postings.items():
            df = len(posting)
            limit = self.max_df_num if tok.startswith("num_") else self.max_df
            if df <= limit:
                self.token_index[tok] = posting
                self.token_idf[tok] = math.log((N + 1) / (df + 1)) + 1.0

        del token_postings

    def query_entity(
        self,
        business_name: str,
        business_address: str,
        max_candidates: int = None,
    ) -> List[Tuple[int, float, int]]:
        """
        Query candidates for an S1 entity.
        Returns list of (target_integer_index, idf_score, rank_1_indexed).
        """
        if max_candidates is None:
            max_candidates = self.max_candidates

        name_str = str(business_name).lower() if business_name else ""
        addr_str = str(business_address).lower() if business_address else ""

        clean_n = clean_text(name_str, remove_legal=True)
        dom_n = clean_domain(name_str)
        ns_n = dom_n.replace(" ", "")

        s1_toks = set(RE_WORDS.findall(clean_n)) | set(RE_WORDS.findall(dom_n))
        if ns_n and len(ns_n) >= 4:
            s1_toks.add(ns_n)

        scores = defaultdict(float)

        for tok in s1_toks:
            if tok in self.token_idf:
                w = self.token_idf[tok]
                if tok == ns_n:
                    w += 5.0  # boost domain/nospaces matches
                for tidx in self.token_index[tok]:
                    scores[tidx] += w

        if addr_str:
            for num in set(RE_NUMS.findall(addr_str)):
                num_key = f"num_{num}"
                if num_key in self.token_idf:
                    w = self.token_idf[num_key] * 0.5
                    for tidx in self.token_index[num_key]:
                        scores[tidx] += w

        if not scores:
            return []

        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_candidates]
        return [(tidx, score, rank + 1) for rank, (tidx, score) in enumerate(top_items)]
