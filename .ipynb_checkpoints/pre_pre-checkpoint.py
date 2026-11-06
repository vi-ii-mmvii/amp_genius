#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from typing import List, Optional

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_RESIDUES = set("AILMFWV")

def is_valid_sequence(seq: str, min_len: int = 5, max_len: int = 50) -> bool:
    """Filter out non-canonical amino acids and extreme lengths."""
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    return (min_len <= len(seq) <= max_len) and set(seq).issubset(STANDARD_AMINO_ACIDS)

def calculate_net_charge(seq: str) -> float:
    """Calculate net charge at physiological pH ~7.4."""
    seq = seq.upper()
    pos = seq.count('K') + seq.count('R') + 0.1 * seq.count('H')
    neg = seq.count('D') + seq.count('E')
    return round(pos - neg, 1)

def derive_condition_tags(seq: str, is_amp: bool = False, is_acp: bool = False) -> List[str]:
    """Generate discrete control tokens for ProtGPT2 conditioning."""
    tags = []
    if is_amp: tags.append("[AMP]")
    if is_acp: tags.append("[ACP]")
        
    charge = calculate_net_charge(seq)
    if charge >= 4.0: tags.append("[CHARGE_HIGH_POS]")
    elif charge > 0: tags.append("[CHARGE_POS]")
    elif charge == 0: tags.append("[CHARGE_NEUT]")
    else: tags.append("[CHARGE_NEG]")
        
    length = len(seq)
    if length < 20: tags.append("[LEN_SHORT]")
    elif length <= 35: tags.append("[LEN_MEDIUM]")
    else: tags.append("[LEN_LONG]")
    return tags

def format_protgpt2_prompt(seq: str, tags: List[str]) -> str:
    """Format entry for ProtGPT2 fine-tuning."""
    tag_prefix = "".join(tags)
    return f"<|startoftext|>\n{tag_prefix}\n{seq}\n<|endoftext|>"
