"""
Static constants and data for Proteus.

This module provides amino acid alphabets, conversion dictionaries,
AMBER ff19SB partial charges, and atom orderings used throughout the model.
"""

from .constants import (
    # Amino acid conversion dictionaries
    three_2_one,
    one_2_three,

    # Amino acid alphabets
    canonical_aas,
    noncanonical_aas,
    special_chars,
    alphabet,

    # Label conversion dictionaries
    aa_2_lbl_dict,
    lbl_2_aa_dict,

    # Conversion functions
    aa_2_lbl,
    lbl_2_aa,
    seq_2_lbls,

    # AMBER partial charges and atom orderings
    amber_partial_charges,
    atoms,
    bb_atoms,
)

__all__ = [
    # Conversion dictionaries
    "three_2_one",
    "one_2_three",

    # Alphabets
    "canonical_aas",
    "noncanonical_aas",
    "special_chars",
    "alphabet",

    # Label dictionaries
    "aa_2_lbl_dict",
    "lbl_2_aa_dict",

    # Conversion functions
    "aa_2_lbl",
    "lbl_2_aa",
    "seq_2_lbls",

    # AMBER data
    "amber_partial_charges",
    "atoms",
    "bb_atoms",
]
