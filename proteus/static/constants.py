'''
hold constants that are reused throughout the model here
'''
from proteus.static.amber.parse_amber_lib import parse_amber_lib
import numpy as np
from enum import StrEnum

# dict to convert amino acid three letter codes to one letter
three_2_one = {
    'ALA': 'A',  # Alanine
    'CYS': 'C',  # Cysteine
    'ASP': 'D',  # Aspartic acid
    'GLU': 'E',  # Glutamic acid
    'PHE': 'F',  # Phenylalanine
    'GLY': 'G',  # Glycine
    'HIS': 'H',  # Histidine
    'ILE': 'I',  # Isoleucine
    'LYS': 'K',  # Lysine
    'LEU': 'L',  # Leucine
    'MET': 'M',  # Methionine
    'ASN': 'N',  # Asparagine
    'PRO': 'P',  # Proline
    'GLN': 'Q',  # Glutamine
    'ARG': 'R',  # Arginine
    'SER': 'S',  # Serine
    'THR': 'T',  # Threonine
    'VAL': 'V',  # Valine
    'TRP': 'W',  # Tryptophan
    'TYR': 'Y',  # Tyrosine
}
# one to three letter
one_2_three = {one_letter: three_letter for three_letter, one_letter in three_2_one.items()}

# model only predicts canonical aas for now, but the input can be non-canonical (to-do)
canonical_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
noncanonical_aas = []
special_chars = ["X", "<mask>"]
alphabet = canonical_aas + noncanonical_aas + special_chars

aa_2_lbl_dict = {aa: idx for idx, aa in enumerate(alphabet)}
lbl_2_aa_dict = {idx: aa for aa, idx in aa_2_lbl_dict.items()}

def aa_2_lbl(aa: str) -> int:
    if aa in aa_2_lbl_dict:
        return aa_2_lbl_dict[aa]
    else:
        return aa_2_lbl_dict["X"]

def lbl_2_aa(label: int) -> str:
    if label in lbl_2_aa_dict:
        return lbl_2_aa_dict[label]
    else:
        return "X"

# 256-entry LUT for byte values -> label, used to vectorize seq to label
lut = np.full(256, aa_2_lbl("X"), dtype=np.int64) # X is default for unkown characters
for aa, lbl in aa_2_lbl_dict.items(): # populate the lut array, each index corresponds to a characters encoding, the value is the label
    byte = aa.encode('ascii')  # will raise if non-ASCII; use .encode('latin1') if needed
    if len(byte)==1: # skip mult character aa (e.g. <mask>)
        lut[byte[0]] = lbl

def seq_2_lbls(seq: str) -> np.ndarray:
    '''
    converts a string of amino acids (one-letter codes) to a tensor of labels
    does this in vectorized fashion
    '''

    byte_idx: np.ndarray = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)  # shape [len(s)], each in [0,255]

    # vectorized lookup
    labels: np.ndarray = lut[byte_idx]  # shape [len(s)], dtype int64

    return labels

# now parse the amber partial charges from ff19SB lib file into a dict of {aa3letter: {N: pc, CA: pc, ...}, ...}
raw_charges = parse_amber_lib()

# now need to define the order that pmpnn uses to define the atoms in their xyz tensor. same order as in PDBs, but the atom types vary per AA
bb_atoms = ["N", "CA", "C", "O"] # these are always the first four
atoms = {
    "ALA": bb_atoms + ["CB"],
    "CYS": bb_atoms + ["CB", "SG"],
    "ASP": bb_atoms + ["CB", "CG", "OD1", "OD2"],
    "GLU": bb_atoms + ["CB", "CG", "CD", "OE1", "OE2"],
    "PHE": bb_atoms + ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "GLY": bb_atoms, # glycine only has the backbone atoms
    "HIS": bb_atoms + ["CB", "CG", "CD2", "ND1", "CE1", "NE2"],
    "ILE": bb_atoms + ["CB", "CG1", "CG2", "CD1"],
    "LYS": bb_atoms + ["CB", "CG", "CD", "CE", "NZ"],
    "LEU": bb_atoms + ["CB", "CG", "CD1", "CD2"],
    "MET": bb_atoms + ["CB", "CG", "SD", "CE"],
    "ASN": bb_atoms + ["CB", "CG", "ND2", "OD1"],
    "PRO": bb_atoms + ["CB", "CG", "CD"],
    "GLN": bb_atoms + ["CB", "CG", "CD", "NE2", "OE1"],
    "ARG": bb_atoms + ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "SER": bb_atoms + ["CB", "OG"],
    "THR": bb_atoms + ["CB", "CG2", "OG1"],
    "VAL": bb_atoms + ["CB", "CG1", "CG2"],
    "TRP": bb_atoms + ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"],
    "TYR": bb_atoms + ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]
}

# initialize the amber pc tensor
amber_partial_charges = np.zeros((len(alphabet), max(len(atom_list) for atom_list in atoms.values())))

# loop through atom types in the raw amber dict and assign the pc to the tensor
for aa, atom_pcs in raw_charges.items():
    if aa == "HIE":
        aa = "HIS"  # amber does not include HIS, but HIE is most common form
    if aa not in atoms.keys(): continue # skip irrrelevant aas
    for atom, pc in atom_pcs.items():
        if atom not in atoms[aa]: continue # skip irrelevant atoms, i.e. hydrogens
        aa_idx = aa_2_lbl(three_2_one[aa])
        atom_idx = atoms[aa].index(atom)
        amber_partial_charges[aa_idx, atom_idx] = pc

# X is treated as glycine
amber_partial_charges[aa_2_lbl("X"), :] = amber_partial_charges[aa_2_lbl("G"), :]

class TrainingStage(StrEnum):
	VAE = "vae"
	DIFFUSION = "diffusion"


# Import and reorder residue constants from AlphaFold's residue_constants.py
# AlphaFold uses a different amino acid ordering (sorted by 3-letter codes alphabetically):
#   'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X'
# Our model uses canonical_aas ordering:
#   'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
# We need to reorder the arrays so that indexing by our labels gives correct values.

from proteus.static import residue_constants as rc

# AlphaFold's amino acid order (with X at index 20)
af2_restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
af2_order = {aa: i for i, aa in enumerate(af2_restypes)}

# Create mapping: our_label_index -> af2_index
# For each position in our alphabet, find the corresponding AF2 index
def _get_af2_index(aa: str) -> int:
    """Get AlphaFold2 index for an amino acid, defaulting to X (20) for unknown."""
    return af2_order.get(aa, 20)

# Build reorder indices: reorder_idx[our_label] = af2_index
reorder_idx = np.array([_get_af2_index(aa) for aa in alphabet], dtype=np.int64)

# Reorder the arrays - index with our label, get data for correct amino acid
# These arrays have shape [21, ...] where first dim is restype
restype_rigid_group_default_frame = rc.restype_rigid_group_default_frame[reorder_idx]  # [22, 8, 4, 4]
restype_atom14_rigid_group_positions = rc.restype_atom14_rigid_group_positions[reorder_idx]  # [22, 14, 3]
restype_atom14_to_rigid_group = rc.restype_atom14_to_rigid_group[reorder_idx]  # [22, 14]
restype_atom14_mask = rc.restype_atom14_mask[reorder_idx]  # [22, 14]

# chi_angles_mask has shape [20,4] or [21,4] - handle both cases
_chi_mask = np.array(rc.chi_angles_mask)
if _chi_mask.shape[0] == 20:
    # Add row for X (same as glycine - no chi angles)
    _chi_mask = np.vstack([_chi_mask, np.zeros(4)])
chi_angles_mask = _chi_mask[reorder_idx]  # [22, 4]