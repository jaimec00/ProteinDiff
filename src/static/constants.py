'''
hold constants that are reused throughout the model here
'''
from static.amber.parse_amber_lib import parse_amber_lib
import numpy as np

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

def aa_2_lbl(aa):
    if aa in aa_2_lbl_dict:
        return aa_2_lbl_dict[aa]
    else:
        return aa_2_lbl_dict["X"]

def lbl_2_aa(label): 
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

def seq_2_lbls(seq):
    '''
    converts a string of amino acids (one-letter codes) to a tensor of labels
    does this in vectorized fashion
    '''

    byte_idx = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)  # shape [len(s)], each in [0,255]

    # vectorized lookup
    labels = lut[byte_idx]  # shape [len(s)], dtype int64

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