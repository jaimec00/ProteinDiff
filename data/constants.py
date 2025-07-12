'''
hold constants that are reused throughout the model here
'''
from data.amber.parse_amber_lib import parse_amber_lib
import torch

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
alphabet = canonical_aas + noncanonical_aas + ['X']

def aa_2_lbl(aa):
    if aa in alphabet:
        return alphabet.index(aa)
    else:
        return alphabet.index("X")

def lbl_2_aa(label): 
    if label==-1:
        return "X"
    else:
        return alphabet[label]

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
amber_partial_charges = torch.zeros(len(alphabet), max(len(atom_list) for atom_list in atoms.values()))

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

# will be using the electric field 
coulomb_constant = 1 # need to figure out what it is in the correct units