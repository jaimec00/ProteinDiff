# amber partial charges (ff19SB) are in the data/amber/amino19.lib file (taken from https://github.com/csimmerling/ff19SB_201907/blob/master/forcefield_files/amino19.lib), this script parses it

import re
import os

def parse_amber_lib():

    amber_folder = os.path.dirname(os.path.abspath(__file__))
    amber_lib_file = os.path.join(amber_folder, "amino19.lib")

    with open(amber_lib_file, 'r') as file:
        lines = file.readlines()

    charges = {}
    current_residue = None
    in_atoms_section = False

    for line in lines:
        # Check for residue definition start
        residue_match = re.match(r"!entry\.(\w+)\.unit\.atoms table", line)
        if residue_match:
            current_residue = residue_match.group(1)
            charges[current_residue] = {}
            in_atoms_section = True
            continue

        # Skip irrelevant sections
        if in_atoms_section:
            if line.startswith('!entry') or line.startswith('!'):
                in_atoms_section = False
                continue

            # Match atom lines (format: "AtomName" "AtomType" ...)
            atom_line_match = re.match(r'\s*"(\S+)"\s*"\S+"\s*\d+\s*\d+\s*\d+\s*\d+\s*\d+\s*(-?\d+\.\d+)', line)
            if atom_line_match:
                atom_name = atom_line_match.group(1)
                charge = float(atom_line_match.group(2))
                charges[current_residue][atom_name] = charge

    return charges

