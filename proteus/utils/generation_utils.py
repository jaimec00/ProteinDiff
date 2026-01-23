"""Utilities for generating protein structures from model outputs."""

import torch
import numpy as np
from typing import Union, Optional
from pathlib import Path

from proteus.types import Float, Int, T
from proteus.static import residue_constants as rc


def atom14_to_pdb(
    coords: Float[T, "L 14 3"],
    labels: Int[T, "L"],
    atom_mask: Optional[Float[T, "L 14"]] = None,
    chain_id: str = "A",
    b_factors: Optional[Float[T, "L 14"]] = None,
) -> str:
    """
    Convert atom14 coordinates and amino acid labels to PDB format string.

    Args:
        coords: Atom coordinates in atom14 format (L, 14, 3)
        labels: Amino acid labels (0-19) for each residue (L,)
        atom_mask: Optional mask for valid atoms (L, 14). If None, inferred from labels.
        chain_id: Chain identifier for PDB output (default "A")
        b_factors: Optional per-atom B-factors (L, 14). If None, uses 0.0.

    Returns:
        PDB format string
    """
    # Convert tensors to numpy if needed
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if atom_mask is not None and isinstance(atom_mask, torch.Tensor):
        atom_mask = atom_mask.detach().cpu().numpy()
    if b_factors is not None and isinstance(b_factors, torch.Tensor):
        b_factors = b_factors.detach().cpu().numpy()

    L = coords.shape[0]

    # Get atom mask from residue constants if not provided
    if atom_mask is None:
        atom_mask = rc.restype_atom14_mask[labels]  # (L, 14)

    # Default b-factors to 0.0
    if b_factors is None:
        b_factors = np.zeros((L, 14), dtype=np.float32)

    pdb_lines = []
    atom_serial = 1

    for res_idx in range(L):
        aa_idx = int(labels[res_idx])

        # Handle unknown residue type
        if aa_idx >= len(rc.restypes):
            aa_1letter = 'X'
            res_name = 'UNK'
        else:
            aa_1letter = rc.restypes[aa_idx]
            res_name = rc.restype_1to3[aa_1letter]

        # Get atom names for this residue type
        atom_names = rc.restype_name_to_atom14_names[res_name]

        for atom_idx in range(14):
            # Skip if atom is not present
            if atom_mask[res_idx, atom_idx] < 0.5:
                continue

            atom_name = atom_names[atom_idx]
            if atom_name == '':
                continue

            x, y, z = coords[res_idx, atom_idx]
            b_factor = b_factors[res_idx, atom_idx]

            # Determine element from atom name
            element = atom_name[0]
            if len(atom_name) > 1 and atom_name[1].isalpha() and not atom_name[1].isdigit():
                # Check for two-letter elements (e.g., FE, ZN)
                # For standard amino acids, first char is always the element
                pass

            # Format atom name with proper spacing
            # PDB convention: 4 chars, left-justified if starting with digit
            if len(atom_name) < 4:
                if atom_name[0].isdigit():
                    formatted_atom_name = atom_name.ljust(4)
                else:
                    formatted_atom_name = ' ' + atom_name.ljust(3)
            else:
                formatted_atom_name = atom_name

            # Build ATOM record (PDB format)
            # Columns: 1-6=ATOM, 7-11=serial, 13-16=name, 17=altLoc, 18-20=resName,
            # 22=chain, 23-26=resSeq, 27=iCode, 31-38=x, 39-46=y, 47-54=z,
            # 55-60=occupancy, 61-66=tempFactor, 77-78=element
            pdb_line = (
                f"ATOM  {atom_serial:5d} {formatted_atom_name}{' ':1s}"
                f"{res_name:3s} {chain_id:1s}{res_idx + 1:4d}{' ':1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.0:6.2f}{b_factor:6.2f}          {element:>2s}"
            )
            pdb_lines.append(pdb_line)
            atom_serial += 1

    # Add TER record
    pdb_lines.append(f"TER   {atom_serial:5d}      {res_name:3s} {chain_id:1s}{L:4d}")
    pdb_lines.append("END")

    return '\n'.join(pdb_lines)


def save_pdb(
    coords: Float[T, "L 14 3"],
    labels: Int[T, "L"],
    output_path: Union[str, Path],
    atom_mask: Optional[Float[T, "L 14"]] = None,
    chain_id: str = "A",
    b_factors: Optional[Float[T, "L 14"]] = None,
) -> None:
    """
    Save atom14 coordinates and sequence to a PDB file.

    Args:
        coords: Atom coordinates in atom14 format (L, 14, 3)
        labels: Amino acid labels (0-19) for each residue (L,)
        output_path: Path to save the PDB file
        atom_mask: Optional mask for valid atoms (L, 14)
        chain_id: Chain identifier for PDB output (default "A")
        b_factors: Optional per-atom B-factors (L, 14)
    """
    pdb_string = atom14_to_pdb(coords, labels, atom_mask, chain_id, b_factors)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(pdb_string)


def labels_to_sequence(labels: Int[T, "L"]) -> str:
    """
    Convert amino acid labels to sequence string.

    Args:
        labels: Amino acid labels (0-19) for each residue (L,)

    Returns:
        Amino acid sequence as a string
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    sequence = []
    for idx in labels:
        idx = int(idx)
        if idx >= len(rc.restypes):
            sequence.append('X')
        else:
            sequence.append(rc.restypes[idx])

    return ''.join(sequence)


def sequence_to_labels(sequence: str) -> torch.Tensor:
    """
    Convert amino acid sequence string to labels.

    Args:
        sequence: Amino acid sequence (1-letter codes)

    Returns:
        Tensor of amino acid labels (0-19), unknown mapped to 20
    """
    indices = []
    for aa in sequence.upper():
        if aa in rc.restype_order:
            indices.append(rc.restype_order[aa])
        else:
            indices.append(rc.unk_restype_index)  # 20 for unknown

    return torch.tensor(indices, dtype=torch.long)


def atom14_to_cif(
    coords: Float[T, "L 14 3"],
    labels: Int[T, "L"],
    atom_mask: Optional[Float[T, "L 14"]] = None,
    chain_id: str = "A",
    b_factors: Optional[Float[T, "L 14"]] = None,
    entry_id: str = "PRED",
) -> str:
    """
    Convert atom14 coordinates and amino acid labels to mmCIF format string.

    Args:
        coords: Atom coordinates in atom14 format (L, 14, 3)
        labels: Amino acid labels (0-19) for each residue (L,)
        atom_mask: Optional mask for valid atoms (L, 14). If None, inferred from labels.
        chain_id: Chain identifier for CIF output (default "A")
        b_factors: Optional per-atom B-factors (L, 14). If None, uses 0.0.
        entry_id: Entry identifier for CIF header (default "PRED")

    Returns:
        mmCIF format string
    """
    # Convert tensors to numpy if needed
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if atom_mask is not None and isinstance(atom_mask, torch.Tensor):
        atom_mask = atom_mask.detach().cpu().numpy()
    if b_factors is not None and isinstance(b_factors, torch.Tensor):
        b_factors = b_factors.detach().cpu().numpy()

    L = coords.shape[0]

    # Get atom mask from residue constants if not provided
    if atom_mask is None:
        atom_mask = rc.restype_atom14_mask[labels]  # (L, 14)

    # Default b-factors to 0.0
    if b_factors is None:
        b_factors = np.zeros((L, 14), dtype=np.float32)

    cif_lines = []

    # Header
    cif_lines.append(f"data_{entry_id}")
    cif_lines.append("#")

    # Entry information
    cif_lines.append("_entry.id   " + entry_id)
    cif_lines.append("#")

    # Entity information (single polypeptide chain)
    cif_lines.append("_entity.id                         1")
    cif_lines.append("_entity.type                       polymer")
    cif_lines.append("_entity.pdbx_description           'Generated protein'")
    cif_lines.append("#")

    # Entity poly information
    sequence = labels_to_sequence(labels)
    cif_lines.append("_entity_poly.entity_id                      1")
    cif_lines.append("_entity_poly.type                           'polypeptide(L)'")
    cif_lines.append(f"_entity_poly.pdbx_seq_one_letter_code       {sequence}")
    cif_lines.append("#")

    # Struct_asym (chain mapping)
    cif_lines.append("_struct_asym.id                            " + chain_id)
    cif_lines.append("_struct_asym.entity_id                     1")
    cif_lines.append("#")

    # Atom site header
    cif_lines.append("loop_")
    cif_lines.append("_atom_site.group_PDB")
    cif_lines.append("_atom_site.id")
    cif_lines.append("_atom_site.type_symbol")
    cif_lines.append("_atom_site.label_atom_id")
    cif_lines.append("_atom_site.label_alt_id")
    cif_lines.append("_atom_site.label_comp_id")
    cif_lines.append("_atom_site.label_asym_id")
    cif_lines.append("_atom_site.label_entity_id")
    cif_lines.append("_atom_site.label_seq_id")
    cif_lines.append("_atom_site.pdbx_PDB_ins_code")
    cif_lines.append("_atom_site.Cartn_x")
    cif_lines.append("_atom_site.Cartn_y")
    cif_lines.append("_atom_site.Cartn_z")
    cif_lines.append("_atom_site.occupancy")
    cif_lines.append("_atom_site.B_iso_or_equiv")
    cif_lines.append("_atom_site.pdbx_formal_charge")
    cif_lines.append("_atom_site.auth_seq_id")
    cif_lines.append("_atom_site.auth_comp_id")
    cif_lines.append("_atom_site.auth_asym_id")
    cif_lines.append("_atom_site.auth_atom_id")
    cif_lines.append("_atom_site.pdbx_PDB_model_num")

    atom_serial = 1

    for res_idx in range(L):
        aa_idx = int(labels[res_idx])

        # Handle unknown residue type
        if aa_idx >= len(rc.restypes):
            res_name = 'UNK'
        else:
            aa_1letter = rc.restypes[aa_idx]
            res_name = rc.restype_1to3[aa_1letter]

        # Get atom names for this residue type
        atom_names = rc.restype_name_to_atom14_names[res_name]

        for atom_idx in range(14):
            # Skip if atom is not present
            if atom_mask[res_idx, atom_idx] < 0.5:
                continue

            atom_name = atom_names[atom_idx]
            if atom_name == '':
                continue

            x, y, z = coords[res_idx, atom_idx]
            b_factor = b_factors[res_idx, atom_idx]

            # Determine element from atom name (first character for standard amino acids)
            element = atom_name[0]

            # Build atom_site record
            # Format: group_PDB id type_symbol label_atom_id label_alt_id label_comp_id
            #         label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code
            #         Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv pdbx_formal_charge
            #         auth_seq_id auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num
            cif_line = (
                f"ATOM {atom_serial} {element} {atom_name} . {res_name} "
                f"{chain_id} 1 {res_idx + 1} ? "
                f"{x:.3f} {y:.3f} {z:.3f} 1.00 {b_factor:.2f} ? "
                f"{res_idx + 1} {res_name} {chain_id} {atom_name} 1"
            )
            cif_lines.append(cif_line)
            atom_serial += 1

    cif_lines.append("#")

    return '\n'.join(cif_lines)


def save_cif(
    coords: Float[T, "L 14 3"],
    labels: Int[T, "L"],
    output_path: Union[str, Path],
    atom_mask: Optional[Float[T, "L 14"]] = None,
    chain_id: str = "A",
    b_factors: Optional[Float[T, "L 14"]] = None,
    entry_id: str = "PRED",
) -> None:
    """
    Save atom14 coordinates and sequence to a mmCIF file.

    Args:
        coords: Atom coordinates in atom14 format (L, 14, 3)
        labels: Amino acid labels (0-19) for each residue (L,)
        output_path: Path to save the CIF file
        atom_mask: Optional mask for valid atoms (L, 14)
        chain_id: Chain identifier for CIF output (default "A")
        b_factors: Optional per-atom B-factors (L, 14)
        entry_id: Entry identifier for CIF header (default "PRED")
    """
    cif_string = atom14_to_cif(coords, labels, atom_mask, chain_id, b_factors, entry_id)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(cif_string)
