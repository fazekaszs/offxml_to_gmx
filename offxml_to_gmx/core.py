"""Core functionality for converting SMILES to GROMACS files."""

import copy
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from openff.toolkit import Molecule, Topology
from openff.nagl import GNNModel
from openff.nagl_models import list_available_nagl_models
from openff.units import unit
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.interchange import Interchange


# Load GNN model once at module level
try:
    GNN_MODEL_PATH = list_available_nagl_models()[-1]
    GNN_MODEL = GNNModel.load(GNN_MODEL_PATH)
except Exception:
    GNN_MODEL = None

ETKDG_PARAMS = AllChem.ETKDGv3()
ETKDG_PARAMS.randomSeed = 1994
ETKDG_PARAMS.maxIterations = 1000
ETKDG_PARAMS.useRandomCoords = True


def create_polymer(mol_smiles: str, n_members: int) -> Chem.Mol:
    """
    Create a polymer from a monomer SMILES with attachment points [*:1] and [*:2].
    
    Args:
        mol_smiles: SMILES string with [*:1] and [*:2] as attachment points
        n_members: Number of monomer units
        
    Returns:
        RDKit molecule object
    """
    # Create a monomer unit
    monomer = Chem.MolFromSmiles(mol_smiles)

    # Create an object names polymer, which will be grown progressively
    polymer = copy.deepcopy(monomer)

    # Calculate monomer size, start and end atom indices
    monomer_size = monomer.GetNumAtoms()
    start_atom_idx, end_atom_idx = [
        idx for idx, atom in enumerate(polymer.GetAtoms())
        if atom.GetSmarts() in {"[*:1]", "[*:2]"}
    ]

    # Close one end of the polymer object
    e_polymer = Chem.EditableMol(polymer)
    e_polymer.ReplaceAtom(start_atom_idx, Chem.Atom(1))
    polymer = e_polymer.GetMol()

    # Add new monomer units progressively
    for progress in range(n_members - 1):
        # Add a new monomer element
        new_polymer = Chem.CombineMols(polymer, monomer)

        # Determine the atom indices for the new bond
        atom1_idx = polymer.GetAtomWithIdx(
            progress * monomer_size + end_atom_idx
        ).GetNeighbors()[0].GetIdx()
        # this is the index of the atom to which [*:1] is bound in the newly added monomer
        atom2_idx = polymer.GetNumAtoms() + monomer.GetAtomWithIdx(
            start_atom_idx
        ).GetNeighbors()[0].GetIdx()

        # Convert the Mol object to EditableMol and add the new bond
        e_polymer = Chem.EditableMol(new_polymer)
        e_polymer.AddBond(
            atom1_idx, atom2_idx,
            Chem.BondType.SINGLE
        )

        # Convert the EditableMol object to Mol
        new_polymer = e_polymer.GetMol()
        polymer = new_polymer

    # Close the other end of the polymer object
    e_polymer = Chem.EditableMol(polymer)
    e_polymer.ReplaceAtom((n_members - 1) * monomer_size + end_atom_idx, Chem.Atom(1))
    polymer = e_polymer.GetMol()

    # Remove dummy atoms
    polymer = Chem.DeleteSubstructs(polymer, Chem.MolFromSmarts("[#0]"))

    return polymer


def optimize_molecule_coordinates(mol: Chem.Mol, run_xtb: bool = False) -> Chem.Mol:
    """
    Optimize molecule coordinates using ETKDG and MMFF.
    
    Args:
        mol: RDKit molecule
        run_xtb: Whether to run xTB optimization (requires additional setup)
        
    Returns:
        Optimized RDKit molecule
    """
    optimized_mol = Chem.AddHs(mol)
    Chem.SanitizeMol(optimized_mol)

    # Conformer generation and classical FF optimization
    AllChem.EmbedMolecule(optimized_mol, ETKDG_PARAMS)

    # Save intermediate files like in notebook
    with open("etkdg_embedded.mol", "w") as f:
        f.write(Chem.MolToMolBlock(optimized_mol))

    AllChem.MMFFOptimizeMolecule(optimized_mol)

    with open("mmff_optimized.mol", "w") as f:
        f.write(Chem.MolToMolBlock(optimized_mol))

    if run_xtb:
        try:
            from ase import Atoms
            from ase.optimize import FIRE2
            from tblite.ase import TBLite

            # Building the ASE system
            conf = optimized_mol.GetConformer()
            atomic_numbers, positions = list(), list()
            for atom in optimized_mol.GetAtoms():
                atom_position = conf.GetAtomPosition(atom.GetIdx())
                positions.append([atom_position.x, atom_position.y, atom_position.z])
                atomic_numbers.append(atom.GetAtomicNum())

            atoms = Atoms(numbers=atomic_numbers, positions=positions)

            # TBLite GFN2-xTB optimization
            atoms.calc = TBLite(
                method="GFN2-xTB",
                charge=Chem.GetFormalCharge(mol),
                verbosity=0,
                max_iterations=1000
            )
            FIRE2(
                atoms,
                logfile="tblite_gfn2_opt.log",
                trajectory="tblite_gfn2_opt.pckl",
                restart="mace_opt_restart.json"
            ).run(fmax=0.1, steps=1000)

            # Transfer of coordinates back to the RDKit world
            opt_positions = atoms.get_positions()
            for i in range(optimized_mol.GetNumAtoms()):
                x, y, z = opt_positions[i]
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))

            with open("xtb_optimized.mol", "w") as f:
                f.write(Chem.MolToMolBlock(optimized_mol))
        except Exception as e:
            print(f"Warning: xTB optimization failed: {e}")

    return optimized_mol


def generate_gmx_files(
    smiles: str = None,
    mol_name: str = "MOL",
    n_monomers: int = 1,
    output_dir: str = "mol_generator_out",
    run_xtb: bool = False,
    force_field: str = "openff-2.2.0.offxml",
    box_size: float = 5.0,
    monomer_smiles: str = None,
    save_images: bool = True
) -> str:
    """
    Generate GROMACS files from SMILES.
    
    Args:
        smiles: SMILES string - if it contains [*:1]/[*:2], it will be used as polymer monomer
        mol_name: Name/prefix for output files
        n_monomers: Number of monomer units for polymer
        output_dir: Directory to save output files
        run_xtb: Whether to run xTB optimization
        force_field: OpenFF force field to use
        box_size: Box size in nanometers
        monomer_smiles: Alternative parameter name for smiles (for compatibility)
        
    Returns:
        Path to output directory
    """
    if smiles is None and monomer_smiles is not None:
        smiles = monomer_smiles
    if smiles is None:
        raise ValueError("Either smiles or monomer_smiles must be provided")

    os.makedirs(output_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    try:
        # Create polymer or use molecule as-is
        if '[*:1]' in smiles or '[*:2]' in smiles:
            mol = create_polymer(smiles, n_monomers)
            monomer_for_img = Chem.MolFromSmiles(smiles)
        else:
            mol = Chem.MolFromSmiles(smiles)
            monomer_for_img = Chem.Mol(mol)
            if n_monomers > 1:
                print("Warning: No attachment points in SMILES, treating as single molecule regardless of n_monomers")

        Chem.SanitizeMol(mol)

        # Save polymer image like in notebook
        if save_images:
            try:
                from rdkit.Chem import Draw
                AllChem.Compute2DCoords(mol)
                Draw.MolToFile(mol, "polymer.png", size=(200 * max(1, n_monomers), 500))
                # Also save monomer image if it has attachment points
                if '[*:1]' in smiles or '[*:2]' in smiles:
                    try:
                        monomer_substructure = Chem.Mol(monomer_for_img)
                        monomer_substructure = Chem.AddHs(monomer_substructure)
                        monomer_substructure = Chem.DeleteSubstructs(monomer_substructure, Chem.MolFromSmarts("[#0]"))
                        AllChem.Compute2DCoords(monomer_substructure)
                        Draw.MolToFile(monomer_substructure, "monomer_substructure.png", size=(200, 500))
                    except Exception as e:
                        print(f"Warning: Could not save monomer image: {e}")
            except Exception as e:
                print(f"Warning: Could not save images: {e}")

        # Optimize coordinates
        optimized_mol = optimize_molecule_coordinates(mol, run_xtb=run_xtb)

        # Load into OpenFF and compute charges
        optimized_mol_openff = Molecule.from_rdkit(optimized_mol)

        if GNN_MODEL is not None:
            nagl_charges = GNN_MODEL.compute_properties(optimized_mol_openff, check_domains=True)
            optimized_mol_openff.partial_charges = nagl_charges["am1bcc_charges"] * unit.elementary_charge
        else:
            raise RuntimeError("GNN model not available. Cannot compute charges.")

        # Rename atoms to have unique names
        counters = {atom.symbol: 0 for atom in optimized_mol_openff.atoms}
        for atom in optimized_mol_openff.atoms:
            element = atom.symbol
            counters[element] += 1
            atom.name = f"{element}{counters[element]}"

        # Create interchange
        ff = ForceField(force_field)
        top = Topology.from_molecules([optimized_mol_openff])
        optimized_mol_interchange = Interchange.from_smirnoff(
            ff, top,
            charge_from_molecules=[optimized_mol_openff],
            box=np.array([box_size, box_size, box_size]) * unit.nanometer
        )

        # Build mappings
        vdw_to_atom_type = dict()
        for pot_idx, (pot_key, pot) in enumerate(optimized_mol_interchange["vdW"].potentials.items()):
            current_atom_type = f"{mol_name}_T{pot_idx}"
            vdw_to_atom_type[pot_key] = current_atom_type

        rtp_field_atoms = [
            [atom.name, None, None]
            for atom in optimized_mol_interchange.topology.molecule(0).atoms
        ]
        ffnonbonded_records = {atom_type: [None, None, None, None] for atom_type in vdw_to_atom_type.values()}

        # Assign atom types
        for atom_indices, potential_key in optimized_mol_interchange["vdW"].key_map.items():
            current_atom_type = vdw_to_atom_type[potential_key]
            current_atom_index = atom_indices.atom_indices[0]
            rtp_field_atoms[current_atom_index][1] = current_atom_type

            vdw_params = optimized_mol_interchange["vdW"].potentials[potential_key].parameters
            current_atom_num = optimized_mol_interchange.topology.molecule(0).atoms[current_atom_index].atomic_number
            current_atom_mass = optimized_mol_interchange.topology.molecule(0).atoms[current_atom_index].mass.m
            if ffnonbonded_records[current_atom_type][0] is None:
                ffnonbonded_records[current_atom_type][0] = current_atom_num
                ffnonbonded_records[current_atom_type][1] = current_atom_mass
                ffnonbonded_records[current_atom_type][2] = vdw_params["sigma"].m_as("nanometer")
                ffnonbonded_records[current_atom_type][3] = vdw_params["epsilon"].m_as("kilojoule / mol")
            else:
                err_msg = f"Multiple assignation found for atom idx {current_atom_index}!"
                assert ffnonbonded_records[current_atom_type][0] == current_atom_num, err_msg

        # Fill charges
        for atom_indices, potential_key in optimized_mol_interchange["Electrostatics"].key_map.items():
            charge = optimized_mol_interchange["Electrostatics"].potentials[potential_key].parameters["charge"].m
            rtp_field_atoms[atom_indices.this_atom_index][2] = charge

        # Build strings
        rtp_field_atoms_str = " [ atoms ]\n" + "\n".join(
            f"  {f1:10s} {f2:10s} {f3:10.6f} {idx+1:6d}"
            for idx, (f1, f2, f3) in enumerate(rtp_field_atoms)
        )

        atp_field_masses_str = "\n".join(
            f"{atom_type:10s} {f2:8.5f}"
            for atom_type, (_, f2, _, _) in ffnonbonded_records.items()
        )

        ffnonbonded_records_str = "\n".join(
            f"{atom_type:10s} {f1:6d} {f2:10.6f} {0.:6.4f} A {f3:.5e} {f4:.5e}"
            for atom_type, (f1, f2, f3, f4) in ffnonbonded_records.items()
        )

        # Bonds
        rtp_field_bonds = list()
        for atom_indices, potential_key in optimized_mol_interchange["Bonds"].key_map.items():
            current_potential = optimized_mol_interchange["Bonds"].potentials[potential_key].parameters
            rtp_field_bonds.append([
                rtp_field_atoms[atom_indices.atom_indices[0]][0],
                rtp_field_atoms[atom_indices.atom_indices[1]][0],
                current_potential["length"].m_as("nanometer"),
                current_potential["k"].m_as("kilojoule / mol / nanometer**2"),
            ])
        rtp_field_bonds_str = " [ bonds ]\n" + "\n".join(
            f"  {f1:10s} {f2:10s} {f3:10.6f} {f4:10.6f}"
            for f1, f2, f3, f4 in rtp_field_bonds
        )

        # Angles
        rtp_field_angles = list()
        for atom_indices, potential_key in optimized_mol_interchange["Angles"].key_map.items():
            current_potential = optimized_mol_interchange["Angles"].potentials[potential_key].parameters
            rtp_field_angles.append([
                rtp_field_atoms[atom_indices.atom_indices[0]][0],
                rtp_field_atoms[atom_indices.atom_indices[1]][0],
                rtp_field_atoms[atom_indices.atom_indices[2]][0],
                current_potential["angle"].m_as("degree"),
                current_potential["k"].m_as("kilojoule / mol / radian**2"),
            ])
        rtp_field_angles_str = " [ angles ]\n" + "\n".join(
            f"  {f1:10s} {f2:10s} {f3:10s} {f4:10.6f} {f5:10.6f}"
            for f1, f2, f3, f4, f5 in rtp_field_angles
        )

        # Dihedrals
        rtp_field_dihedrals = list()
        for atom_indices, potential_key in optimized_mol_interchange["ProperTorsions"].key_map.items():
            current_potential = optimized_mol_interchange["ProperTorsions"].potentials[potential_key].parameters
            rtp_field_dihedrals.append([
                rtp_field_atoms[atom_indices.atom_indices[0]][0],
                rtp_field_atoms[atom_indices.atom_indices[1]][0],
                rtp_field_atoms[atom_indices.atom_indices[2]][0],
                rtp_field_atoms[atom_indices.atom_indices[3]][0],
                current_potential["phase"].m_as("degree"),
                current_potential["k"].m_as("kilojoule / mol") / current_potential["idivf"].m,
                current_potential["periodicity"].m,
            ])
        rtp_field_dihedrals_str = " [ dihedrals ]\n" + "\n".join(
            f"  {f1:10s} {f2:10s} {f3:10s} {f4:10s} {f5:10.6f} {f6:10.6f} {f7:6d}"
            for f1, f2, f3, f4, f5, f6, f7 in rtp_field_dihedrals
        )

        # Impropers
        rtp_field_impropers = list()
        try:
            for atom_indices, potential_key in optimized_mol_interchange["ImproperTorsions"].key_map.items():
                current_potential = optimized_mol_interchange["ImproperTorsions"].potentials[potential_key].parameters
                rtp_field_impropers.append([
                    rtp_field_atoms[atom_indices.atom_indices[0]][0],
                    rtp_field_atoms[atom_indices.atom_indices[1]][0],
                    rtp_field_atoms[atom_indices.atom_indices[2]][0],
                    rtp_field_atoms[atom_indices.atom_indices[3]][0],
                    current_potential["phase"].m_as("degree"),
                    current_potential["k"].m_as("kilojoule / mol") / current_potential["idivf"].m,
                    current_potential["periodicity"].m,
                ])
        except Exception:
            pass
        rtp_field_impropers_str = " [ impropers ]\n" + "\n".join(
            f"  {f1:10s} {f2:10s} {f3:10s} {f4:10s} {f5:10.6f} {f6:10.6f} {f7:6d}"
            for f1, f2, f3, f4, f5, f6, f7 in rtp_field_impropers
        )

        bonded_types_definition = """
[ bondedtypes ]
; Col 1: Type of bond
; Col 2: Type of angles
; Col 3: Type of proper dihedrals
; Col 4: Type of improper dihedrals
; Col 5: Generate all dihedrals if 1, only heavy atoms of 0.
; Col 6: Number of excluded neighbors for nonbonded interactions
; Col 7: Generate 1,4 interactions between pairs of hydrogens if 1
; Col 8: Remove impropers over the same bond as a proper if it is 1
; bonds  angles  dihedrals  impropers all_dihedrals nrexcl HH14 RemoveDih
     1       1          9          4        1         3      1     0

"""

        full_rtp_str = (
            bonded_types_definition + f"[ {mol_name} ]\n" +
            rtp_field_atoms_str + "\n" +
            rtp_field_bonds_str + "\n" +
            rtp_field_angles_str + "\n" +
            rtp_field_dihedrals_str + "\n" +
            rtp_field_impropers_str
        )

        # Write files
        optimized_mol_interchange.to_gro(file_path=f"{mol_name}.gro")

        with open(f"{mol_name}.rtp", "w") as f:
            f.write(full_rtp_str)

        with open(f"{mol_name}_atomtypes.atp", "w") as f:
            f.write(atp_field_masses_str)

        with open(f"{mol_name}_ffnonbonded.itp", "w") as f:
            f.write(ffnonbonded_records_str)

        return os.path.abspath(output_dir)
    finally:
        os.chdir(original_cwd)
