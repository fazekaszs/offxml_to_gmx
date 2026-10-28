"""Command line interface for offxml_to_gmx."""

import argparse
import sys
from .core import generate_gmx_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMILES to GROMACS parameters using OpenFF"
    )
    parser.add_argument("smiles", nargs="?", help="SMILES string")
    parser.add_argument("--smiles", dest="smiles_opt", help="SMILES string (alternative)")
    parser.add_argument("-n", "--name", default="MOL", help="Molecule name/prefix (default: MOL)")
    parser.add_argument("--mol-name", dest="mol_name_opt", help="Molecule name (alternative)")
    parser.add_argument("-m", "--n-monomers", type=int, default=1, help="Number of monomer units (default: 1)")
    parser.add_argument("-o", "--output-dir", default="mol_generator_out", help="Output directory (default: mol_generator_out)")
    parser.add_argument("--run-xtb", action="store_true", help="Run xTB optimization")
    parser.add_argument("--ff", default="openff-2.2.0.offxml", help="OpenFF force field (default: openff-2.2.0.offxml)")
    parser.add_argument("--box-size", type=float, default=5.0, help="Box size in nm (default: 5.0)")

    args = parser.parse_args()

    smiles = args.smiles_opt or args.smiles
    mol_name = args.mol_name_opt or args.name

    if not smiles:
        parser.print_help()
        sys.exit(1)

    try:
        output_dir = generate_gmx_files(
            smiles=smiles,
            mol_name=mol_name,
            n_monomers=args.n_monomers,
            output_dir=args.output_dir,
            run_xtb=args.run_xtb,
            force_field=args.ff,
            box_size=args.box_size,
        )
        print(f"Files generated in {output_dir}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
