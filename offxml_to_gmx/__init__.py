"""offxml_to_gmx - Convert SMILES to GROMACS parameters using OpenFF"""

from .core import generate_gmx_files, create_polymer, optimize_molecule_coordinates

__version__ = "0.1.0"
__all__ = ["generate_gmx_files", "create_polymer", "optimize_molecule_coordinates"]
