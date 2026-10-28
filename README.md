# offxml_to_gmx

Convert SMILES representations (polymers or molecules) to GROMACS parameter files using OpenFF force fields.

This tool generates the same files as the original molGenerator notebook:
- `.gro` structure file
- `.rtp` residue topology parameters
- `_atomtypes.atp` atom type masses
- `_ffnonbonded.itp` nonbonded parameters

## Usage

```bash
offxml_to_gmx "SMILES_STRING" --name MOL_NAME --n-monomers 1 --output-dir ./output
```

Or import as a module:
```python
from offxml_to_gmx import generate_gmx_files
generate_gmx_files(smiles="...", mol_name="TAF", n_monomers=1)
```

## Supported polymer SMILES with attachment points

Use `[*:1]` and `[*:2]` for polymer growth points (like in the notebook). For a single molecule without growth points, set `n_monomers=1` and ensure the SMILES represents the final molecule.
