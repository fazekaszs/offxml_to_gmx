from setuptools import setup, find_packages

setup(
    name="offxml_to_gmx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "openff-toolkit",
        "openff-nagl",
        "openff-nagl-models",
        "openff-interchange",
        "openff-units",
        "numpy",
        "ase",
        "tblite",
        "mace-torch",
        "pillow",
    ],
    entry_points={
        "console_scripts": [
            "offxml_to_gmx=offxml_to_gmx.cli:main",
        ],
    },
)
