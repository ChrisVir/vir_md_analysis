"""
Constants used throughout the vir_md_analysis package.

This module contains reference values and constants used in feature calculations,
particularly for SASA (Solvent Accessible Surface Area) calculations.
"""

# Maximum surface area for each amino acid type from Tien et al (2013)
# "Maximum Allowed Solvent Accessibilites of Residues in Proteins"
# https://doi.org/10.1371/journal.pone.0080635
#
# Note: These values are used to calculate the relative surface area (RSA) of amino acids.
# RSA is calculated as the ratio of the surface area of an amino acid in a protein to its
# maximum surface area. The maximum surface area values are used to normalize the surface
# area of each amino acid in a protein.
# The values are in square angstroms (A^2). So need to divide by 100 to get square nanometers (nm^2).

MAX_AA_SASA = {
    'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0,
    'E': 223.0, 'Q': 225.0, 'G': 104.0, 'H': 224.0, 'I': 197.0,
    'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0,
    'S': 155.0, 'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0
}

# Default parameters for various feature calculations
DEFAULT_CUTOFFS = {
    'hydrogen_bond_distance': 0.35,  # nm
    'contact_distance': 0.45,        # nm
    'hydrogen_bond_threshold': -0.5,  # kcal/mol for Kabsch-Sander
}

# Common selection strings for MDTraj and PyTraj
SELECTION_STRINGS = {
    'protein': 'protein',
    'backbone': 'backbone',
    'sidechain': 'sidechain',
    'heavy': 'not element H',
}
