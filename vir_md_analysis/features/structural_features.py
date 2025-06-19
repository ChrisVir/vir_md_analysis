import MDAnalysis as mda
import mdtraj as md
import pandas as pd
from Bio.SeqUtils import seq1

# Maximum surface area for each amino acid type from Tien et al # (2013)
# "Maximum Allowed Solvent Accessibilites of Residues in Proteins
# https://doi.org/10.1371/journal.pone.0080635
# Note: These values are used to calculate the relative surface area (RSA) of amino acids.
# RSA is calculated as the ratio of the surface area of an amino acid in a protein to its maximum surface area.
# The maximum surface area values are used to normalize the surface area of each amino acid in a protein.
# The values are in square angstroms (A^2). So need to divide by 100 to get square nanometers (nm^2).

max_aa_sasa = {'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 'C': 167.0, 'E': 223.0, 'Q': 225.0, 'G': 104.0,
               'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0, 'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0,
               'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0}


def get_topology_dataframe(traj: md.Trajectory, residues_only: bool = False) -> pd.DataFrame:
    """
    Get a DataFrame containing the topology information of a trajectory.

    Args:
    traj : mdtraj.Trajectory
        The trajectory object.
    residues_only : bool, optional
        If True, only include residue information, by default False.

    Returns:
        topology_df: pandas.DataFrame
            A DataFrame containing the topology information of the trajectory.
    """
    topology_df = (traj.topology.to_dataframe()[0]
                   .reset_index()
                   .rename(columns={'index': 'atomIndex'})
                   )

    if residues_only:
        cols = ['resSeq', 'resName', 'chainID', 'segmentID']
        topology_df = (topology_df.drop_duplicates(subset=cols)
                       [cols]
                       .reset_index(drop=True)
                       )

    return topology_df


def merge_hydrogen_bonds_with_topology(hbonds_df: pd.DataFrame,
                                       topology_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge hydrogen bonds DataFrame with topology DataFrame.

    Parameters
    ----------
    hbonds_df : pd.DataFrame
        DataFrame containing hydrogen bonds information.
    topology_df : pd.DataFrame
        DataFrame containing topology information of the residues.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with hydrogen bonds and topology information.
    """
    topo_cols = ['atomIndex', 'element', 'resSeq', 'resName', 'chainID', 'segmentID']
    hbonds_df = (hbonds_df.merge(topology_df[topo_cols], left_on='donor', right_on='atomIndex')
                 .drop(columns='atomIndex')
                 .rename(columns={col: f'donor_{col}' for col in topo_cols})
                 .merge(topology_df[topo_cols], left_on='acceptor', right_on='atomIndex')
                 .drop(columns='atomIndex')
                 .rename(columns={col: f'acceptor_{col}' for col in topo_cols})
                 )
    return hbonds_df


def calculate_rg_for_specific_region(u: mda.Universe, region: str = 'protein',
                                     region_name: str = 'protein') -> pd.DataFrame:
    """Selects a region of a universe and calculate radius of gyration

    Args:
        u (md.Universe): The universe object to analyze.
        region (str, optional): The region to select in MD Analysis selection language.
                                 Defaults to 'protein'.
        region_name (str, optional): The name of the region. Defaults to 'protein'.

    Returns:
        pd.DataFrame: dataframe with radius of gyration data
    """

    region = u.select_atoms(region)
    frames = []
    times = []
    radii = []

    for ts in u.trajectory:
        frames.append(ts.frame)
        times.append(u.trajectory.time)
        radii.append(region.radius_of_gyration())

    df = pd.DataFrame({
        'Frame': frames,
        'Time (ns)': times,
        'Radius of Gyration (A)': radii
    })
    df.insert(0, 'Region', region_name)

    return df


def compute_rg_on_specific_regions_with_mdtraj(traj: md.Trajectory, region: str = 'backbone'):
    """
    Calculate the radius of gyration for a specific region of the protein using MDTraj.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory object containing the simulation data.
    region : str
        The region of the protein to calculate the radius of gyration for.
        The regions can be any as defined by MDTRaj selection language.

    Returns
    -------
    rg_df : pd.DataFrame
        A DataFrame containing the time and radius of gyration for the specified region.
    """

    atoms = traj.topology.select(region)
    traj_atoms = traj.atom_slice(atoms)

    rg = md.compute_rg(traj_atoms)

    rg_df = pd.DataFrame({'Frame': len(rg),
                          'Time (ns)': traj.time,
                          'Radius of Gyration (A)': rg})

    rg_df.insert(0, 'Region', region)
    return rg_df


def select_mdtraj_atoms(traj, region='protein'):
    """
    Select atoms from a trajectory based on a selection string.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory object.
    region : str
        The selection string for atoms to include in the SASA calculation.


    Returns
    -------
    traj_atoms: md.Trajectory
        A trajectory object containing the selected atoms.
    """
    # Select atoms based on the region string
    atoms = traj.topology.select(region)
    return traj.atom_slice(atoms)


def compute_sasa(traj, region_map: pd.DataFrame = None, chains: list[str] = None,
                 relative: bool = False):
    """
    Compute the solvent accessible surface area (SASA) of a trajectory.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory object.
    region_map : pd.DataFrame
        A DataFrame containing the regions of interest for the SASA calculation.
    chains : list[str]
        A list of chain IDs to include in the SASA calculation.
        If None, all chains will be included.

    Returns
    -------
    sasa_df: pandas.DataFrame
        A DataFrame containing the SASA values for each frame in the trajectory.
    """

    if chains:
        # Select atoms based on the chain IDs
        region = ' or '.join([f'chainid {chainid}' for chainid in chains])
        traj = select_mdtraj_atoms(traj, region=region)

    results = md.shrake_rupley(traj, get_mapping=True, mode='residue')
    sasa_df = pd.DataFrame(results[0], columns=traj.topology.residues)

    if relative:
        aminos = [seq1(residue.name) for residue in traj.topology.residues]
        for aa, col in zip(aminos, sasa_df.columns):
            sasa_df[col] /= (max_aa_sasa[aa]/100)

    if region_map is not None:
        # Map the SASA values to the regions
        region_map = region_map.set_index('residue')
        sasa_df = pd.DataFrame(sasa_df, columns=region_map.index)
        sasa_df = sasa_df.rename(columns=region_map.to_dict())

    return sasa_df, results[1]


def compute_hydrogen_bonds(traj: md.Trajectory, region: str | None = None, method: str = 'baker',
                           freqfloat: float = 0.1, exclude_water: bool = True, periodic: bool = True,
                           sidechain_only: bool = False) -> pd.DataFrame:
    """ computes the hydrogen in a trajectory for a specific region using MDTraj.
    Args:
        traj (md.Trajectory): The trajectory object to analyze.
        region (str): The region of the protein to calculate the hydrogen bonds for.
                      The regions can be any as defined by MDTRaj selection language.
                      Optional, defaults to None, which means all atoms in the trajectory will be considered.
        method (str): The method to use for calculating hydrogen bonds. Can be 'baker', 'kabsh' or 'wernet'
        freqfloat (float, optional): The frequency of the trajectory in nanoseconds. Defaults to 0.1.
        exclude_water (bool, optional): Whether to exclude water molecules from the calculation. Defaults to True.
        periodic (bool, optional): Whether to use periodic boundary conditions. Defaults to True.
        sidechain_only (bool, optional): Whether to consider only sidechain atoms for hydrogen bond calculation.
                                         Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the hydrogen bond data for the specified region.
        Exact output format may vary based on the method used.
    """
    if region:
        traj = select_mdtraj_atoms(traj, region=region)

    if method not in ['baker', 'kabsh', 'wernet']:
        raise ValueError("Method must be one of 'baker', 'kabsh', or 'wernet'.")
    if method == 'baker':
        # calculate hydrogen bonds using the Baker-Hubbard method
        hbonds = md.baker_hubbard(traj)

        # Organize the hydrogen bond data into a DataFrame
        hbonds_df = pd.DataFrame(hbonds, columns=['donor', 'hydrogen', 'acceptor'])
        topo_df = get_topology_dataframe(traj)
        hbonds_df = merge_hydrogen_bonds_with_topology(hbonds_df, topo_df)

        return hbonds_df

    elif method == 'kabsh':
        pass
    elif method == 'wernet':
        pass

    return None
