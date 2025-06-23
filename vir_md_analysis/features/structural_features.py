
import mdtraj as md
import pytraj as pt
import pandas as pd
from Bio.SeqUtils import seq1
import numpy as np
import scipy as sp

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


# ------------------------------------------------------------------------
# MDTraj Feature Extraction Functions
# ------------------------------------------------------------------------
# The following functions are designed to extract structural features from
# molecular dynamics trajectories using the MDTraj library. These features
# include:
#   - Topology extraction and merging with hydrogen bond data
#   - Radius of gyration calculations for specific regions
#   - Solvent accessible surface area (SASA) calculations
#   - Hydrogen bond analysis using various methods (Baker-Hubbard, Kabsch-Sander, Wernet-Nilsson)
#   - Utility functions for processing hydrogen bond results
#
# All functions are intended to work with MDTraj Trajectory objects and
# return results as pandas DataFrames for downstream analysis.
# ------------------------------------------------------------------------


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


def compute_radius_of_gyration_by_region(traj: md.Trajectory, region: str = 'backbone'):
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

    traj_atoms = select_mdtraj_atoms(traj, region=region)

    rg = md.compute_rg(traj_atoms)

    rg_df = pd.DataFrame({'Frame': len(rg),
                          'Time (ns)': traj.time,
                          'Radius of Gyration (A)': rg})

    rg_df.insert(0, 'Region', region)
    return rg_df


def compute_sasa(traj, region_map: pd.DataFrame = None,
                 chains: list[str] = None,
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


def identify_hydrogen_bonds(traj: md.Trajectory, region: str | None = None, method: str = 'baker',
                            freq: float = 0.1, exclude_water: bool = True, periodic: bool = True,
                            sidechain_only: bool = False, distance_cutoff: float = 0.35) -> pd.DataFrame:
    """ computes the hydrogen in a trajectory for a specific region using MDTraj.
    Args:
        traj (md.Trajectory): The trajectory object to analyze.
        region (str): The region of the protein to calculate the hydrogen bonds for.
                      The regions can be any as defined by MDTRaj selection language.
                      Optional, defaults to None, which means all atoms in the trajectory will be considered.
        method (str): The method to use for calculating hydrogen bonds. Can be 'baker', 'kabsh' or 'wernet'
        freq (float, optional): Threshold for fraction of frames a hydrogen bond must be present to be considered
                                stable.
        exclude_water (bool, optional): Whether to exclude water molecules from the calculation. Defaults to True.
        periodic (bool, optional): Whether to use periodic boundary conditions. Defaults to True.
        sidechain_only (bool, optional): Whether to consider only sidechain atoms for hydrogen bond calculation.
                                         Defaults to False.
        distance_cutoff (float, optional): Distance cutoff for hydrogen bonds in nanometers. Defaults to 0.35 nm.

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
        hbonds = md.baker_hubbard(traj, freq=freq, exclude_water=exclude_water,
                                  periodic=periodic, sidechain_only=sidechain_only,
                                  distance_cutoff=distance_cutoff)

        # Organize the hydrogen bond data into a DataFrame
        hbonds_df = pd.DataFrame(hbonds, columns=['donor', 'hydrogen', 'acceptor'])
        topo_df = get_topology_dataframe(traj)
        hbonds_df = merge_hydrogen_bonds_with_topology(hbonds_df, topo_df)

        return hbonds_df

    elif method == 'kabsh':
        # calculate hydrogen bonds using the Kabsch-Sander method
        hbond_matrices = md.kabsch_sander(traj)
        num_hbonds_per_frame = calculate_number_of_hydrogen_bonds_per_frame(hbond_matrices)
        return num_hbonds_per_frame

    elif method == 'wernet':
        hbonds = md.wernet_nilsson(traj, exclude_water=exclude_water, periodic=periodic,
                                   sidechain_only=sidechain_only)
        num_hbonds_per_frame = (wernet_nilsson_hbond_results_to_dataframes(hbonds)
                                .sum(axis=0))

        return num_hbonds_per_frame

    else:
        raise ValueError("Method must be one of 'baker', 'kabsh', or 'wernet'.")


def calculate_number_of_hydrogen_bonds_per_frame(ks_results: list[sp.sparse._csc.csc_matrix],
                                                 threshold: float = -0.5) -> pd.DataFrame:
    """
    Calculate the number of hydrogen bonds per frame from Kabsch-Sander results.

    Parameters
    ----------
    ks_results : list[scipy.sparse._csc.csc_matrix]
        List of Kabsch-Sander results matrices.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the number of hydrogen bonds per frame.
    """
    num_hbonds = [np.sum(res.todense() <= threshold) for res in ks_results]
    return pd.DataFrame({'Frame': range(len(num_hbonds)), 'Number of Hydrogen Bonds': num_hbonds})


def determine_residues_in_hydrogen_bonds(ks_results: list[sp.sparse.csc_matrix],
                                         threshold: float = -0.5,
                                         topology_df: pd.DataFrame | None = None,
                                         digits: int = 3) -> pd.DataFrame:
    """
    Determine the residues involved in hydrogen bonds from Kabsch-Sander results.

    Parameters
    ----------
    ks_results : list[scipy.sparse.csc_matrix]
        List of Kabsch-Sander results matrices.
    threshold : float, optional
        Threshold for determining hydrogen bonds, by default -0.5.
    topology_df : pd.DataFrame, optional
        DataFrame containing the topology information of the residues, by default None.
    If provided, it will be used to map residue indices to residue names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the residues involved in hydrogen bonds.
    """

    num_frames = len(ks_results)

    mean_energy = ks_results[0].copy().todense()
    for res in ks_results[1:]:
        mean_energy += res.todense()
    mean_energy /= num_frames

    mean_energy = mean_energy

    mean_energy_squared = (ks_results[0].copy().todense() ** 2)
    for res in ks_results[1:]:
        mean_energy_squared += res.todense() ** 2
    mean_energy_squared /= num_frames

    std_energy = np.sqrt(mean_energy_squared - mean_energy ** 2)

    number_frames = (ks_results[0].copy().todense() <= threshold)*1
    for res in ks_results[1:]:
        number_frames += (res.todense() <= threshold)*1
    number_frames = number_frames

    percent_frames = number_frames / num_frames * 100

    acceptor, donor = np.where(mean_energy <= threshold)

    hbonds_df = pd.DataFrame({'Acceptor': acceptor, 'Donor': donor,
                              'Mean Energy': [mean_energy[i, j] for i, j in zip(acceptor, donor)],
                              'Std Energy': [std_energy[i, j] for i, j in zip(acceptor, donor)],
                              'Number Frames': [number_frames[i, j] for i, j in zip(acceptor, donor)],
                              'Percent Frames': [percent_frames[i, j] for i, j in zip(acceptor, donor)]})

    if topology_df is not None:
        hbonds_df['AcceptorResidue'] = topology_df.loc[hbonds_df['Acceptor']]['resName'].values
        hbonds_df['DonorResidue'] = topology_df.loc[hbonds_df['Donor']]['resName'].values
    return hbonds_df.round(digits)


def wernet_nilsson_hbond_results_to_dataframes(ws_results: pd.DataFrame) -> list[pd.DataFrame]:
    """Convert Wernet-Nilsson hydrogen bond results to a list of DataFrames.

    Args:
        ws_results (pd.DataFrame): DataFrame containing Wernet-Nilsson hydrogen bond results.

    Returns:
        list[pd.DataFrame]: List of DataFrames, each representing hydrogen bond information for a specific frame.
    """
    dfs = []
    for res in ws_results:
        dfs.append(pd.DataFrame(res, columns=['donor', 'hydrogen', 'acceptor']))
    return dfs


def wernet_nilsson_hbonds_to_number_contacts_by_frame_df(ws_results: pd.DataFrame) -> pd.DataFrame:
    """Convert Wernet-Nilsson hydrogen bond results to a DataFrame with counts of contacts by frame.

    Args:
        ws_results (pd.DataFrame): DataFrame containing Wernet-Nilsson hydrogen bond results.

    Returns:
        pd.DataFrame: DataFrame with counts of contacts by frame.
    """
    frame_dict = {}
    n = len(ws_results)
    for i, res in enumerate(ws_results):
        res = pd.DataFrame(res, columns=['donor', 'hydrogen', 'acceptor'])
        for row in res.itertuples(index=False):
            frame_dict.setdefault(row, [0]*n)
            frame_dict[row][i] += 1
    df = pd.DataFrame.from_dict(frame_dict, orient='index')

    donor_hydrogen_acceptor = pd.DataFrame(dict(donor=pd.Series(df.index).apply(lambda x: x[0]),
                                           hydrogen=pd.Series(df.index).apply(lambda x: x[1]),
                                           acceptor=pd.Series(df.index).apply(lambda x: x[2])))

    contacts_by_frame_df = pd.concat([donor_hydrogen_acceptor, df.reset_index(drop=True)], axis=1)
    return contacts_by_frame_df.set_index(['donor', 'hydrogen', 'acceptor'])


def identify_stable_hydrogen_bonds_from_wernet_nilsson(hydrogens_df: pd.DataFrame,
                                                       topology_df: pd.DataFrame | None = None,
                                                       threshold: float = 0.1) -> pd.DataFrame:
    """
    Get stable hydrogen bonds from the Wernet-Nilsson results.

    Parameters
    ----------
    hydrogens_df : pd.DataFrame
        DataFrame containing the hydrogen bond results. Each row should represent a hydrogen bond,
        and each column should represent a frame, with binary (0/1) or fractional values indicating
        the presence of the bond in each frame.
    threshold : float, optional
        Minimum fraction of frames a hydrogen bond must be present to be considered stable, by default 0.1.

    Returns
    -------
    pd.DataFrame
    """
    fraction_present = hydrogens_df.mean(axis=1)
    stable_hbonds = (fraction_present[fraction_present >= threshold]
                     .reset_index()
                     .rename(columns={0: 'Fraction Frames'})
                     )

    fraction_frames = hydrogens_df.mean(axis=1)
    stable_hbonds = (fraction_frames[fraction_frames >= threshold]
                     .reset_index()
                     .rename(columns={0: 'Fraction Frames'})
                     )

    if topology_df is not None:
        stable_hbonds = merge_hydrogen_bonds_with_topology(stable_hbonds, topology_df)

    return stable_hbonds


def compute_stats_from_series(series: pd.Series, prefix: str, protein_name: str = 'protein') -> pd.DataFrame:
    """
    Compute statistics (mean, std, min, max) from a pandas Series.

    Parameters
    ----------
    series : pd.Series
        The input series to compute statistics from.
    prefix : str
        A prefix to use for the column names in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the computed statistics.
    """
    return pd.DataFrame({
        f'{prefix} Mean': [series.mean()],
        f'{prefix} Std': [series.std()],
        f'{prefix} Min': [series.min()],
        f'{prefix} Max': [series.max()]
    }, index=[protein_name])


def compute_rmsd_on_specific_regions(traj: md.Trajectory, region: str | None = 'backbone',
                                     parallel: bool = True, precentered: bool = False) -> pd.DataFrame:
    """
    Compute the root mean square deviation (RMSD) on specific regions of a trajectory.

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory object.
    region : str
        The selection string for atoms to include in the RMSD calculation.

    Returns
    -------
    rmsd_df: pandas.DataFrame
        A DataFrame containing the RMSD values for each frame in the trajectory.
    """
    if region:
        target = select_mdtraj_atoms(traj, region=region)
    else:
        target = traj

    rmsd = md.rmsd(target, target, frame=0, parallel=parallel, precentered=precentered)
    return pd.DataFrame({'Frame': range(len(rmsd)), 'RMSD (nm)': rmsd})


def compute_rmsf_on_specific_regions(traj: md.Trajectory, region: str | None = 'backbone',
                                     parallel: bool = True, precentered: bool = False) -> pd.DataFrame:
    """
    Compute the root mean square fluctuation (RMSF) on specific regions of a trajectory.

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory object.
    region : str
        The selection string for atoms to include in the RMSF calculation.

    Returns
    -------
    rmsf_df: pandas.DataFrame
        A DataFrame containing the RMSF values for each frame in the trajectory.
    """
    if region:
        target = select_mdtraj_atoms(traj, region=region)
    else:
        target = traj

    # Compute the RMSF
    rmsf = md.rmsf(target, target, frame=0, parallel=parallel, precentered=precentered)

    return pd.DataFrame({'Frame': range(len(rmsf)), 'RMSF (nm)': rmsf})


def compute_contacts_on_specific_regions(traj: md.Trajectory, region: str | None = None,
                                         cutoff: float = 0.45, contacts: str = 'all',
                                         scheme: str = 'closest-heavy') -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """
    Compute the number of contacts in a specific region of a trajectory.
    Args:
        traj (md.Trajectory): The trajectory object.
        region (str, optional): The selection string for atoms to include in the contact calculation.
            If None, all atoms will be included. Defaults to None.
        cutoff (float, optional): The distance cutoff for contacts in nanometers. Defaults to 0.45.
        contacts (str, optional): The type of contacts to compute. Defaults to 'all'.
        scheme (str, optional): The scheme for computing contacts. Defaults to 'closest-heavy'.
    Returns
        num_contacts (pd.Series): An array containing the number of contacts for each frame.
        distances (np.ndarray): An array containing the distances between atom pairs.
        atom_pairs (np.ndarray): An array containing the atom pairs involved in the contacts.
    """

    # Select atoms based on the specified region
    if region:
        traj = select_mdtraj_atoms(traj, region=region)
    else:
        traj = traj

    # Compute contacts for the selected region
    distances, atom_pairs = md.compute_contacts(traj, contacts=contacts, scheme=scheme, periodic=True)
    num_contacts = pd.Series((distances <= cutoff).sum(axis=1))

    return num_contacts, distances, atom_pairs


# ------------------------------------------------------------------------
# PyTraj Feature Extraction Functions
# ------------------------------------------------------------------------

def identify_chains_from_pytraj_trajectory(traj: pt.Trajectory, chain_names: list[str] | None) -> list[int]:
    """
    Identify chains in a PyTraj trajectory.

    Parameters
    ----------
    traj : pytraj.Trajectory
        The trajectory object.
    chain_names : list[str] | None
        A list of chain names to identify. If None, all chains will be identified.

    Returns
    -------
    indices : list[int]
        A list of indices representing the start of each chain.
    """
    chains = {}
    if chain_names is None:
        chain_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split()
    else:
        chain_names.append('Other')
    chainid = 0
    chain_name = chain_names[chainid]
    chain = []
    for i, res in enumerate(traj.top.residues):
        first_atom = traj.top[res.first_atom_index]
        if first_atom.chain != chainid:
            chains[chain_name] = chain
            chainid += 1
            chain_name = chain_names[chainid]
            chain = []
        chain.append(i+1)  # +1 to convert from 0-based index to 1-based index

    chains[chain_name] = chain  # add the last chain
    return chains


def select_chains(traj: pt.Trajectory, select_chains: list[str], chains: dict[str, list[int]]) -> pt.Trajectory:
    """
    Select chains from a PyTraj trajectory.

    Parameters
    ----------
    traj : pytraj.Trajectory
        The trajectory object.
    select_chains : list[str]
        A list of chain names to select.
    chains : dict[str, list[int]]
        A dictionary mapping chain names to lists of residue indices.

    Returns
    -------
    pt.Trajectory
        A trajectory object containing the selected chains.
    """
    residues = []
    for chain in select_chains:
        if chain in chains:
            chain_residues = chains[chain]
            residues.append([chain_residues[0], chain_residues[-1]])  # select first and last residue of the chain
    residue_query = ':' + ','.join([f'{c[0]}-{c[1]}' for c in residues])
    return traj[residue_query]


def calculate_s2_parameter(traj: pt.Trajectory, chains: list[str] | None = None,
                           chains_dict: dict[str, list[int]] | None = None,
                           tcorr: int = 4000, tstep: float = 1.0,
                           h_symbol: str = 'HN') -> pd.DataFrame:
    """
    Calculate the S2 order parameter for a trajectory.

    Parameters
    ----------
    traj : pytraj.Trajectory
        The trajectory object.
    chains : dict[str, list[int]] | None
        A dictionary mapping chain names to lists of residue indices.
        If None, all chains will be included.
    tcorr : int
        The correlation time in picoseconds.
    tstep : float
        The time step in nanoseconds.
    h_symbol : str
        The symbol for the hydrogen atom to use in the calculation.
        Default is 'HN' for backbone amide hydrogens.


    Returns
    -------
    pd.DataFrame
        A DataFrame containing the S2 order parameter values for each residue.
    """
    if chains is not None and chains_dict is not None:
        traj = select_chains(traj, select_chains=chains, chains=chains_dict)

    n_residues = traj.top.n_residues
    H_mask = ':' + ','.join(str(i) for i in range(2, n_residues+1)) + f'@{h_symbol}'

    h_indices = pt.select_atoms(traj.top, H_mask)

    # select N (backbone) indices
    n_indices = h_indices - 1

    # create pairs
    nh_pairs = list(zip(n_indices, h_indices))

    s2 = pt.NH_order_parameters(traj, nh_pairs, tcorr=tcorr, tstep=tstep)

    return pd.DataFrame({'Residue': range(1, len(s2)+1), 'S2': s2})
