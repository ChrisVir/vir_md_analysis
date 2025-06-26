
import mdtraj as md
import pytraj as pt
import pandas as pd
from Bio.SeqUtils import seq1
import numpy as np
import scipy as sp
from collections import defaultdict

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


def compute_sasa_per_frame(traj: md.Trajectory,
                           chains: list[str] = None,
                           selection: str = None,
                           relative: bool = True,
                           return_sasa_per_residue: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the solvent accessible surface area (SASA) of a trajectory for each frame.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory object.
    chains : list[str]
        A list of chain IDs to include in the SASA calculation.
        If None, all chains will be included.
    selection : str
        A selection string to filter atoms in the trajectory.
        If None, all atoms will be included.
    relative : bool
        If True, the SASA will be calculated relative to the maximum possible SASA for each
    return_sasa_per_residue : bool
        If True, the SASA will be calculated for each residue and returned as a separate DataFrame.

    Returns
    -------
    sasa_per_frame: pandas.DataFrame
        A DataFrame containing the SASA values for each frame in the trajectory.

    sasa_per_residue: pandas.DataFrame (optional)

    # TODO: Refactor this to handle regions more flexibly and simplify the logic.
    """

    if chains:
        # Select atoms based on the chain IDs
        region = ' or '.join([f'chainid {chainid}' for chainid in chains])
        traj = select_mdtraj_atoms(traj, region=region)
    elif selection:
        # Select atoms based on the provided selection string
        traj = select_mdtraj_atoms(traj, region=selection)

    results = md.shrake_rupley(traj, get_mapping=True, mode='residue')
    sasa_per_residue = pd.DataFrame(results[0], columns=traj.topology.residues)

    amino_acids = [seq1(residue.name) for residue in traj.topology.residues]
    total_possible_sasa = sum([max_aa_sasa[aa] for aa in amino_acids])

    sasa_per_frame = pd.DataFrame({'SASA': sasa_per_residue.sum(axis=1),
                                  'Frame': range(len(sasa_per_residue))})
    if relative:
        # Calculate relative SASA
        sasa_per_frame['SASA'] = sasa_per_frame['SASA'] / total_possible_sasa

    if return_sasa_per_residue:
        return sasa_per_frame, sasa_per_residue

    return sasa_per_frame


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

    ##TODO: We should split this funtion into three separate functions for each method
    ##       to avoid confusion and make it easier to use.
    ##       This will also allow us to handle the different return types more cleanly.
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

        num_hbonds_per_frame = (format_wernet_nilsson_hbond_results_to_dataframe(hbonds)
                                .sum(axis=0)
                                .reset_index()
                                .rename(columns={'index': 'Frame', 0: 'Number of Hydrogen Bonds'})
                                )

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


def convert_wernet_nilsson_hbond_results_to_dataframes(ws_results: list[np.ndarray]) -> list[pd.DataFrame]:
    """Convert Wernet-Nilsson hydrogen bond results to a list of DataFrames.

    Args:
        ws_results (list[np.ndarray]): List of NumPy arrays containing Wernet-Nilsson hydrogen bond results.

    Returns:
        list[pd.DataFrame]: List of DataFrames, each representing hydrogen bond information for a specific frame.
    """
    dfs = []
    for res in ws_results:
        dfs.append(pd.DataFrame(res, columns=['donor', 'hydrogen', 'acceptor']))
    return dfs


def format_wernet_nilsson_hbond_results_to_dataframe(ws_results: list[pd.DataFrame]) -> pd.DataFrame:
    """Convert Wernet-Nilsson hydrogen bond results to a DataFrame with columns for donor, hydrogen, and acceptor
    and columns indicating whether the contact exists in each frame.

    Args:
        ws_results (pd.DataFrame): DataFrame containing Wernet-Nilsson hydrogen bond results.

    Returns:
        pd.DataFrame: DataFrame with hbonds  by frame, where each row represents a unique hydrogen bond
        and each column represents a frame. The DataFrame contains columns for donor, hydrogen, acceptor, and
        the number of frames in which each hydrogen bond is present.
    """
    # ws_results = convert_wernet_nilsson_hbond_results_to_dataframes(ws_results)
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

    hbonds_by_frame_df = (pd.concat([donor_hydrogen_acceptor, df.reset_index(drop=True)], axis=1)
                            .set_index(['donor', 'hydrogen', 'acceptor'])
                          )
    return hbonds_by_frame_df


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
                                         scheme: str = 'closest-heavy',
                                         return_all: bool = False) -> (pd.Series |
                                                                       tuple[pd.Series, np.ndarray, np.ndarray]):
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
    num_contacts = (pd.DataFrame({'Number of Contacts': (distances <= cutoff).sum(axis=1),
                                 'Frame': range(len(distances))}))

    num_contacts = num_contacts.set_index('Frame')

    if return_all:
        return num_contacts, distances, atom_pairs
    return num_contacts


# ------------------------------------------------------------------------
# PyTraj Feature Extraction Functions
# ------------------------------------------------------------------------


def get_residue_chain_id(residue, traj: pt.Trajectory) -> int:
    """Get the chain ID of a residue in a Pytraj trajectory.
    Args:
        residue: A residue object from the trajectory.
        traj: The Pytraj trajectory object.
    Returns:
        int: The chain ID of the residue.
    """

    first_atom_index = residue.first_atom_index
    chain_id = traj.top[first_atom_index].chain
    return chain_id


def identify_chains_from_pytraj_trajectory(traj: pt.Trajectory,
                                           chain_names: list[str] | None = None) -> dict[list[int]]:
    """Identify chains in a pytraj trajectory.
    traj: pt.Trajectory
        The pytraj trajectory object.
    chain_names: list of str or None
        List of chain names to filter. If None, all chains are returned.
    Returns:
        chains: defaultdict
            A dictionary where keys are chain names and values are lists of residue indices in that chain.
    """
    num_chains = traj.top.n_mols

    if not chain_names:
        chain_names = [f'Chain {i+1}' for i in range(num_chains)]

    chains = defaultdict(list)

    for residue in traj.top.residues:
        chain_id = get_residue_chain_id(residue, traj)
        chain_name = chain_names[chain_id]

        chains[chain_name].append(residue.index)

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


def identify_nh_bonds(traj: pt.Trajectory,
                      resids: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Identify NH bonds in the trajectory.
    Args:
        traj: pytraj.Trajectory
            The trajectory object.
        resids: tuple[int, int] | None
            A tuple of residue indices to filter the bonds. If None, all NH bonds will be identified.
    Returns:
        list[tuple[int, int]]
            A list of tuples, where each tuple contains the indices of the nitrogen and hydrogen atoms.
    """
    nh_bonds = []
    if resids is not None:
        # Filter bonds based on the specified residue indices
        traj = traj[f':{resids[0]}-{resids[1]}']

    h_types = {'H', 'HN', 'H1', 'H2', 'H3'}
    for bond in traj.top.bonds:
        atom1, atom2 = bond.indices.tolist()
        if np.abs(atom2 - atom1) == 1:
            if (traj.top.atom(atom1).type == 'N' and
                    traj.top.atom(atom2).type in h_types):
                nh_bonds.append((atom1, atom2))
    nh_bonds.sort(key=lambda x: x[0])  # Sort by the first atom index
    return nh_bonds


def calculate_s2_parameter(traj: pt.Trajectory,
                           chains: str | list[str] | None = None,
                           chains_dict: dict[str, list[int]] | None = None,
                           tcorr: int = 4000,
                           tstep: float = 1.0) -> pd.DataFrame:
    """
    Calculate the S2 order parameter for a trajectory.

    Parameters
    ----------
    traj : pytraj.Trajectory
        The trajectory object.
    chains : str| list[str] | None
        List of chain names to include in the calculation.
    chains_dict : dict[str, list[int]] | None
        A dictionary mapping chain names to lists of residue indices.
        If None, all chains will be included.
    tcorr : int
        The correlation time in picoseconds.
    tstep : float
        The time step in nanoseconds.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the S2 order parameter values for each residue.
    """
    if chains is not None and chains_dict is not None:
        traj = select_chains(traj, select_chains=chains, chains=chains_dict)

    nh_pairs = identify_nh_bonds(traj)
    s2 = pt.NH_order_parameters(traj, nh_pairs, tcorr=tcorr, tstep=tstep)

    return (pd.DataFrame({'Residue': range(1, len(s2)+1), 'S2': s2})
            .set_index('Residue')
            )
