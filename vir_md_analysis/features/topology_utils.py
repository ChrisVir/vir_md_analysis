import pandas as pd
import pytraj as pt
from abnumber import Chain
from Bio.SeqUtils import seq1


def pytraj_top_to_dataframe(traj: pt.Trajectory) -> pd.DataFrame:
    """Convert a pytraj trajectory topology to a DataFrame."""
    topology_dict = traj.top.to_dict()

    keys = ['atom_name', 'atom_type', 'resname', 'resid', 'mol_number']
    data = {key: topology_dict[key] for key in keys}

    return pd.DataFrame(data)


def three_code_to_one_code(three_letter_code: pd.Series) -> str:
    """Convert three-letter amino acid codes to one-letter codes."""
    one_letter_code = three_letter_code.apply(seq1).values
    return "".join(one_letter_code)


def abnumber_chain_to_regions_df(chain: Chain) -> pd.DataFrame:
    """Convert an abnumber Chain object to a DataFrame of regions."""
    regions = chain.regions
    dfs = []
    for region, region_dict in regions.items():
        region_df = pd.DataFrame.from_dict(region_dict, orient='index')
        region_df['region'] = region
        dfs.append(region_df)

    df = (pd.concat(dfs)
          .reset_index(drop=True)
          .rename(columns={0: 'residue'}))

    fv_chain_type = 'H' if chain.is_heavy_chain() else 'L'
    df.insert(0, 'FV chain', fv_chain_type)

    return df


def is_immunoglobulin_chain(sequence: str, scheme: str = 'imgt'):
    """Check if a given sequence is an immunoglobulin chain.
    sequence: str
        Amino acid sequence of the chain.
    scheme: str
        Scheme to use for the chain (default is 'imgt').
    Returns: bool
        True if the chain is a variable chain, False otherwise.
    """
    try:
        Chain(sequence, scheme=scheme)
        return True

    except Exception:
        return False


def label_chains(traj: pt.Trajectory, scheme: str = 'imgt') -> pd.DataFrame:
    """Label chains in a pytraj trajectory."""
    df = pytraj_top_to_dataframe(traj)

    # deduplicate the data frame to get only resname, resid and mol_number
    df = df[['resname', 'resid', 'mol_number']].drop_duplicates()

    molecules = df['mol_number'].unique()

    dfs = []
    for molecule in molecules:
        mol_df = df.query('mol_number == @molecule')
        sequence = three_code_to_one_code(mol_df['resname'])
        if is_immunoglobulin_chain(sequence, scheme):

            chain = Chain(sequence, scheme)
            region_df = abnumber_chain_to_regions_df(chain)

            dfs.append(pd.concat([mol_df.reset_index(drop=True), region_df], axis=1))

    return pd.concat(dfs).reset_index(drop=True).fillna('')


def format_mdtraj_selection(start_residue: int, end_residue: int, backbone_only: bool = True) -> str:
    """Format a selection string for mdtraj.
    Parameters:
        start_residue: int
            The starting residue number.
        end_residue: int
            The ending residue number.
        backbone_only: bool
            If True, only select the backbone atoms.
    Returns:
        str
            A formatted selection string for mdtraj.
    """
    if backbone_only:
        return f"resid {start_residue} to {end_residue} and backbone"
    else:
        return f"resid {start_residue} to {end_residue}"


def format_pytraj_selection(start_residue: int, end_residue: int) -> str:
    """Format a selection string for pytraj.
    Parameters:
        start_residue: int
            The starting residue number.
        end_residue: int
            The ending residue number.
    Returns:
        str
            A formatted selection string for pytraj.
    """
    return f":{start_residue}-{end_residue}"


def get_start_end_residues(region: str, regions_df: pd.DataFrame) -> tuple[int, int]:
    """Get the start and end residues for a given region.
    Parameters:
        region: str
            The name of the region (e.g., 'CDR1', 'CDR2', etc.).
        regions_df: pd.DataFrame
            A DataFrame containing region information.
    Returns:
        tuple[int, int]
            A tuple containing the start and end residue numbers for the specified region.
    """
    if region not in regions_df['region'].values:
        raise ValueError(f"Region '{region}' not found in regions DataFrame.")

    region_df = regions_df.query('region == @region')
    start, stop = region_df['resid'].min(), region_df['resid'].max()
    return start, stop


def create_region_position_map(region_df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Create a mapping of regions to their start and end residue numbers.
    Parameters:
        region_df: pd.DataFrame
            A DataFrame containing region information.
    Returns:
        dict[str, tuple[int, int]]
            A dictionary mapping region names to tuples of (start_residue, end_residue).
    """
    region_position_map = {}
    for region in region_df['region'].unique():
        start, stop = get_start_end_residues(region, region_df)
        region_position_map[region] = (start, stop)
    return region_position_map
