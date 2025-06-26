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
