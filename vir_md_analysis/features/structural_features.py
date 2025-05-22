import MDAnalysis as md
import pandas as pd


def calculate_rg_for_specific_region(u: md.Universe, region: str = 'protein',
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
        'Time (ps)': times,
        'Radius of Gyration (A)': radii
    })
    df.insert(0, 'Region', region_name)

    return df
