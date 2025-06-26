import mdtraj as md
import pytraj as pt
import pandas as pd
from pathlib import Path
from vir_md_analysis.features.topology_utils import (format_pytraj_selection, format_mdtraj_selection,
                                                     label_chains, create_region_position_map)
from vir_md_analysis.features.loaders import load_mdtraj, load_pytraj
from vir_md_analysis.features.structural_features import (compute_contacts_on_specific_regions,
                                                          compute_radius_of_gyration_by_region,
                                                          compute_sasa_per_frame,
                                                          identify_hydrogen_bonds,
                                                          compute_stats_from_series,
                                                          compute_rmsd_on_specific_regions,
                                                          compute_rmsf_on_specific_regions,
                                                          calculate_s2_parameter
                                                          )


def compute_feature_on_mdtraj(traj: md.Trajectory, feature: str, feature_map: dict[str, callable],
                              region_position_map: dict[str, tuple[int, int]] | None = None):
    """Compute a feature for a trajectory on multiple regions.
    Parameters:
        traj: md.Trajectory
            The trajectory object from mdtraj.
        feature: str
            The  name of the feature to compute (e.g., 'rmsd', 'rmsf').
        feature_map: dict
            A dict with key as feature name and value as the function to compute the feature.
        region_position_map: dict[str, tuple[int, int]] | None
            A mapping of region names to their start and end residue positions.
            If None, the feature will be computed for the entire trajectory.
    """
    all_stats = []

    # do entire protein
    feature_func = feature_map[feature]
    feature_results = feature_func(traj, region='backbone')
    prefix = f"backbone {feature}"
    stats = compute_stats_from_series(feature_results[feature], prefix=prefix)
    all_stats.append(stats)

    # do CDRs and framework regions
    if region_position_map is not None:

        for region, (start, stop) in region_position_map.items():
            selection = format_mdtraj_selection(start, stop, backbone_only=True)
            feature_results = feature_func(traj, region=selection)
            prefix = f"{region} {feature}"
            stats = compute_stats_from_series(feature_results[feature], prefix=prefix)
            all_stats.append(stats)
    all_stats = pd.concat(all_stats, axis=1)
    return all_stats


def compute_feature_on_pytraj(traj: pt.Trajectory, feature: str, feature_map: dict[str, callable],
                              region_position_map: dict[str, tuple[int, int]] | None = None):
    """Compute a feature for a trajectory on multiple regions.
    Parameters:
        traj: pt.Trajectory
            The trajectory object from pytraj.
        feature: str
            The  name of the feature to compute (e.g., 'rmsd', 'rmsf').
        feature_map: dict
            A dict with key as feature name and value as the function to compute the feature.
        region_position_map: dict[str, tuple[int, int]] | None
            A mapping of region names to their start and end residue positions. If None, the feature will be computed
            for the entire trajectory.
    """
    all_stats = []

    # do entire protein
    feature_func = feature_map[feature]
    feature_results = feature_func(traj)
    prefix = f"total {feature}"
    stats = compute_stats_from_series(feature_results[feature], prefix=prefix)
    all_stats.append(stats)

    # do CDRs and framework regions
    if region_position_map is not None:

        for region, (start, stop) in region_position_map.items():
            selection = format_pytraj_selection(start+1, stop+1)  # pytraj is 1-indexed

            feature_results = feature_func(traj, region=selection)
            prefix = f"{region} {feature}"
            stats = compute_stats_from_series(feature_results[feature], prefix=prefix)
            all_stats.append(stats)
    all_stats = pd.concat(all_stats, axis=1)
    return all_stats


def extract_features(trajectory_file: Path | str,
                     topology_file: Path | str,
                     md_feature_func_map: dict[str, callable] = None,
                     pytraj_feature_func_map: dict[str, callable] = None,
                     system_name: str | None = None,
                     start_frame: int = 0,
                     end_frame: int | None = None,
                     save_fname: str | None = None,
                     output_dir: Path | str | None = None,
                     save_results: bool = True,
                     return_path: bool = True
                     ):
    """Extract features from a trajectory using either mdtraj or pytraj.
    Arguments:
        trajectory_file: Path or str
            Path to the trajectory file (e.g., .dcd, .nc).
        topology_file: Path or str
            Path to the topology file (e.g., .pdb, .prmtop).
        md_feature_func_map: dict[str, callable] | None
            Mapping of feature names to functions for mdtraj.
        pytraj_feature_func_map: dict[str, callable] | None
            Mapping of feature names to functions for pytraj.
        system_name: str or None
            Name of the system for the output DataFrame. If None, the name will be derived from the
            trajectory file name.
        start_frame: int
            The starting frame for the trajectory to analyze.
        end_frame: int | None
            The ending frame for the trajectory to analyze. If None, the entire trajectory will be used.
        save_fname: str | None
            Filename to save the extracted features. If None, the features will not be saved.
        output_dir: Path or str | None
            Directory to save the extracted features. If None, the current working directory will be used.
        save_results: bool
            Whether to save the extracted features to a file. Default is True.
        return_path: bool
            Whether to return the path to the saved file. Default is False.
    Returns:
        pd.DataFrame
            1 row DataFrame containing the extracted features with the system name as the first column.

    """

    if isinstance(trajectory_file, str):
        trajectory_file = Path(trajectory_file)
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file {trajectory_file} does not exist.")
    if isinstance(topology_file, str):
        topology_file = Path(topology_file)
        if not topology_file.exists():
            raise FileNotFoundError(f"Topology file {topology_file} does not exist.")

    if not md_feature_func_map:
        # TODO: consider making a load_md_feature_func_map function
        # that loads the feature functions from a config file or similar.
        md_feature_func_map = {
            'RMSD (nm)': compute_rmsd_on_specific_regions,
            'RMSF (nm)': compute_rmsf_on_specific_regions,
            'Number of Contacts': compute_contacts_on_specific_regions,
            'Radius of Gyration (A)': compute_radius_of_gyration_by_region,
            'Relative SASA': compute_sasa_per_frame,
            'Number of Hydrogen Bonds': identify_hydrogen_bonds
        }

    if not pytraj_feature_func_map:
        pytraj_feature_func_map = {
            'S2': calculate_s2_parameter
        }

    # Load the trajectory using mdtraj
    md_traj = load_mdtraj(trajectory_file, topology_file)
    if end_frame is not None:
        md_traj = md_traj[start_frame:end_frame]
    else:
        md_traj = md_traj[start_frame:]

    # load the trajectory using pytraj
    pt_traj = load_pytraj(trajectory_file, topology_file)
    if end_frame is not None:
        pt_traj = pt_traj[start_frame:end_frame]
    else:
        pt_traj = pt_traj[start_frame:]

    # Get the region position map
    region_df = label_chains(pt_traj)
    region_position_map = create_region_position_map(region_df)

    feature_stats = []

    for feature, func in md_feature_func_map.items():
        stats = compute_feature_on_mdtraj(md_traj, feature, md_feature_func_map, region_position_map)
        feature_stats.append(stats)

    for feature, func in pytraj_feature_func_map.items():
        stats = compute_feature_on_pytraj(pt_traj, feature, pytraj_feature_func_map, region_position_map)
        feature_stats.append(stats)

    # Concatenate all feature stats into a single DataFrame
    feature_stats_df = pd.concat(feature_stats, axis=1).reset_index(drop=True)

    if system_name is None:
        system_name = trajectory_file.stem

    feature_stats_df.insert(0, 'System Name', system_name)

    if save_results:
        if not output_dir:
            output_dir = trajectory_file.parent
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        if not save_fname:
            save_fname = f"{system_name}_features.csv"
        path = output_dir / save_fname
        print(f"Saving feature statistics to {path}")

        feature_stats_df.to_csv(path, index=False)

        if return_path:
            return path, feature_stats_df

    return feature_stats_df


if __name__ == "__main__":
    data_dir = Path().cwd().parent.parent / 'vir_md_analysis' / 'data'
    nc = data_dir / '8md_prod.nc'
    top = data_dir / 'MAbFv.parm7.hmass.parm7'
    extract_features(nc, top, system_name='8md_prod', start_frame=0, end_frame=100)
