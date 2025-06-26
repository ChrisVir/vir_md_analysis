from pathlib import Path
import mdtraj as md
import pytraj as pt


def load_mdtraj(trajectory_file: str | Path, topology_file: str | Path) -> md.Trajectory:
    """Load a trajectory using mdtraj.
    trajectory_file: str or Path
        Path to the trajectory file (e.g., .dcd, .nc).
    topology_file: str or Path
        Path to the topology file (e.g., .pdb, .prmtop).
    Returns:
        md.Trajectory object.
    """

    if isinstance(trajectory_file, str):
        trajectory_file = Path(trajectory_file)
    if isinstance(topology_file, str):
        topology_file = Path(topology_file)
    # Check if the files exist

    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file {trajectory_file} does not exist.")
    if not topology_file.exists():
        raise FileNotFoundError(f"Topology file {topology_file} does not exist.")

    if trajectory_file.suffix == '.nc':
        traj = md.load_netcdf(trajectory_file, top=topology_file)
    elif trajectory_file.suffix == '.dcd':
        traj = md.load_dcd(trajectory_file, top=topology_file)
    else:
        raise ValueError("Unsupported trajectory file format. Only .nc and .dcd are supported.")
    return traj


def load_pytraj(trajectory_file: str | Path, topology_file: str | Path) -> pt.Trajectory:
    """Load a trajectory using pytraj.
    trajectory_file: str or Path
        Path to the trajectory file (e.g., .dcd, .nc).
    topology_file: str or Path
        Path to the topology file (e.g., .pdb, .prmtop).
    Returns:
        pt.Trajectory object.
    """
    if isinstance(trajectory_file, str):
        trajectory_file = Path(trajectory_file)
    if isinstance(topology_file, str):
        topology_file = Path(topology_file)

    # Check if the files exist
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file {trajectory_file} does not exist.")
    if not topology_file.exists():
        raise FileNotFoundError(f"Topology file {topology_file} does not exist.")

    traj = pt.load(str(trajectory_file), top=str(topology_file))
    return traj
