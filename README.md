vir_md_analysis
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/vir_md_analysis/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/vir_md_analysis/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/vir_md_analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/vir_md_analysis/branch/main)


Modules for performing analysis on MD trajectories for the VirMelt project.

This repo is devoted to 1) writing and testing function for creating descriptors from MD trajectories and 2) training machine learning models on those descriptors to predict protein thermal stability. 

This is based on the work from the [AbMelt](https://www.sciencedirect.com/science/article/abs/pii/S0006349524003850) paper by Rollins et al. 


### To extract features from MD trajectories
1. Install the package using `pip install -e .`
2. Run the `extract_features` command line tool with the appropriate arguments, e.g
    ```bash
    extract_features --traj trajectory.nc --top topology.nc --topology 'my_system'
    ```
See the cli help for more details:
```bash
extract_features --help
```

Note that the only netcdf files and prmtop files are supported at the moment. The topology file should be a prmtop file, and the trajectory file should be a netcdf file.



### Copyright

Copyright (c) 2025, VirBiotech/MLTeam


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
