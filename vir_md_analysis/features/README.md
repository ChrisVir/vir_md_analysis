Many of the features for this REPO are from the Abmelt Paper. 

The features of interest from the paper are: 
1. Solvent Acessible Surface Area (SASA)
2. Number of Hydrogen Bonds
3. Number of Lennard-Jones internal contacts
4. Radius of gyration (Rg)
6. Root mean-square fluctuation (RMSF)
7. $S^2$:N-H bond vector order parameter of FV substructure at T. 
8. $\Lambda$ dimensionless number related to heat capacity which quantifies the temperature dependence of $S^2$. 
9. $r-\Lambda$ coefficient of determination of the linear fit of $S^2$ vs. temperature for $\Lambda$ values.

The features are calculated using the MDAnalysis library.
The features are calculated for each frame of the trajectory and then averaged over the trajectory.

The features are calculated for the entire antibody and for different regions of the antibody.
The regions of the antibody are:
1. CDRH1
2. CDRH2
3. CDRH3
4. CDRL1
5. CDRL2
6. CDRL3
7. Framework regions (FR1, FR2, FR3, FR4)

The regions are determined using a neural net to assign the CDRs and framework regions.
MD simulations are done at different temperatures and the features are calculated for each temperature.
The features are then used to train a machine learning model to predict the melting temperature of the antibody.