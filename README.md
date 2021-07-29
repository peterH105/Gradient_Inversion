# Gradient_Inversion

This collection contains data and code to invert satellite gravity gradient data for the Moho depth and density contrasts at the Moho depth.

The data files are taken from the following sources: \
Gravity gradient of GOCE - http://eo-virtual-archive1.esa.int/Index.html \
Seismological Regionalization - https://schaeffer.ca/models/sl2013sv-tectonic-regionalization/ \
Seismic Moho depth - USGS database as published by Chulick et al. 2013 (South America only)

Africa: \
A new regionalization map is calculated usign the AF2019 seismic tomography model of Celli et al: https://nlscelli.wixsite.com/ncseismology/models \
Seismic Moho depths are taken from the USGS data base and from Globig et al. 2016: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JB012972 

The collection contains Jupyter notebook files that can be downloaded and run with Python 3. They are designed to run the inversion for various study areas: South America (i.e. Amazonian Craton), Africa and a more flexible version in a study area defined by the user.

For official use, please cite the paper: Haas, P., Ebbing J., Szwillus W. - Sensitivity analysis of gravity gradient inversion of the Moho depth – A case example for the Amazonian Craton, Geophysical Journal International, 2020, doi: 10.1093/gji/ggaa122 

You need Python 3 to run this notebook. This notebok has been tested on a Windows machine.

May 2020: An updated version, including a new notebook at user-defined study area has been added. The forward calculation of the gravitational effect now uses conversion of tesseroids to point masses, as written by Wolfgang Szwillus. This substitutes previous calculations of the executable tesseroids file of Leonardo Uieda (http://tesseroids.leouieda.com).

April 2021: The repository has been revised and extended with the inversion for the African continent. This inversion is split in two steps and allows flexible density contrasts for individual cratons. For official use, please cite the following: Haas, P., Ebbing, J., Celli, N., Rey P. - Two-step gravity inversion reveals variable architecture of African cratons, submitted to Frontiers in Earth Science.

July 2021: A textfile with the estimated Moho depth and density contrasts for the African continent has been uploaded.

