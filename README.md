# Gradient_Inversion

This collection contains data and code to invert satellite gravity gradient data for the Moho depth and density contrasts at the Moho depth.

The data files are taken from the following sources: \
Gravity gradient of GOCE - http://eo-virtual-archive1.esa.int/Index.html \
Seismological Regionalization - https://schaeffer.ca/models/sl2013sv-tectonic-regionalization/ \
Seismic Moho depth - USGS database as published by Chulick et al. 2013

The collection contains two Jupyter notebook files that can be downloaded and run with Python 3. They are designed to run the inversion for the Amazonian Craton, using synthetic and satellite gravity data, as published in the paper. 
A third version is stored in the subfolder "Variable study area". It includes Bouguer correction and allows the inversion of satellite gravity data in a user-defined study area. This notebook can alternatively be run in the browser using the Binder tool below:

https://mybinder.org/v2/gh/peterH105/Gradient_Inversion/master

For official use, please cite the paper: Haas, P., Ebbing J., Szwillus W. - Sensitivity analysis of gravity gradient inversion of the Moho depth – A case example for the Amazonian Craton, Geophysical Journal International, 2020, doi: 10.1093/gji/ggaa122

You need Python 3 to run this notebook. This notebok has been tested on a Windows machine.

04/05/20: An updated version, including a new notebook at user-defined study area has been added. The forward calculation of the gravitational effect now uses conversion of tesseroids to point masses, as written by Wolfgang Szwillus. This substitutes previous calculations of the executable tesseroids file of Leonardo Uieda (http://tesseroids.leouieda.com).

