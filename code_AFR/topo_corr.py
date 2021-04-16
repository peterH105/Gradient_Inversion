import numpy as np
from grad_inv_functions_AFR import *


def create_Jacobian_topo_corr(lon,lat,top_layer,bottom_layer, heights, density, point_mass_number):
    """Calculates the Jacobian Matrix of the inversion
    The shape of the Matrix is n Stations (row) and m Tesseroids (column). 
    The function get_inversion_design_matrix calculates the gravitational effect of each tesseroid for each station.
    
    Input:
    lon, lat - Vectors of Longitude and Latitude for all stations
    top_layer, bottom_layer - Vectors of top and bottom layer of tesseroid model
    heights - Vector of heights of stations
    density - Vector of density contrast of the tesseroid model
    point_mass_number - Number of point masses the tesseroids are converted (Float)
    
    All vectors have the length of n stations!
    
    Output:
    J - Jacobian or Design matrix"""
    
    # resolution of grid (should be 1 degree)
    dx=(np.amax(lon)-np.amin(lon))/(np.sqrt(len(lon))-1)
    dy=(np.amax(lat)-np.amin(lat))/(np.sqrt(len(lat))-1)

    print("Calculate Jacobian")
    
    # Convert the tesseroids to point masses
    lon0,lat0,depths,masses=tesses_to_pointmass(lon,lat,dx,dy,
        top_layer,bottom_layer,density,hsplit=point_mass_number,vsplit=point_mass_number)
        
    #Calculate gravitational effect of point masses for each stations-point mass combination and store them in a matrix
    J=memory_save_masspoint_calc_FTG_2(lon0,lat0,depths,masses,lon,lat,heights,point_mass_number)
    return J


def topo_corr(lon,lat,bed,ice,heights,point_mass_number):

    """Calculates the gravitational effect of topography, discretized into a tesseroid model, which is transformed into point masses
    
    Input: 
    bed, ice - 2D arrays of ice and bedrock topography (Longitude, Latitude, Value)
    heights_km - Height, where gravitational effect is calculated (Float)
    point_mass_number - Number of point masses the tesseroids are converted (Float)
    
    Output:
    bouguer - Vector of gravitational effect of topography"""
    
    #Define densities for topo corr
    dens_rock=2670
    dens_water=1030
    dens_ice=917
    reference_topo=0.0

    (bed_top,bed_bottom)=construct_layers_of_model(bed[:,2],reference_topo)
    (ice_above_top,ice_above_bottom)=construct_layers_of_model(ice[:,2],reference_topo)
    (ice_below_top,ice_below_bottom)=construct_layers_of_model(bed[:,2]-ice_above_bottom,reference_topo) 
    diff=ice_above_top-bed_top
    bed[np.where(diff <0),2]=ice[np.where(diff <0),2]
    #ice_below_top[np.where(diff <0)]=ice_below_bottom[np.where(diff <0)]
    ice_above_bottom=np.copy(bed_top)
    ice_above_bottom[ice_above_bottom<0]=0.0
    count=-1
    for i in range(0,len(ice)):
        
        if bed[i,2]<0 and ice[i,2]<0 and ice[i,2]!=bed[i,2] and np.abs(bed[i,2]-ice[i,2])<10:
            bed[i,2]=ice[i,2]
        # rock
        if ice[i,2]>0 and bed[i,2]>0 and ice[i,2]==bed[i,2]:
            ice_above_top[i]=0.0
            ice_above_bottom[i]=0.0
            ice_below_top[i]=0.0
            ice_below_bottom[i]=0.0
            count=count+1
        #water       
        if ice[i,2]<0 and bed[i,2]<0 and ice[i,2]==bed[i,2]:
            ice_above_top[i]=0.0
            ice_above_bottom[i]=0.0
            ice_below_top[i]=0.0
            ice_below_bottom[i]=0.0
            count=count+1
        # ice above       
        if ice[i,2]>0 and bed[i,2]>0 and ice[i,2]>bed[i,2]:
            ice_below_top[i]=0.0
            ice_below_bottom[i]=0.0
            count=count+1
        # ice below     
        if ice[i,2]>0 and bed[i,2]<0 and ice[i,2]>bed[i,2]:
            bed_top[i]=0.0
            bed_bottom[i]=0.0
            count=count+1
    
    density_topo=np.ones(bed_top.shape[0])*dens_rock
    density_topo[np.where(bed_top == 0)]=(dens_rock-dens_water)*-1
    density_ice_above=np.ones(ice_above_top.shape[0])*dens_ice
    density_ice_below=np.ones(ice_below_top.shape[0])*(dens_rock-dens_ice)*-1

    print("Calculate topographic effect")
    J=create_Jacobian_topo_corr(lon,lat,bed_top*-1,bed_bottom*-1,heights, density_topo, point_mass_number)
    bouguer=np.sum(J.T, axis=1) 
    print(np.amin(bouguer),np.amax(bouguer))
    if np.any(ice_above_top) or np.any(ice_above_bottom):
        print("Calculate Ice Effect -- Above sea level")
        J=create_Jacobian_topo_corr(lon,lat,ice_above_top*-1,ice_above_bottom*-1,heights, density_ice_above, point_mass_number)
        bouguer_ice_above=np.sum(J.T, axis=1)
        print(np.amin(bouguer_ice_above),np.amax(bouguer_ice_above))
        bouguer=np.copy(bouguer)+bouguer_ice_above 
     
    if np.any(ice_below_top) or np.any(ice_below_bottom):
        print("Calculate Ice Effect -- Below sea level")
        J=create_Jacobian_topo_corr(lon,lat,ice_below_top*-1,ice_below_bottom*-1,heights, density_ice_below, point_mass_number)
        bouguer_ice_below=np.sum(J.T, axis=1)
        print(np.amin(bouguer_ice_below),np.amax(bouguer_ice_below))
        bouguer=np.copy(bouguer)-bouguer_ice_below

    return bouguer