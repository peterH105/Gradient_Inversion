import numpy as np
from grad_inv_functions import *

def topo_corr(bed,ice,height_km,point_mass_number):
    
    """Calculates the gravitational effect of topography, discretized into a tesseroid model, which is transformed into point masses
    
    Input: 
    bed, ice - 3D arrays of ice and bedrock topography (Longitude, Latitude, Value)
    heights_km - Height, where gravitational effect is calculated (Float)
    point_mass_number - Number of point masses the tesseroids are converted (Float)
    
    Output:
    bouguer - Vector of gravitational effect of topography"""
    
    
    #Define heights as Vector layer
    heights=np.ones(len(bed[:,0]))*height_km*1000
    
    #Define densities for topo corr
    dens_rock=2670
    dens_water=1030
    dens_ice=917
    
    #Define layers of topography model (distinguishing between rock, water and ice)
    (bed_top,bed_bottom)=construct_layers_of_model(bed[:,2],0.0)
    (ice_above_top,ice_above_bottom)=construct_layers_of_model(ice[:,2],0.0)
    (ice_below_top,ice_below_bottom)=construct_layers_of_model(bed[:,2]-ice_above_bottom,0.0) 
    diff=ice_above_top-bed_top
    bed[np.where(diff <0),2]=ice[np.where(diff <0),2]
    bed_top=np.array((bed[:,0],bed[:,1],bed_top,bed_top*np.nan)).T
    bed_bottom=np.array((bed[:,0],bed[:,1],bed_bottom,bed_bottom*np.nan)).T
    ice_above_top=np.array((ice[:,0],ice[:,1],ice_above_top,ice_above_top*np.nan)).T
    ice_above_bottom=np.array((ice[:,0],ice[:,1],ice_above_bottom*0,ice_above_bottom*np.nan)).T
    ice_below_top=np.array((ice[:,0],ice[:,1],ice_below_top*0,ice_below_top*np.nan)).T
    ice_below_bottom=np.array((ice[:,0],ice[:,1],ice_below_bottom,ice_below_bottom*np.nan)).T

    # Define correct topography values for each layer
    for i in range(0,len(ice)):
        # 1st case: Rock
        if ice[i,2]>0 and bed[i,2]>0 and ice[i,2]==bed[i,2]:
            bed_top[i,3]=bed_top[i,2]
            bed_bottom[i,3]=bed_bottom[i,2]
            ice_above_top[i,2:3]=0.0
            ice_above_bottom[i,2:3]=0.0
            ice_below_top[i,2:3]=0.0
            ice_below_bottom[i,2:3]=0.0
        # 2nd case: Water
        if ice[i,2]<0 and bed[i,2]<0 and ice[i,2]==bed[i,2]:
            bed_top[i,3]=bed_top[i,2]
            bed_bottom[i,3]=bed_bottom[i,2]
            ice_above_top[i,2:3]=0.0
            ice_above_bottom[i,2:3]=0.0
            ice_below_top[i,2:3]=0.0
            ice_below_bottom[i,2:3]=0.0
        # 3rd case: Ice Above --> includes rocks and ice    
        if ice[i,2]>0 and bed[i,2]>0 and ice[i,2]>bed[i,2]:
            bed_top[i,3]=bed_top[i,2]
            bed_bottom[i,3]=ice_above_top[i,2]
            ice_above_top[i,3]=ice_above_top[i,2]
            ice_above_bottom[i,3]=ice_above_bottom[i,2]
            ice_below_top[i,2:3]=0.0
            ice_below_bottom[i,2:3]=0.0
        # 4th case: Ice Below --> includes both ice layers
        if ice[i,2]>0 and bed[i,2]<0 and ice[i,2]>bed[i,2]:
            ice_above_top[i,3]=ice_above_top[i,2]
            ice_above_bottom[i,3]=ice_above_bottom[i,2]
            ice_below_top[i,3]=ice_below_top[i,2]
            ice_below_bottom[i,3]=ice_below_bottom[i,2]

    # Multiply to get negative values
    bed_top=np.copy(bed_top[:,2])*-1
    bed_bottom=np.copy(bed_bottom[:,2])*-1
    ice_above_top=np.copy(ice_above_top[:,2])*-1
    ice_above_bottom=np.copy(ice_above_bottom[:,2])*-1
    ice_below_top=np.copy(ice_below_top[:,2])*-1
    ice_below_bottom=np.copy(ice_below_bottom[:,2])*-1
    
    # Assign densities to each layer
    density_topo=np.ones(bed_top.shape[0])*dens_rock
    density_topo[np.where(bed_top == 0)]=(dens_rock-dens_water)*-1
    density_ice_above=np.ones(ice_above_top.shape[0])*dens_ice
    density_ice_below=np.ones(ice_below_top.shape[0])*(dens_rock-dens_ice)*-1

    # Calculate gravitational effect of topography
    print("Calculate topographic effect")
    J=create_Jacobian(bed[:,0],bed[:,1],bed_top,bed_bottom,heights, density_topo, point_mass_number)
    bouguer=np.sum(J.T, axis=1) 

    # Check if ice above sea level is present and calculate gravitational effect
    if np.any(ice_above_top) or np.any(ice_above_bottom):
        print("Calculate Ice Effect -- Above sea level")
        J=create_Jacobian(ice[:,0],ice[:,1],ice_above_top,ice_above_bottom,heights, density_ice_above, point_mass_number)
        bouguer_ice_above=np.sum(J.T, axis=1)
        bouguer=np.copy(bouguer)+bouguer_ice_above 
    
    # Check if ice above below level is present and calculate gravitational effect
    if np.any(ice_below_top) or np.any(ice_below_bottom):
        print("Calculate Ice Effect -- Below sea level")
        J=create_Jacobian(ice[:,0],ice[:,1],ice_below_top,ice_below_bottom,heights, density_ice_below, point_mass_number)
        bouguer_ice_below=np.sum(J.T, axis=1)
        bouguer=np.copy(bouguer)+bouguer_ice_below

    return bouguer