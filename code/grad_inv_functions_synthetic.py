import numpy as np
from scipy.interpolate import interp1d,RegularGridInterpolator
import subprocess
import tempfile
from matplotlib import path
import progressbar
    
def create_Jacobian(lon,lat,top_layer,bottom_layer, heights, density, point_mass_number):
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
    
def invert_and_calculate(prefix,moho,bouguer,J,J_shift,dmatrix,save_fields,shape):
    
    """Inverts the Moho depth with the calcualted Jacobian Matrix and optionally calculates residual fields
    
    Input:
    prefix - prefix of datafile (String)
    moho - Vector of Moho depth of starting model
    bouguer - Vector of gravity data
    J, J_shift - Jacobian matrices
    dmatrix - Smoothing matrix, same shape as J
    save_fields - "yes" or "no" option to save calculated fields
    shape - Size of the data
    
    Output:
    moho final - estimated Moho depth
    bouguer_fit - Fit to gravity data"""
    
    # Restore Jacobian matrix
    N_points=dmatrix.shape[0]
    big_G = J.reshape((1*N_points,N_points))
    big_delta_G = J_shift.reshape((1*N_points,N_points))
    
    # modelled gravitational effect per station is the sum of each tesseroid 
    bouguer_mod=np.sum(big_G, axis=1) 
    big_d=bouguer.reshape((1*N_points)) - bouguer_mod 
    bouguer_obs=bouguer.reshape((1*N_points))
    
    # Solve linear equation system and calculate Moho depth and data fit
    rhs = big_delta_G.T.dot(big_d)-dmatrix.T.dot(dmatrix).dot(moho/1000)
    lhs = big_delta_G.T.dot(big_delta_G)+dmatrix.T.dot(dmatrix)
    moho_shift = np.linalg.solve(lhs,rhs)
    predicted = big_delta_G.dot(moho_shift).reshape((1,shape[1],shape[0]))
    moho_final = moho_shift + moho/1000
    bouguer_res=bouguer_obs-bouguer_mod
    bouguer_fit=np.copy(bouguer_res)
    
    if save_fields=="yes":
        np.savetxt(prefix+"_inverted_Moho.txt",moho_final)
        np.savetxt(prefix+"_predicted_field.txt",bouguer_mod)
        np.savetxt(prefix+"_observed.txt",bouguer_obs)
        np.savetxt(prefix+"_deltay_it0.txt",bouguer_res)
  
    return moho_final,bouguer_fit
    
def weight_Jacobian(J,J_shift,dens,dens_start):
    """Weights the Jacobian matrices with the respective density contrasts
    
    Input:
    J, J_shift - Jacobian matrices of initial inversion
    dens_start - initial density contrasts
    dens - density contrast of each iteration
    
    Output:
    J_new, J_new_shift - Weighted Jacobian matrices"""
    
    J_new=(np.abs(dens)/dens_start)*J
    J_shift_new=(np.abs(dens)/dens_start)*J_shift

    return J_new,J_shift_new
    
   
def load_grav_data(lateral_var,area):

    """Load gravitational data for the inversion 
    If inversion is carried out for different component than gzz, gravity data have to be changed manually inside the function 
    Gravity data for synthetic example has been calculated from an isostatic Moho depth with two different assumptions:
      1. Laterally constant density contrast (400 kg/m³)
      2. Laterally variable density contrast (for values see in paper)
    
    Input:
    lateral_var - "yes" or "no" statement
    area - boundaries of the study area
    
    Output:
    arrays - 3-column Matrix of the gravity data (Lon,Lat, Gravity)"""
    
    if lateral_var=="no":
        data=np.loadtxt('Gravity_Data/Effect_IsoMoho_Amazonia_guu_1degree_225km.xyz') 
    
    if lateral_var=="yes":
        data=np.loadtxt('Gravity_Data/Effect_IsoMoho_Amazonia_guu_1degree_225km_vardens.xyz') 
    
    data=np.array((data[:,0],data[:,1],data[:,2])).T                         
    data=cut_data_to_study_area(data,area)
    data=data[np.lexsort((data[:,0],data[:,1]))]

    return data

def cut_data_to_study_area(data,area):

    # Cuts the data to the user-defined study area
    data=data[~(data[:,0]<area[2])]
    data=data[~(data[:,0]>area[3])]
    data=data[~(data[:,1]<area[0])]
    data=data[~(data[:,1]>area[1])]
    return data

def create_density_combinations(k,number_of_units):
    
    """Creates and sorts all density combinations for k density contrasts of n number of units
    The combinations are stored in a matrix
    total number of combinations is n^k
    
    Input:
    k - range of density contrasts
    number_of_units - Number of tectonic units
    
    Output:
    dens_mat - matrix containing all density combinations (row) of different tectonic units (column)"""
    
    dens_mat=np.transpose(np.tile(k,(number_of_units,len(k)**(number_of_units-1))))

    for i in range(1,number_of_units):
        dens_mat[:,number_of_units-1-i]=np.transpose(np.reshape(dens_mat[:,number_of_units-1-i],(len(k)**i,len(k)**(number_of_units-i)))).flatten()
    return dens_mat
   
def interp_regular_grid_on_irregular_database(area,dx,moho,seismic_stations):

    """Interpolate regular grid values on irregular distributed points
    points have to be inside the boundaries of the grid
    
    Input: 
    area - boundaries of the study area
    dx - step size of the grid
    moho - 1-column layer of Moho depth
    seismic stations - 3-column layer of seismic stations (Lon,Lat,Station) 
    
    Output:
    interp_arr - Interpolated values of gridded data
    moho_diff - Difference between point estimates and interpolated data""" 
    
    lon=np.arange(area[2],area[3]+dx,dx)
    lat=np.arange(area[0],area[1]+dx,dx)

    moho_for_interp=moho.reshape((len(lat),len(lon)))
    moho_for_interp=np.transpose(moho_for_interp)

    interp_func=RegularGridInterpolator((lon,lat),moho_for_interp)
    interp_arr=interp_func(seismic_stations[:,0:2],method="linear")
    moho_diff=(seismic_stations[:,2]-interp_arr)/1000
    return moho_diff,interp_arr
    
    
def create_rms_matrix(rms_matrix,data,moho_resid_points,moho_resid_grid,i,bouguer_fit):

    """Creates a matrix of RMS-values for residual Moho depth and residual gravity field"""
    
    bouguer_fit=bouguer_fit[~(data[:,2]==0)] # remove values outside of coastline
    bouguer_fit_rms=np.sqrt(np.sum(bouguer_fit**2)/(bouguer_fit.shape)) # compute RMS     

    moho_resid_points_rms=np.sqrt(np.sum(moho_resid_points**2)/(moho_resid_points.shape)) # Compute RMS of points only
    
    moho_resid_grid=moho_resid_grid[~(data[:,2]==0)] # remove values outside of coastline
    moho_resid_grid_rms=np.sqrt(np.sum(moho_resid_grid**2)/(moho_resid_grid.shape)) # compute RMS of grid  

    rms_matrix[i,0]=bouguer_fit_rms # construct columns of matrix, containing RMS values of residual field, residual Moho depth and residual binned only Moho depth
    rms_matrix[i,1]=moho_resid_points_rms
    rms_matrix[i,2]=moho_resid_grid_rms
    return rms_matrix

def construct_layers_of_model(data_layer,reference):

    """constructs layers which are required for the inversion
    
    Input:
    data_layer - Vector of initial Moho depth
    reference - Reference layer, where data_layer is discretized (Float)
    
    Output:
    layer_top,layer_bottom - Top and bottom layer of discretized model"""
    
    data_layer=np.copy(data_layer)/1000
    reference=reference/1000
    layer_top=data_layer.copy()
    layer_bottom=data_layer.copy()
    layer_top[layer_top < reference] = reference # upper layer
    layer_bottom[layer_bottom > reference] = reference # lower layer


    return layer_top,layer_bottom
    
def Doperator(nfs,sul,suw):
    """composes the Dmatrix 'dmat' for roughness determination
    nfs:      (2x1) number of patches 
    sul:      (1) patch length
    suw:      (1) patch width
    
    Output:
    DD - Smoothing matrix following 2nd order Tikhonov regularization"""
    
    k=0;
    # dmat is the pre-operator matrix
    # defines from location of patches the neighboring patches
    dmat=np.ones((nfs[0]*nfs[1],4))  
    dmat[0:nfs[0],0]=0 # I 1/suw
    dmat[-2-nfs[0]+1:-1,1]=0 # II 1/suw
    dmat[-1,1]=0
    dmat[:len(dmat):nfs[0],2]=0 # III 1/sul
    dmat[nfs[0]-1:len(dmat)+2:nfs[0],3]=0 # IV 1/sul

    DD=np.zeros((nfs[0]*nfs[1],nfs[0]*nfs[1])) # square

    for i in range(0,nfs[0]*nfs[1]):

        flags=dmat[i,:];
       
        # diagonal
        DD[i,i]=(-1)*np.dot(flags,np.array([1/suw**2, 1/suw**2, 1/sul**2, 1/sul**2]).T);
        
        # neighboring patches if any
        if flags[0]==1:
            DD[i,i-nfs[k]]=1/suw**2
        if flags[1]==1:
            DD[i,i+nfs[k]]=1/suw**2
        if flags[2]==1:
            DD[i,i-1]=1/sul**2
        if flags[3]==1:
            DD[i,i+1]=1/sul**2
            
    return DD


# The following functions are included to calculate the construct point masses from tesseroids and to calculate their gravitational effect.
# In the original publication of Haas et al. the gravitational effect of tesseroids has been calculated with the executable tesseroid files.
# However, the .exe do not allow calculation on the browser for safety reasons. Therefore, the forward calculation has been changed to point masses.
# The accuracy is very similar to tesseroids.
# The following functions, as well as the annotations, are written by Wolfgang Szwillus and have to be treated confidentially.


def masspoint_calc_FTG_2(lon,lat,depths,mass,lons,lats,heights,**kwargs):
    """Calculate gravity field from a collection of point masses

    lon, lat, depths, mass have arbitrary but identical shape
    lons, lats and heights are vectors

    Units
    ---
    depths are POSITIVE DOWN in km
    heights is POSITIVE UP in m
    mass is in kg
    returns gravity in SI units

    kwargs
    ---
    calc_mode can be grad, grav or potential

    Returns
    ---
    The sensitivity matrix -> effect of each mass on all of the stations 
    """

    G=6.67428e-11
    lon = lon[...,None]
    lat = lat[...,None]
    depths =depths[...,None]
    mass = mass[...,None]
    dLon = lon - lons
    coslat1 = np.cos(lat/180.0*np.pi)
    cosPsi = (coslat1 * np.cos(lats/180.0*np.pi) * np.cos(dLon/180.0*np.pi) +
            np.sin(lat/180.0*np.pi) * np.sin(lats/180.0*np.pi))
    cosPsi[cosPsi>1] = 1

    KPhi = (np.cos(lats/180.0*np.pi) * np.sin(lat/180.0*np.pi) - 
        np.sin(lats/180.0*np.pi) * coslat1 * np.cos(dLon/180.0*np.pi))


    rStat = (heights + 6371000.0)
    g = np.zeros((len(lons),len(lon)))
    rTess = (6371000.0 - 1000.0*depths)
    spherDist = np.sqrt(rTess**2 + rStat**2
                            - 2 * rTess * rStat*cosPsi)

    dx = KPhi * rTess
    dy = rTess * coslat1 * np.sin(dLon/180.0*np.pi)
    dz = rTess * cosPsi - rStat
    
    T = np.zeros(spherDist.shape+(6,))
    T[...,0] = ((3.0*dz*dz/(spherDist**5)- 1.0/(spherDist**3))*mass*G)
    return T

def memory_save_masspoint_calc_FTG_2(lon,lat,depths,mass,lons,lats,heights,point_mass_number,**kwargs):
    """Calculate gravity effect of a collection of point masses in chunks to save memory.
    max_size gives the maximum size in bytes used for the sensitivity matrix
    See docu for masspoint_calc_FTG_2
    """
    N = lon.size
    M = lons.size
    calc_mode = kwargs.get("calc_mode","grad")
    max_size = kwargs.get("max_size",1000000000) # in bytes
    verbose = kwargs.get("verbose",False)

    T = np.zeros(lons.shape+(1,))
    partitions = np.ceil(8 * N * M / max_size).astype(int)
    if verbose:
        print('Number of partitions ',partitions)
    
    ixs = np.array_split(np.arange(M,dtype=int),partitions)
    for ix in ixs:
        design_matrix = masspoint_calc_FTG_2(lon,lat,depths,mass,lons[ix],lats[ix],heights[ix],**kwargs)
        J=sum_over_multidimensional_matrix(design_matrix,point_mass_number,lons,lats)

        if verbose:
            print('Parition  done')
    return J

def sum_over_multidimensional_matrix(design_matrix,point_mass_number,lons,lats):
    J=np.zeros((len(lons),len(lats)))
    for i in range(point_mass_number):
        for j in range(point_mass_number):
            for k in range(point_mass_number):
                J_calc=np.copy(design_matrix)*-1
                J_calc=J_calc[i][j][k][:][:][:]
                J_calc=J_calc[:, :, 0]/10**-9
                J=J+J_calc
    return J 
    
def tesses_to_pointmass(lon,lat,dx,dy,tops,bottoms,dens,hsplit,vsplit):
    """Convert several tesseroids into a set of point masses
    """
	
    depths = np.zeros((vsplit,hsplit,hsplit)+lon.shape)
    lon0 = np.zeros(depths.shape)
    lat0 = np.zeros(depths.shape)
    masses = np.zeros(depths.shape)
    dlon = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dx
    dlat = (2*np.arange(0,hsplit,1)-hsplit+1)/(2.0*hsplit)*dy
    for k in range(vsplit):
        print('Calculation point mass No',k+1)
        if k==0:
            top = tops
        else:
            top = (bottoms-tops)/(1.0*vsplit)*k + tops
        if k==vsplit-1:
            bot = bottoms
        else:
            bot = (bottoms-tops)/(1.0*vsplit)*(k+1) + tops
        r_top = 6371-top
        r_bot = 6371-bot
        r_term = (r_top**3-r_bot**3)/3.0
		
        ind=0
        for i in range(hsplit):
            for j in range(hsplit):
                lon0[k,i,j] = lon + dlon[j]
                lat0[k,i,j] = lat + dlat[i]
                depths[k,i,j] = 0.5 * (top+bot)
                surface_Area = -dx*(np.pi/(180.0*hsplit)) *np.cos(
                    lat0[k,i,j]/180.0*np.pi) * 2 * np.sin(dy/(360.0*hsplit)*np.pi)
                masses[k,i,j] = surface_Area*dens*r_term*1e9
                ind=ind+1
    return lon0,lat0,depths,masses