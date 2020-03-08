import numpy as np
from scipy.interpolate import interp1d,RegularGridInterpolator
import subprocess
import tempfile
from matplotlib import path
import progressbar

def construct_tesseroids_from_interface(lonGrid,latGrid,topGrid,bottomGrid, dens):
    """Create tesseroids from two interfaces (bottom and top).
    """
    assert len(np.unique(lonGrid))==lonGrid.shape[1]
    assert len(np.unique(latGrid))==lonGrid.shape[0]
    tesses=[]
    N = len(lonGrid.ravel())
    dx = (lonGrid.max() - lonGrid.min())/(lonGrid.shape[1]-1)
    dy = (latGrid.max() - latGrid.min())/(lonGrid.shape[0]-1)
    
    for i in range(N):
        lon0 = lonGrid.ravel()[i]
        lat0 = latGrid.ravel()[i]
        top = 1000.0 * topGrid.ravel()[i]
        bottom = 1000.0 * bottomGrid.ravel()[i]
        tessString = '%.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (
        lon0-0.5*dx,lon0+0.5*dx,lat0-0.5*dy,lat0+0.5*dy,top, bottom, dens[i])
        tesses.append(tessString)

    return tesses
import time
import os

def get_inversion_design_matrix(tesses,stations,component):
        
    big_mat = np.zeros((len(stations),len(tesses)))
    bigger_mat=[]
    stat_file,stat_file_path=tempfile.mkstemp()
    os.close(stat_file)
    np.savetxt(stat_file_path,stations,fmt='%.5f')

    bar = progressbar.ProgressBar(maxval=len(stations+1), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    #bar=progressbar.ProgressBar(max_value=len(stations+1))
    for i,tess in enumerate(tesses):
        bar.update(i+1)
        tess_file,tess_file_path = tempfile.mkstemp()
        os.close(tess_file)
        with open(tess_file_path,'w') as f:
            f.write(tess)
            
        with open(stat_file_path) as c,open(stat_file_path) as d,open(stat_file_path) as e,open(stat_file_path) as f,open(stat_file_path) as g,open(stat_file_path) as h,open(stat_file_path) as j:

            if component=="gz":
                subz = subprocess.Popen(("Tesseroids/tessgz.exe",tess_file_path),stdin=h, stdout=subprocess.PIPE)  
                big_mat[:,i] = np.genfromtxt(subz.communicate()[0].decode().split('\n'))[:,3]
                os.remove(tess_file_path)				
            if component=="gxx":
                subz = subprocess.Popen(("Tesseroids/tessgxx",tess_file_path),stdin=h, stdout=subprocess.PIPE)  
                big_mat[:,i] = np.genfromtxt(subz.communicate()[0].split('\n'))[:,3]
                os.remove(tess_file_path)
            if component=="gxy":
                subz = subprocess.Popen(("Tesseroids/tessgxy",tess_file_path),stdin=h, stdout=subprocess.PIPE)  
                big_mat[0,:,i] = np.genfromtxt(subz.communicate()[0].split('\n'))[:,3]
                os.remove(tess_file_path)
            if component=="gxz":
                subz = subprocess.Popen(("Tesseroids/tessgxz",tess_file_path),stdin=h, stdout=subprocess.PIPE)  
                big_mat[0,:,i] = np.genfromtxt(subz.communicate()[0].split('\n'))[:,3]
                os.remove(tess_file_path)	
            if component=="gyy":
                subz = subprocess.Popen(("Tesseroids/tessgyy",tess_file_path),stdin=h, stdout=subprocess.PIPE)  
                big_mat[:,i] = np.genfromtxt(subz.communicate()[0].split('\n'))[:,3]
                os.remove(tess_file_path)	
            if component=="gyz":
                subz = subprocess.Popen(("Tesseroids/tessgyz",tess_file_path),stdin=h, stdout=subprocess.PIPE)  
                big_mat[0,:,i] = np.genfromtxt(subz.communicate()[0].split('\n'))[:,3]
                os.remove(tess_file_path)	
            if component=="gzz":
                subzz = subprocess.Popen(("Tesseroids/tessgzz.exe",tess_file_path),stdin=h, stdout=subprocess.PIPE)  
                big_mat[:,i] = np.genfromtxt(subzz.communicate()[0].decode().split('\n'))[:,3]
                os.remove(tess_file_path)	
    bar.finish()
    return big_mat, bigger_mat
    
def create_Jacobian(lonGrid,latGrid,component,top,bottom, top_shift, bottom_shift, height_km,dens):
    # Calculates the Jacobian Matrix of the inversion
    # The shape of the Matrix is n Stations (row) and m Tesseroids (column). 
    # The function get_inversion_design_matrix calculates the gravitational effect of each tesseroid for each station.
    #
    # Input:
    # mode - forward calculation of single or all gravity components
    # lonGrid,latGrid - Grid of Longitude and Latitude values, n X m shape
    # component - component of which the gravitational effect is calculated
    # top, bottom - grids of top layer and bottom layer of discretized Moho depth
    # top_shift, bottom_shift - shifted layers of top and bottom
    # height_km - Measured height of the data (1D-scalar)
    # dens - 1D-vector of density contrast
    #
    # Output:
    # J,J_shift - Jacobian matrices
    # bigger_mat - 3D Jacobian matrix (only necessary for FTG-mode)
    
    loni = lonGrid
    lati = latGrid
    heights = np.ones(loni.flatten().shape)*height_km*1000.0
    stations = np.vstack((loni.flatten(),lati.flatten(),heights)).T
    ny = loni.shape[0]
    nx = loni.shape[1]
    N_points = nx*ny
    tesses = construct_tesseroids_from_interface(lonGrid,latGrid,top*0.001,bottom*0.001, dens)
    dens=np.abs(dens)
    print("Calculate 1st Jacobian")
    J, bigger_mat = get_inversion_design_matrix(tesses,stations,component)
    tesses_shift = construct_tesseroids_from_interface(lonGrid,latGrid,top_shift*0.001,bottom_shift*0.001, dens) 
    mode="single"
    print("Calculate 2nd Jacobian")
    J_shift, trash  = get_inversion_design_matrix(tesses_shift,stations,component)
    return J,J_shift,bigger_mat
    
def invert_and_calculate(prefix,moho,bouguer,J,J_shift,bigger_mat,dmatrix,save_fields,mode,shape):
    
    # Inverts the Moho depth with the calcualted Jacobian Matrix and optionally calculates residual fields
    #
    # Input:
    # prefix - prefix of datafile
    # moho - Moho depth of starting model
    # bouguer - gravity data
    # J, J_shift - Jacobian matrices
    # bigger_mat - 3D Jacobian matrix (only necessary for calculation of all gravity components)
    # dmatrix - Smoothing matrix 
    # save_fields - "yes" or "no" option to save calculated fields
    # mode - forward calculation of single or all gravity components
    # shape - Size of the data
    #
    # Output:
    # moho final - inverted Moho depth
    # bouguer_fit - Fit to gravity data
    
    N_points=dmatrix.shape[0]
    big_G = J.reshape((1*N_points,N_points))
    big_delta_G = J_shift.reshape((1*N_points,N_points))
    bouguer_mod=np.sum(big_G, axis=1) 
    big_d=bouguer.reshape((1*N_points)) - bouguer_mod 
    bouguer_obs=bouguer.reshape((1*N_points))

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
    # Weights the Jacobian matrices with the respective density contrasts
    #
    # Input:
    # J, J_shift - Jacobian matrices of initial inversion
    # dens_start - initial density contrasts
    # dens - density contrast of each iteration
    #
    # Output:
    # J_new, J_new_shift - Weighted Jacobian matrices
    J_new=(np.abs(dens)/dens_start)*J
    J_shift_new=(np.abs(dens)/dens_start)*J_shift

    return J_new,J_shift_new
    
   
def load_grav_data(farfield,sediments,area):

    # Load gravitational data for the inversion 
    # If inversion is carried out for different component than gzz, gravity data have to be changed manually inside the function 
    # Gravitational data is corrected for global topography
    # --> Farfield gravitational effect has to be replaced when changing the study area!
    # Resolution of the data must be identical with resolution of initial Moho depth! (1 degree)
    #
    # Farfield effect accounts for isostatic effect outside of study area and has to be calculated separately 
    # If farfield effect is activated, gravity data with global topographic correction and farfield compensation is calculated
    # Gravitational effect of sediments is optional and has to be calcualted separately and must be in the same format
    #
    # Input:
    # farfield - "yes" or "no" statement
    # sediments - "yes" or "no" statement 
    # area - boundaries of the study area
    #
    # Output:
    # arrays - 3-column Matrix of the gravity data (Lon,Lat, Gravity)

    data=np.loadtxt('Gravity_Data/guu_global_Topo_corrected_1degree.xyz') 
    data=np.array((data[:,0],data[:,1],data[:,2])).T                           
    data=cut_data_to_study_area(data,area)
    data=data[np.lexsort((data[:,0],data[:,1]))]
    if farfield=="yes":
        iso_outside=np.loadtxt('Gravity_Data/IsoEffect_farfield_Amazonia_225km_1degree_guu.xyz')
        iso_outside=iso_outside[np.lexsort((iso_outside[:,0],iso_outside[:,1]))]
        iso_outside=iso_outside[:,2]
        arrays = np.array((data[:,0], data[:,1], data[:,2]+iso_outside))              
    if farfield=="yes" and sediments=="yes":
        sed=np.loadtxt('Gravity_Data/SedEffect_Amazonia_CRUST_225km_1degree_guu.xyz')
        sed=sed[np.lexsort((sed[:,0],sed[:,1]))]
        sed=sed[:,2]
        arrays = np.array((data[:,0], data[:,1], data[:,2]-sed+iso_outside))
    if farfield!="yes" and sediments=="yes":
        sed=np.loadtxt('Gravity_Data/SedEffect_Amazonia_CRUST_225km_1degree_guu.xyz')
        sed=sed[np.lexsort((sed[:,0],sed[:,1]))]
        sed=sed[:,2]
        arrays = np.array((data[:,0], data[:,1], data[:,2]-sed)) 
    if farfield!="yes" and sediments!="yes":
        arrays = np.array((data[:,0], data[:,1], data[:,2]))
    return np.transpose(arrays)

def cut_data_to_study_area(data,area):

    # Cuts the data to the user-defined study area
    data=data[~(data[:,0]<area[2])]
    data=data[~(data[:,0]>area[3])]
    data=data[~(data[:,1]<area[0])]
    data=data[~(data[:,1]>area[1])]
    return data

def create_density_combinations(k,number_of_units):
    
    # Creates and sorts all density combinations for k density contrasts of n number of units
    # The combinations are stored in a matrix
    # total number of combinations is n^k
    #
    # Input:
    # k - range of density contrasts
    # number_of_units - Number of tectonic units
    #
    # Output:
    # dens_mat - matrix containing all density combinations (row) of different tectonic units (column)
    dens_mat=np.transpose(np.tile(k,(number_of_units,len(k)**(number_of_units-1))))

    for i in range(1,number_of_units):
        dens_mat[:,number_of_units-1-i]=np.transpose(np.reshape(dens_mat[:,number_of_units-1-i],(len(k)**i,len(k)**(number_of_units-i)))).flatten()
    return dens_mat
   
def interp_regular_grid_on_irregular_database(area,dx,moho,seismic_stations):

    # Interpolate regular grid values on irregular distributed points
    # points have to be inside the boundaries of the grid
    #
    # Input: 
    # area - boundaries of the study area
    # dx - step size of the grid
    # moho - 1-column layer of Moho depth
    # seismic stations - 3-column layer of seismic stations (Lon,Lat,Station) 
    #
    # Output:
    # interp_arr - Interpolated values of gridded data
    # moho_diff - Difference between point estimates and interpolated data 
    
    lon=np.arange(area[2],area[3]+dx,dx)
    lat=np.arange(area[0],area[1]+dx,dx)

    moho_for_interp=moho.reshape((len(lat),len(lon)))
    moho_for_interp=np.transpose(moho_for_interp)

    interp_func=RegularGridInterpolator((lon,lat),moho_for_interp)
    interp_arr=interp_func(seismic_stations[:,0:2],method="linear")
    moho_diff=(seismic_stations[:,2]-interp_arr)/1000
    return moho_diff,interp_arr
    
    
def create_rms_matrix(rms_matrix,data,moho_resid_points,moho_resid_grid,i,bouguer_fit):

    # Creates a matrix of RMS-values for residual Moho depth and residual gravity field
    
    bouguer_fit=bouguer_fit[~(data[:,2]==0)] # remove values outside of coastline
    bouguer_fit_rms=np.sqrt(np.sum(bouguer_fit**2)/(bouguer_fit.shape)) # compute RMS     

    moho_resid_points_rms=np.sqrt(np.sum(moho_resid_points**2)/(moho_resid_points.shape)) # Compute RMS of points only
    
    moho_resid_grid=moho_resid_grid[~(data[:,2]==0)] # remove values outside of coastline
    moho_resid_grid_rms=np.sqrt(np.sum(moho_resid_grid**2)/(moho_resid_grid.shape)) # compute RMS of grid  

    rms_matrix[i,0]=bouguer_fit_rms # construct columns of matrix, containing RMS values of residual field, residual Moho depth and residual binned only Moho depth
    rms_matrix[i,1]=moho_resid_points_rms
    rms_matrix[i,2]=moho_resid_grid_rms
    return rms_matrix

def construct_layers_for_gradient_inversion(refmoho,moho,area,dx,dy):

    # constructs layers which are required for the inversion
    #
    # Input:
    # refmoho - Reference Moho depth, which is a single values
    # moho - Moho depth of starting model, is undulating around refmoho 
    # area - boundaries of the study area
    # dx and dy - resolution of the layers
    #
    # Output:
    # individual layers of top and bottom of discretized Moho depth
    reference=refmoho
    moho[moho==reference]=reference+10 # avoid singularity

    moho_top=moho.copy()
    moho_bottom=moho.copy()
    
    moho_shift_const=np.copy(moho) - 1000.01 # shift of the Moho depth, essential for second tesseroid model

    moho_top[moho_top < reference] = reference # upper layer
    moho_bottom[moho_bottom > reference] = reference # lower layer
    moho_top[np.argwhere(moho_top-moho_bottom<=1)]=moho_top[np.argwhere(moho_top-moho_bottom<=1)]+10 # avoid singularity
    
    moho_bottom_shift = np.copy(moho_shift_const) # shifted layer
    moho_top_shift = np.copy(moho) #initial Moho

    # prepare grids
    moho_topgrid=moho_top.reshape(int((area[3]-area[2])/dx+1),int((area[1]-area[0])/dy+1), order='F').copy()
    moho_topgrid=np.transpose(moho_topgrid)
    moho_bottomgrid=moho_bottom.reshape(int((area[3]-area[2])/dx+1),int((area[1]-area[0])/dy+1), order='F').copy()
    moho_bottomgrid=np.transpose(moho_bottomgrid)

    moho_topgrid_shift=moho_top_shift.reshape(int((area[3]-area[2])/dx+1),int((area[1]-area[0])/dy+1), order='F').copy()
    moho_topgrid_shift=np.transpose(moho_topgrid_shift)
    moho_bottomgrid_shift=moho_bottom_shift.reshape(int((area[3]-area[2])/dx+1),int((area[1]-area[0])/dy+1), order='F').copy()
    moho_bottomgrid_shift=np.transpose(moho_bottomgrid_shift)
    return moho,moho_topgrid,moho_bottomgrid,moho_topgrid_shift,moho_bottomgrid_shift,reference
    
def Doperator(nfs,sul,suw):
    # composes the Dmatrix 'dmat' for roughness determination
    # nfs:      (2x1) number of patches 
    # sul:      (1) patch length
    # suw:      (1) patch width
    #
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
