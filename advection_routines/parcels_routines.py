########################################################################################################
# Author: Simon Schäfers
# Date: 2024-02-16
# This script is part of the master thesis project "Eddy effects on South Atlantic Ventilat_valson Pathways 
# using Lagrangian trajectories"

# Last modified: 2024-03-16
########################################################################################################

# This script contains subroutines to create the fieldset and the particle set for the particle advection using parcels.
# The `create_fieldset` function creates the fieldset for the particle advection. It is based on the `parcels.fieldset.FieldSet` method.
# The `create_particleset` function creates the particle set for the particle advection. It is based on the `parcels.particleset.ParticleSet` method.
# The `periodic_bc`, `cope_errors`, and `sample_mld` kernels are used in the particle advection and partially adapted from  the parcels documentation.
# The `find_density` function finds particles with a certain density criteria. It is used to create the initial particle set.
# The `delete_stuck_particles` function filters out stuck particles and moves them to a new location where they don't interfere with the advection. 
# It is used for particle sets that are continued from previous simulat_valsons.


from parcels import JITParticle, ParticleSet, Variable ,FieldSet,StatusCode
from advection_routines.displacement import add_displacement
from parcels import Field
from glob import glob
from datetime import timedelta as delta
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#################### fieldset generation #################################

def create_fieldset(**kwargs):
    """Function that creates the fieldset for the particle advection. 
    It is based on the `parcels.fieldset.FieldSet` method. This function builds the fieldset from 
    the velocity files and the mixed layer depth files. It also adds the displacement field if necessary.
    It is configured for a POP-ocean model output. Accordingly the grid is an ARAKAWA B-grid, 
    U,V,W and MLD are given in cm. Deviations from this require adjustments in the code.

    ## Parameters:
    - kwargs: dictionary containing the following keys:
        - isMean: boolean, if True, the fieldset will be created from seasonal rolling mean files. If False, the fieldset will be created from the daily mean files.
        - isConstant: boolean, if True, the fieldset will be created from a constant yearmean. If False, the fieldset will be created from the daily mean files.
        - canInterpolate: boolean, if True, the fieldset will be modified with the interpolated files to smoothen the yearly transition.
        - canDisplace: boolean, if True, the fieldset will be modified with the displacement field.
        - YEAR: int, the year for which the fieldset will be created.
        - COMP_TIMESTEP: int, the time step for the computation of the fieldset.
        - CHUNK_FRACTION: int, the fraction of the original chunk size. If 'auto', the chunk size will be set to the original size. If None, the chunk size will be set to the default size.
        - INT_LEN: int, the length of the interpolat_valson period. Default is 14.

    Output: the fieldset for the particle advection. Used in `advection_routines.advection_execution.fullExecution`.
    
    WARNING: 
    - The fieldset is based on the POP-ocean model output. Adjustments are necessary for other models.
    - The filenames are based on the file structure of the POP-ocean model output. Adjustments are necessary for other file structures.
    
    """

    # set default values for kwargs, disable interpolat_valson if isMean is true
    if kwargs['isMean'] or kwargs['isConstant']: kwargs['canInterpolate'] = False
    INT_LEN = kwargs['INT_LEN'] if 'INT_LEN' in kwargs else 14
    
    # set appendix for data access
    if kwargs['isConstant']: kwargs['isMean'] = False
    MEAN_APPENDIX = '_seasmean' if kwargs['isMean'] else ''
    MEAN_APPENDIX = '_yearmean' if kwargs['isConstant'] else MEAN_APPENDIX

    # define the folder and the files for the fieldset
    dataset_folder = 'YOUR-DIRECTORY-HERE/data_%i'%kwargs['YEAR']+MEAN_APPENDIX

    # collect u,v,w velocity files
    files = []    
    for dir in ['u','v','w']:
        files.append(sorted(glob(f'{dataset_folder}/nest_small_%i*%s%s.nc'%(kwargs['YEAR'],dir,MEAN_APPENDIX)))[::kwargs['COMP_TIMESTEP'] ])
        if kwargs['canInterpolate']:
            files[-1][-INT_LEN:] = sorted(glob(f'{dataset_folder}/nest_small_%i*%s_interpolated.nc'%(kwargs['YEAR'],dir)))
    print('[INFO] number of time steps: %i'%len(files[0]))

    # define the variables, dimensions, and filenames for the fieldset
    variables =  {"U": 'UVEL',"V": 'VVEL',"W": 'WVEL'}                               
    dimensions = {key: {"lon": "XU", "lat": "YU", "depth": "W_DEP", "time": "TU"} for key in ["U", "V", "W"]}
    filenames =  {key: {'lon': files[0][0], 'lat': files[0][0], 'depth': files[2][0], 'data': files[i]}
                  for i, key in enumerate(['U', 'V', 'W'])}    
    
    # set the chunk size for the fieldset
    if kwargs['CHUNK_FRACTION'] in ['auto',None]: 
        CS = kwargs['CHUNK_FRACTION']                                          
    else:                                                                         
        CS = {  "time": ("TU", 30),
                "depth": ("W_DEP", 20),
                "lat": ("YU", 1400//kwargs['CHUNK_FRACTION']),
                "lon": ("XU", 3600//kwargs['CHUNK_FRACTION']),
        }
    print('chunking: ', CS)

    # define the timestamps for the fields
    START_DATE = np.datetime64('%i-01-01'%kwargs['YEAR'])
    END_DATE = START_DATE + np.timedelta64(365, 'D')
    dates = np.arange(START_DATE, END_DATE, step=np.timedelta64(kwargs['COMP_TIMESTEP'] , 'D'))
    timestamps = np.expand_dims(dates,axis = 1)
   
    # create the fieldset

    if kwargs['isConstant']:
        fieldset = FieldSet.from_b_grid_dataset(filenames,variables,dimensions,
                                                chunksize = CS,
                                                allow_time_extrapolation=True)      

    else: fieldset = FieldSet.from_b_grid_dataset(filenames,variables,dimensions,
                                            timestamps = timestamps,
                                            time_periodic=delta(days = 365),chunksize = CS) 
    
    
    # add the displacement field if necessary
    if kwargs['canDisplace']:
        fieldset = add_displacement(fieldset, file_path=files[0][0])

    # add the mixed layer depth fields, one for the daily mean and one for the seasonal mean
    mld_folder = 'YOUR-DIRECTORY-HERE/data_mld'
    mld_files = sorted(glob(f'{mld_folder}/HMXL_%i*.nc'%kwargs['YEAR']))[::kwargs['COMP_TIMESTEP']]
    mld_files = [f for f in mld_files if '_seasmean' not in f]
    mld_mean_files = sorted(glob(f'{mld_folder}/HMXL_%i*_seasmean.nc'%kwargs['YEAR']))[::kwargs['COMP_TIMESTEP']]

    # create the mixed layer depth fields
    if kwargs['isConstant']:
        mld_const_file = ['YOUR-DIRECTORY-HERE/data_mld/yearmean%i_MLD.nc'%kwargs['YEAR'],'YOUR-DIRECTORY-HERE/data_mld/yearmean%i_MLD_copy.nc'%kwargs['YEAR']]
        mld_file = xr.open_dataset(mld_const_file[0])
        HMXL = Field('HMXL', data = np.array(mld_file.HMXL),lon = fieldset.U.grid.lon,lat = fieldset.U.grid.lat,allow_time_extrapolation=True)
        HMXL_mean = Field('HMXL_mean', data = np.array(mld_file.HMXL),lon = fieldset.U.grid.lon,lat = fieldset.U.grid.lat,allow_time_extrapolation=True)

    else:
        if len(timestamps) != len(mld_files): raise ValueError('number of timestamps and mldfiles does not match')
        HMXL = Field.from_netcdf(
                #'HMXL',
                filenames={'lon': files[0][0], 'lat': files[0][0], 'data': mld_files},
                variable={"HMXL":"HMXL"},
                dimensions={"lon": "XU", "lat": "YU"},
                timestamps = timestamps,
                time_periodic=delta(days = 365),
                allow_time_extrapolation=False,
            )
        HMXL_mean = Field.from_netcdf(
                #'HMXL_mean',
                filenames={'lon': files[0][0], 'lat': files[0][0], 'data': mld_mean_files},
                variable={"HMXL_mean":"HMXL"},
                dimensions={"lon": "XU", "lat": "YU"},
                timestamps = timestamps,
                time_periodic=delta(days = 365),
                allow_time_extrapolation=False,
            )

    # add the mixed layer depth fields to the fieldset
    fieldset.add_field(HMXL)
    fieldset.add_field(HMXL_mean)

    # set the scaling factors for the fieldset, as the velocity and MLD is given in cm
    fieldset.W.set_scaling_factor(-1/100)
    fieldset.U.set_scaling_factor(1/100)                                            
    fieldset.V.set_scaling_factor(1/100)
    fieldset.HMXL.set_scaling_factor(1/100)
    fieldset.HMXL_mean.set_scaling_factor(1/100)

    # add periodic boundary conditions to the fieldset
    fieldset.add_constant("halo_west", fieldset.U.grid.lon[0])                            
    fieldset.add_constant("halo_east", fieldset.U.grid.lon[-1])
    fieldset.add_periodic_halo(zonal=True)                                          
    
    return fieldset

########################## particle generation #################################

def create_particleset(fieldset, **kwargs):
    """Function that creates the particle set for the particle advection.
    It is based on the `parcels.particleset.ParticleSet` method. This function builds the particle set from initial conditions (default `find_density`) or from the last particle set.
    It optionally filters out stuck particles and moves them to a new location (lon: 300, lat: 0, depth: 1000) where they don't interfere with the advection.
    
    ## Parameters:
    - fieldset: class, the fieldset for the particle advection (use `create_fieldset`).
    - kwargs: dictionary containing the following keys:
        - N_YEARS: int, the passed time of years since the start of the simulat_valson. If 0, the particle set will be created from initial conditions. If >0, the particle set will be created from the last particle set.
        - NAME: str, the name of the simulat_valson.
        - lon_lims: list, the lon_valstude limits for the initial conditions.
        - lat_lims: list, the lat_valstude limits for the initial conditions.
        - depth_lims: list, the depth limits for the initial conditions.
        - canDisplace: boolean, if True, the particle class (`JITParticle`) will be created with the displacement parameters.
        - canDeleteStuck: boolean, if True, the particle set will be filtered for stuck particles and moved to a new location.

    Output: the particle set for the particle advection. Used in `advection_routines.advection_execution.fullExecution`.
    """

    # define the additional variables for the particle class
    variables = [
        Variable("n_up"),
        Variable("HMXL"),
        Variable("n_up_mean"),
        Variable("HMXL_mean"),
        ]
    if kwargs['canDisplace']:
        variables += [
            Variable("d_u"),
            Variable("d_v"),
            Variable("d2s"),
        ]

    # create the particle class
    SampleParticle = JITParticle.add_variables(variables)

    # create the data for the particle set
    # if N_YEARS is 0, the particle set will be created from initial conditions (default `find_density`)
    if kwargs['N_YEARS'] ==0:
        print('[INFO] creating new particle set')
        initial_particles = find_density(definition='max',plot=False,**kwargs)
        transferred_data = {'lon':initial_particles[0],'lat':initial_particles[1],'depth':initial_particles[2]}

    # if N_YEARS is >0, the particle set will be created from the last particle set
    else:
        print('[INFO] using particle set from previous particles')

        # read the last time step of the last particle set from the zarr storage and transfer all attributes
        ds = xr.open_zarr("zarr_storage/%s_"%kwargs['NAME']+"%i.zarr"%(kwargs['N_YEARS']-kwargs['YEARS_PER_ITERATION']))
        transferred_data = {key: np.array(ds[key][:, -1].data.flatten()) for key in ds.keys()}

        # filter out stuck particles by following criteria and move them to a new location
        if kwargs['canDeleteStuck']:
            print('[INFO] detecting & (re)moving stuck particles')
            transferred_data = delete_stuck_particles(transferred_data,ds)

        # change 'depth' to 'z' and remove 'time' to avoid conflicts with the particle class
        transferred_data['depth']=transferred_data['z']
        transferred_data.pop('z')
        transferred_data.pop('time')

    return transferred_data#ParticleSet(fieldset=fieldset, pclass=SampleParticle,**transferred_data) 

########################## kernels ##################################################

def periodic_bc(particle, fieldset,time):                                           
    """Kernel that applies periodic boundary conditions to the particle advection.
    If a particle exceeds the zonal boundaries, it will be moved to the opposite boundary.
    This Kernel is used in a parcels simulat_valson (`pset.execute`)

    ## Parameters:
    - particle: class, the particle the boundary is applied to.
    - fieldset: class, the fieldset containing the field data
    - time: int, the time at which the particle is being advected.
    """
   
    if particle.lon < fieldset.halo_west:                                          
        particle_dlon += fieldset.halo_east - fieldset.halo_west                    
    elif particle.lon > fieldset.halo_east:
        particle_dlon -= fieldset.halo_east - fieldset.halo_west


def cope_errors(particle, fieldset, time):
    """Kernel that copes with errors in the particle advection.
    If a particle exceeds the field boundaries, it will be deleted.
    To avoid numerical errors, the depth of the particle will be set to 0.5 if it is below 0.5.
    If a particle crosses the surface nonetheless, it will be brought back to the surface.
    This Kernel is used in a parcels simulat_valson (`pset.execute`)
    
    ## Parameters:
    - particle: class, the particle the errors are coped with.
    - fieldset: class, the fieldset containing the field data
    - time: int, the time at which the particle is being advected.
    """

    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()
        print('particle deleted')
    
    if particle.depth < 0.5:
        particle_ddepth = 0.5

    if particle.state == StatusCode.ErrorThroughSurface:
            particle_ddepth = 0.0
            particle.state = StatusCode.Success
            print('surfaced')


def sample_mld(particle, fieldset, time):
    """Kernel that samples the mixed layer depth for the particle advection.
    It samples the mixed layer depth and counts the number of times the particle reaches the mixed layer or subducts.
    The MLD crossing is calculated for the daily and the seasonal mean MLD.
    This Kernel is used in a parcels simulat_valson (`pset.execute`)

    ## Parameters:
    - particle: class, the particle the MLD is sampled for.
    - fieldset: class, the fieldset containing the field data
    - time: int, the time at which the particle is being advected.
    """

    # if the particle reaches the (daily) mixed layer, add 1 to the counter and make positive
    if fieldset.HMXL[time, particle.depth, particle.lat, particle.lon] > particle.depth and particle.n_up <=0:
        particle.n_up *= -1                                                      
        particle.n_up += 1

    # if the particle subducts, convert the counter to negative
    elif fieldset.HMXL[time, particle.depth, particle.lat, particle.lon] < particle.depth and particle.n_up >0:
        particle.n_up *= -1                                                         

    # if the particle reaches the (seasonal mean) mixed layer, add 1 to the counter and make positive
    if fieldset.HMXL_mean[time, particle.depth, particle.lat, particle.lon] > particle.depth and particle.n_up_mean <=0:
        particle.n_up_mean *= -1                                                        
        particle.n_up_mean += 1
    
    # if the particle subducts, convert the counter to negative
    elif fieldset.HMXL_mean[time, particle.depth, particle.lat, particle.lon] < particle.depth and particle.n_up_mean >0:
        particle.n_up_mean *= -1                                                        
    
    # sample the mixed layer depth for the particle
    particle.HMXL = fieldset.HMXL[time, particle.depth, particle.lat, particle.lon]
    particle.HMXL_mean = fieldset.HMXL_mean[time, particle.depth, particle.lat, particle.lon]



def find_density(definition = 'max',plot = True,**kwargs):
    """Function that finds particles with a certain density criteria.
    It uses the potential density data and finds grid cells that match the criteria, returning lon, lat and depth values.
    The area of interest is defined by the lon_valstude, lat_valstude and depth limits in the kwargs.
    Optionally, it plots the found particles in a 3D scatter plot.

    ## Parameters:
    - definition: str, can be 'max' or 'min', if density criteria has to be met in at least one observation or in all observaions.
    - plot: boolean, if True, the found particles will be plotted in a 3D scatter plot.
    - kwargs: dictionary containing the following keys:
        - lon_lims: list, the lon_valstude limits for the area of interest.
        - lat_lims: list, the lat_valstude limits for the area of interest.
        - depth_lims: list, the depth limits for the area of interest.

    Output: the found particles. Optionally used in `create_particleset`.
"""

    # adjust the lon_valstude limits to the 0-360° format as a crossing of the 0° meridian is possible
    lon_lims = [lim + 360 if lim<0 else lim for lim in kwargs['lon_lims']]

    # open the potential density file and the velocity bounds
    potential_density_file = xr.open_dataset('/work/uo0780/u241194/u241194/POPCFC/DATANC/YEARLY/PD_t.cfc11.1985-2009.nc')
    u_dims = xr.open_dataset('/work/uo0780/u241194/u241194/POPCFC/CMS/raw/TEST/nest_1_19900101000000u.nc')
    w_dims = xr.open_dataset('/work/uo0780/u241194/u241194/POPCFC/CMS/raw/TEST/nest_1_19900101000000w.nc')

    # define the grid and the depth
    depth_vals = np.array(potential_density_file.depth_t)
    lon_vals = np.array([(x-1090)/10 +360 if (x-1090)/10<0 else (x-1090)/10 for x in np.arange(1,3601)])
    lat_vals = np.array(u_dims.YU.data)

    # find the grid cells that match the spatial limits
    lon_entries = np.where((lon_vals >= lon_lims[0]) | (lon_vals <= lon_lims[1]))[0]
    lat_entries = np.where((lat_vals >= kwargs['lat_lims'][0]) & (lat_vals <= kwargs['lat_lims'][1]))[0]
    depth_entries = np.where((depth_vals >= kwargs['depth_lims'][0]) & (depth_vals <= kwargs['depth_lims'][1]))[0]

    # adjust files to the spatial limits
    potential_density = np.array(potential_density_file.PD[0,depth_entries,lat_entries,lon_entries])
    u_mask = (np.isnan(np.array(u_dims.UVEL[0,depth_entries,lat_entries,lon_entries])) ==False)
    w_mask = (np.isnan(np.array(w_dims.WVEL[0,depth_entries+1,lat_entries,lon_entries])) ==False)

    # define the density mask
    if definition in ['max',1]:
        density_mask = np.zeros_like(potential_density,dtype = 'i4')
    elif definition in ['min',0]:
        density_mask = np.ones_like(potential_density,dtype = 'i4')
    else:         
        print('[ERROR] unknown definition type')
        return
    
    # find the particles that match the density criteria, based on the definition type
    for record in range(len(potential_density_file.PD)):
        potential_density = np.array(potential_density_file.PD[record,depth_entries,lat_entries,lon_entries])
        if definition == 'max' or definition == 1:
            density_mask[(potential_density>1.02682) & (potential_density<1.02743)] = 1
        elif definition == 'min' or definition ==  0:
            density_mask[(potential_density<1.02682) | (potential_density>1.02743)] = 0

    # clear out undefined values 
    final_mask = (density_mask == 1)  &  (u_mask==True) & (w_mask==True) 
    print('Number of particles found: %i'%np.sum(final_mask))
    depth_indices,lat_indices,lon_indices = np.where(final_mask==True)   

    # correct the spatial values
    found_particles = lon_vals[lon_entries[0]+lon_indices],lat_vals[lat_entries[0]+lat_indices],depth_vals[depth_entries[0]+depth_indices]

    # 3d plot the found particles
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(90,-90,0)
        step = len(found_particles[0])//2000
        ax.set_title('lon_valstude modified')
        ax.scatter([item +360 if item<200 else item for item in found_particles[0][::step]],found_particles[1][::step],-found_particles[2][::step])
        plt.savefig('density_particles.png')

    return found_particles



def delete_stuck_particles(transferred_data,ds):
    """Function that filters out stuck particles and moves them to a new location (lon: 300, lat: 0, depth: 1000) where they don't interfere with the advection.
    ## Criteria for stuck particles:

    - previously_stuck: particles that were already stuck 
    - horizontally_stuck: particles that have not moved in the horizontal direction for 30 time steps
    - vertically_stuck: particles that have not moved in the vertical direction for 30 time steps
    - too_far_north: particles that exceed a lat_valstude of 15°N
    - deleted: particles that have been deleted due to errors

    ## Parameters:
    - transferred_data: the particle data from the last time step.
    - ds: the dataset of the previous simulat_valson.
        
    Output: the filtered particle data. Optionally used in `create_particleset`.
    """
    stuck_mask = np.zeros_like(transferred_data['lon'],dtype = bool)

    # define the criteria for stuck particles
    stuck_particles =  {'previously_stuck':   (transferred_data['z'] == 1000) & (transferred_data['lat'] == 0) & (transferred_data['lon'] == 300),
                        'horizontally_stuck':(np.std(np.array(ds.lon[:,-30:]),axis = 1) == 0) | (np.std(np.array(ds.lon[:,-30:]),axis = 1) == 0),
                        'vertically_stuck':  (np.std(np.array(ds.z[:,-30:]),axis = 1) == 0),
                        'too_far_north':      (transferred_data['lat'] > 15),
                        'deleted':  np.isnan(transferred_data['lon']),
    }   

    # print the status of the particles and combine the criteria
    print('[INFO] particle status:')
    for key in stuck_particles.keys():
        print(key,': ',sum(stuck_particles[key]))
        stuck_mask |= stuck_particles[key]

    # displace the particles if stuck to (lon: 300, lat: 0, detph: 1000)
    transferred_data['lon'][stuck_mask] = 300
    transferred_data['lat'][stuck_mask] = 0
    transferred_data['z'][stuck_mask] = 1000

    print('[INFO] moved %i stuck particles to (lon: 300, lat: 0, detph: 1000) '
        %(sum(stuck_mask)-sum(stuck_particles['previously_stuck']))) 
    
    return transferred_data