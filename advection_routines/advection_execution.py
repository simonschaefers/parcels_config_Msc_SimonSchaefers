########################################################################################################
# Author: Simon Schäfers
# Date: 2024-02-16
# This script is part of the master thesis project "Eddy effects on South Atlantic Ventilation Pathways 
# using Lagrangian trajectories"

# Last modified: 2024-04-02
########################################################################################################

# This script contains the main functions to execute the advection of particles using parcels.
# The `full_execution` function is the main function of the script. It executes the advection of
# particles using parcels. The `rechunk_and_merge` function is used to rechunk and merge the zarr files 
# produced by the `full_execution` function.


import os
import re
import zarr
import xarray as xr                                                                   
from glob import glob
from datetime import timedelta
from parcels import AdvectionRK4_3D             
from advection_routines.parcels_routines import (
    create_fieldset, 
    create_particleset,
    periodic_bc, 
    sample_mld, 
    cope_errors,
)                                
from advection_routines.displacement import (
    displace, 
    set_displacement,
)

def full_execution(name,**kwargs):
    """Function to execute the advection of particles using parcels. Produces a zarr file with the particle trajectories.
    Run time and chunk size should be adjusted according to the available memory and the number of particles.
    Data need to be preprocessed and stored in the data folder. 
    The fieldset is created using the `parcels_routines.create_fieldset` function. In this configuration, the function works with POP-ocean model output velocities.
    The particles are created using the `parcels_routines.create_particleset` function. 
    Uses the `AdvectionRK4_3D` kernel for advection, and the `periodic_bc`, `sample_mld`, and `cope_errors` kernels for boundary conditions, MLD sampling, and error handling, respectively.
    If the particles are to be displaced before they touch land, the `displace` and `set_displacement` kernels are used. 
    The output is stored in a zarr file in the zarr_storage folder.
    This function can be used to continue a simulation from a previous execution. The current year is stored in a file, and the function reads it to continue the simulation.
   
    ## Parameters:
    - name: str, the name of the simulation.
    - kwargs: dict, the specifications of the simulation.
        - YEARS_PER_ITERATION: int, the number of years to be advected in each iteration.
        - COMP_TIMESTEP: int, the time step for the computation.
        - SAVE_TIMESTEP: int, the time step for the output.
        - CHUNK_FRACTION: int, the fraction of the fieldset to be loaded at once.
        - canInterpolate: bool, whether the fieldset can be interpolated.
        - YEAR: int, the year for which the simulation is carried out.
        - lat_lims: list, the limits of the latitude for the particle release.
        - lon_lims: list, the limits of the longitude for the particle release.
        - depth_lims: list, the limits of the depth for the particle release.
        - isMean: bool, whether the fieldset is constructed from seasonal mean fields (True) or from daily mean fields (False).
        - isConstant: bool, if True, the fieldset will be created from a constant yearmean. If False, the fieldset will be created from the daily mean files.
        - canDeleteStuck: bool, whether the particles should get removed if they are stuck on land.
        - canDisplace: bool, whether the particles should be displaced before they touch land.

    """
    
    # define default parameters
    preset_kwargs = {'YEARS_PER_ITERATION':2,
                    'COMP_TIMESTEP':1,
                    'SAVE_TIMESTEP':1,
                    'CHUNK_FRACTION':None,
                    'canInterpolate':True,
                    'YEAR':1990,
                    'lat_lims':[-40,-15],
                    'lon_lims':[-60,20],
                    'depth_lims':[400,1200],
                    'isMean':False,
                    'isConstant':False,
                    'canDeleteStuck':True,
                    'canDisplace':False,
    }
    kwargs['NAME'] = name    
    print('Run name and Specifications: ',kwargs)    

    # update kwargs with default parameters
    for key in preset_kwargs:
        if key not in kwargs:
            kwargs[key] = preset_kwargs[key]

    # read current simulation year
    if not os.path.exists("year_tracker"):
        os.makedirs("year_tracker")
    filename = 'year_tracker/n_years_%s.txt'%name
    if os.path.exists(filename) == False:
        kwargs['N_YEARS'] = 0
    else: 
        file = open(filename,'r')
        kwargs['N_YEARS'] = int(file.read().splitlines()[0])
        file.close()
    print('Current simulation year: ',kwargs['N_YEARS'])

    # create fieldset and particleset
    fieldset = create_fieldset(**kwargs)
    pset = create_particleset(fieldset,**kwargs)

    # define kernels
    kernels = [periodic_bc,AdvectionRK4_3D,sample_mld,cope_errors]      
    if kwargs['canDisplace']:
        kernels = [displace,periodic_bc,AdvectionRK4_3D,sample_mld,set_displacement,cope_errors] 

    # create output file, currently stored in memory
    output_memorystore = zarr.storage.MemoryStore()
    output_file = pset.ParticleFile(name=output_memorystore, 
                                    outputdt=timedelta(days=kwargs['SAVE_TIMESTEP']),)

    # execute the simulation, main function of the script
    pset.execute(kernels, 
                 runtime=timedelta(days=kwargs['YEARS_PER_ITERATION']*365),                                                   
                 dt=-timedelta(days=kwargs['COMP_TIMESTEP']),                                          
                 output_file = output_file,
    )

    # save output to zarr file
    if not os.path.exists("zarr_storage"):
        os.makedirs("zarr_storage")
    output_dirstore_name = "zarr_storage/%s_%i.zarr"%(name,kwargs['N_YEARS'])
    output_dirstore = zarr.storage.DirectoryStore(output_dirstore_name)
    zarr.convenience.copy_store(output_memorystore, output_dirstore)
    output_dirstore.close()


    # update current simulation year
    file = open(filename,'w')
    file.write(str(kwargs['N_YEARS']+kwargs['YEARS_PER_ITERATION']))
    file.close()


def rechunk_and_merge(name):
    """Function to rechunk and merge the zarr files produced by the `full_execution` function.
    Takes all the zarr files of one simulation, re-chunks and merges them into one file.
    Rechunking is necessary to faciolitate the access to a single trajectory.
    The output is stored in a zarr file in the zarr_storage folder.

    ## Parameters:
    - name: str, the name of the simulation.

    Output: zarr file in the zarr_storage folder, called `combined_{name}.zarr`.
    """
    
    # get the list of zarr files of simulation
    file_list = glob('zarr_storage/%s*'%name)
    file_list.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    # collect datasets
    ds_list = []
    for item in file_list:
        ds = xr.open_zarr(item)
        ds_list.append(ds)
        print('appended %s'%item)
    print('collecting datasets done')

    # combine datasets into one
    combined = xr.concat(ds_list,dim='obs')
    combined['time']=combined['time'][0]
    print('combining datasets done')

    # rechunk datasets to facilitate access to single trajectory
    rechunked_and_combined =combined.chunk({"trajectory":10,"obs":730*len(file_list)})
    print('rechunking done')
    print(rechunked_and_combined)
    
    # write to zarr file
    print('writing file...')
    for vname, vobj in rechunked_and_combined.data_vars.items():
        if "chunks" in vobj.encoding:
            del vobj.encoding["chunks"]
    output_memory_dir = 'zarr_storage/combined_%s.zarr'%name
    rechunked_and_combined.to_zarr(output_memory_dir, mode="w")
    print('writing file done')
 