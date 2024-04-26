########################################################################################################
# Author: Simon Schäfers
# Date: 2024-02-16
# This script is part of the master thesis project "Eddy effects on South Atlantic Ventilation Pathways 
# using Lagrangian trajectories"

# Last modified: 2024-02-16
########################################################################################################

# This script is a submodule of the advection_routines package. It contains the functions to displace particles that are stuck on land.
# The `add_displacement` function adds the displacement field to the fieldset. It also adds the landmask and the distance to shore.
# The `set_displacement` kernel sets the displacement of the particle based on the distance to shore.
# The `displace` kernel displaces the particle based on the displacement set by the `set_displacement` kernel.

import numpy as np
import xarray as xr 
import math
from parcels import Field, GeographicPolar, Geographic

def add_displacement(fieldset,file_path = "data_1990/nest_small_19900101000000u.nc"):
    """Function that adds the displacement field to the fieldset. It also adds the landmask and the distance to shore.
    ## Parameters:
    - fieldset: the fieldset to which the displacement field will be added.
    - file_path: the path to the file containing the velocity field. Default is a u-velocity file (01-01-1990).
    Output: the fieldset with the displacement field added.
    """
    landmask = make_landmask(file_path)
    v_x, v_y = create_displacement_field(landmask)  
    d_2_s = distance_to_shore(landmask)
    fieldset.add_field(
        Field(
            "disp_u",
            data=v_x,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth=fieldset.U.grid.depth,
            mesh="spherical",
        )
    )
    fieldset.add_field(
        Field(
            "disp_v",
            data=v_y,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth=fieldset.U.grid.depth,
            mesh="spherical",
        )
    )
    # adjust units from m/s to °/s
    fieldset.disp_u.units = GeographicPolar()
    fieldset.disp_v.units = Geographic()

    fieldset.add_field(
        Field(
            "landmask",
            landmask,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth=fieldset.U.grid.depth,
        )
    )
    fieldset.add_field(
        Field(
            "distance2shore",
            d_2_s,
            lon=fieldset.U.grid.lon,
            lat=fieldset.U.grid.lat,
            depth=fieldset.U.grid.depth,
        )
    )
    return fieldset


def set_displacement(particle, fieldset, time):
    """Kernel that sets the displacement (particle.d_u, particle.d_v) of the particle based on the distance to shore.
    This Kernel is used in a parcels simulation (`pset.execute`) to displace particles that are stuck on land.

    ## Parameters:
    - particle: class, the particle to be displaced.
    - fieldset: class, the fieldset containing the distance to shore.
    - time: int, the time at which the particle is being displaced.
    """
    # compute distance to shore
    particle.d2s = fieldset.distance2shore[
        time, particle.depth, particle.lat, particle.lon
    ]
    # if the particle is within 0.5 dx of the shore, set the displacement
    if particle.d2s < 0.5:
        disp_u_ab = fieldset.disp_u[time, particle.depth, particle.lat, particle.lon]
        disp_v_ab = fieldset.disp_v[time, particle.depth, particle.lat, particle.lon]
        particle.d_u = disp_u_ab *0.5
        particle.d_v = disp_v_ab *0.5
    else:
        particle.d_u = 0.0
        particle.d_v = 0.0


def displace(particle, fieldset, time):
    """Kernel that displaces the particle based on the displacement (particle.d_u, particle.d_v) set by the `set_displacement` kernel.
    This Kernel is used in a parcels simulation (`pset.execute`) to displace particles that are stuck on land.

    ## Parameters:
    - particle: the particle to be displaced.
    - fieldset: the fieldset containing the displacement field. (not accessed in this function)
    - time: the time at which the particle is being displaced. (not accessed in this function)
    """
    # if the particle is within 0.5 dx of the shore, displace it
    if particle.d2s < 0.5:
        # take the minimum of the displacement and 0.05 degrees in both + and - direction
        particle_dlon += math.copysign(1,particle.d_u)*min(math.fabs(particle.d_u * particle.dt),0.05)#],key = math.fabs)
        particle_dlat += math.copysign(1,particle.d_v)*min( math.fabs(particle.d_v *particle.dt),0.05)#*math.copysign(1,particle.d_v)],key = math.fabs)


def make_landmask(fielddata):
    """Returns landmask where land = 1 and ocean = 0

    # Parameters:
    - fielddata: is a netcdf file. Should contain a U-velocity field (`UVEL`). Otherwise adjust the function accordingly.
    """
    # read the landmask from U-velocity file
    datafile = xr.open_dataset(fielddata)
    landmask = datafile.variables["UVEL"][0]
    landmask = np.ma.masked_invalid(landmask)
    landmask = landmask.mask.astype("int")

    return landmask

def get_coastal_shore_nodes(landmask,Z_MOD = 1):
    """Function that detects the coastal nodes, i.e. the ocean nodes directly
    next to land. Computes the Laplacian of landmask.

    ## Parameters:
    - landmask: the land mask built using `make_landmask`, where land cell = 1 and ocean cell = 0.
    - Z_MOD: modifier for 3D. Set to 0 for 2D and 1 for 3D, if field has the shape [depth, lat, lon].

    Output: 2D array array containing the coastal nodes, the coastal nodes are
            equal to one, and the rest is zero.
    """
    
    # Laplacian
    mask_lap = np.roll(landmask, -1, axis=0+Z_MOD) + np.roll(landmask, 1, axis=0+Z_MOD)
    mask_lap += np.roll(landmask, -1, axis=1+Z_MOD) + np.roll(landmask, 1, axis=1+Z_MOD)
    mask_lap -= 4 * landmask

    # define coastal nodes and shore nodes
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype("int")
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return coastal,shore

def get_coastal_shore_nodes_diagonal(landmask,Z_MOD = 1):
    """Function that detects the coastal nodes, i.e. the ocean nodes where
    one of the 8 nearest nodes is land. Computes the Laplacian of landmask
    and the Laplacian of the 45 degree rotated landmask.
    
    ## Parameters:
    - landmask: the land mask built using `make_landmask`, where land cell = 1 and ocean cell = 0.
    - Z_MOD: modifier for 3D. Set to 0 for 2D and 1 for 3D, if field has the shape [depth, lat, lon].

    Output: 2D array array containing the coastal nodes, the coastal nodes are
            equal to one, and the rest is zero.
    """
    # Laplacian
    mask_lap = np.roll(landmask, -1, axis=0+Z_MOD) + np.roll(landmask, 1, axis=0+Z_MOD)
    mask_lap += np.roll(landmask, -1, axis=1+Z_MOD) + np.roll(landmask, 1, axis=1+Z_MOD)
    
    # Include diagonal neighbors
    mask_lap += np.roll(landmask, (-1, 1), axis=(0+Z_MOD, 1+Z_MOD)) + np.roll(
        landmask, (1, 1), axis=(0+Z_MOD, 1+Z_MOD)
    )
    mask_lap += np.roll(landmask, (-1, -1), axis=(0+Z_MOD, 1+Z_MOD)) + np.roll(
        landmask, (1, -1), axis=(0+Z_MOD, 1+Z_MOD)
    )
    mask_lap -= 8 * landmask

    # define coastal nodes and shore nodes
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype("int")
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return coastal,shore

def create_displacement_field(landmask,Z_MOD=1):
    """Function that creates a displacement field 1 m/s away from the shore.
    
    ## Parameters:
    - landmask: the land mask built using `make_landmask`.
    - Z_MOD: modifier for 3D. Set to 0 for 2D and 1 for 3D, if field has the shape [depth, lat, lon].

    Output: two 2D arrays, one for each component of the velocity.
    """
    shore = get_coastal_shore_nodes(landmask)[1]

    # nodes bordering ocean directly and diagonally
    shore_d = get_coastal_shore_nodes_diagonal(landmask)[1]
    # corner nodes that only border ocean diagonally
    shore_c = shore_d - shore

    # Simple derivative
    l_y = np.roll(landmask, -1, axis=0+Z_MOD) - np.roll(landmask, 1, axis=0+Z_MOD)
    l_x = np.roll(landmask, -1, axis=1+Z_MOD) - np.roll(landmask, 1, axis=1+Z_MOD)

    l_y_c = np.roll(landmask, -1, axis=0+Z_MOD) - np.roll(landmask, 1, axis=0+Z_MOD)
    # Include y-component of diagonal neighbors
    l_y_c += np.roll(landmask, (-1, -1), axis=(0+Z_MOD, 1+Z_MOD)) + np.roll(
        landmask, (-1, 1), axis=(0+Z_MOD, 1+Z_MOD)
    )
    l_y_c += -np.roll(landmask, (1, -1), axis=(0+Z_MOD, 1+Z_MOD)) - np.roll(
        landmask, (1, 1), axis=(0+Z_MOD, 1+Z_MOD)
    )

    l_x_c = np.roll(landmask, -1, axis=1+Z_MOD) - np.roll(landmask, 1, axis=1+Z_MOD)
    # Include x-component of diagonal neighbors
    l_x_c += np.roll(landmask, (-1, -1), axis=(1+Z_MOD, 0+Z_MOD)) + np.roll(
        landmask, (-1, 1), axis=(1+Z_MOD, 0+Z_MOD)
    )
    l_x_c += -np.roll(landmask, (1, -1), axis=(1+Z_MOD, 0+Z_MOD)) - np.roll(
        landmask, (1, 1), axis=(1+Z_MOD, 0+Z_MOD)
    )

    v_x = -l_x * (shore)
    v_y = -l_y * (shore)

    v_x_c = -l_x_c * (shore_c)
    v_y_c = -l_y_c * (shore_c)

    v_x = v_x + v_x_c
    v_y = v_y + v_y_c

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal nodes between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    nz,ny, nx = np.where(magnitude == 0)
    magnitude[nz,ny, nx] = 1

    v_x = v_x / magnitude
    v_y = v_y / magnitude

    return v_x, v_y

def distance_to_shore(landmask, dx=1):
    """Function that computes the distance to the shore. It is based in the
    the `get_coastal_shore_nodes` & `get_coastal_shore_nodes_diagonal` algorithm.

    ## Parameters:
    - landmask: the land mask built using `make_landmask` function.
    - dx: the grid cell dimension. This is a crude approximation of the real distance (be careful).

    Output: 2D array containing the distances from shore.
    """
    # direct neighbors get a distance of dx
    ci = get_coastal_shore_nodes(landmask)[0]  
    dist = ci * dx  

    # diagonal neighbors get a distance of sqrt(2)*dx
    ci_d = get_coastal_shore_nodes_diagonal(landmask)[0]  
    dist_d = (ci_d - ci) * np.sqrt(2 * dx**2)  

    d_2_s = dist + dist_d
    d_2_s[(landmask==0)&(ci_d==0)]=2

    return d_2_s

