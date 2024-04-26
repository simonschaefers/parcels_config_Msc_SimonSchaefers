########################################################################################################
# Author: Simon Schäfers
# Date: 2024-02-16
# This script is part of the master thesis project "Eddy effects on South Atlantic Ventilation Pathways 
# using Lagrangian trajectories"

# Last modified: 2024-04-20
########################################################################################################

# This script contains a class to analyse the trajectories of advected particles.
# The class `AnalyseTrajectory` produces and loads metadata as a pandas dataframe.
# The metadata can be produced in parallel, or, if already present, be read from a csv file.
# A plotable class is inherited by the `AnalyseTrajectory` class to allow plotting of ventilation events.
# The data worked with is the output of the advection of particles, as carried out in `advection_routines.advection_execution.full_execution`.



import os
import zarr
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import matplotlib.colors as colors


class PlotableClass:
    """Class to be inherited by class `AnalyseTrajectory` to allow plotting of ventilation events.
    Data worked with is the output of the advection of particles, as carried out in `advection_routines.advection_execution.full_execution`.
    Ventilation events are defined as the first time a particle reaches the mixed layer, as the advection is backward in time.

    ## Methods:
    - plot_ventilations: scatter plot of the positions of first ventilation + contour of the landmask.
    - plot_transit_time: histogram of the transit time of the ventilated particles.
    - plot_ventilation_season: histogram of the time in the year of the ventilation events.
    - plot_spatial_distribution: histogram of the spatial distribution (latitude or longitude) of the ventilated particles.
    - show_ventilation_overview: shows all plots in one figure.

    all methods take the same arguments:
    - ax: the axis on which the plot will be drawn. For `show_ventilation_overview`, if `ax` is not given, a new figure is created.
    - kwargs: the specifications for the plot.
        - mask: a mask for criteria like ventilation region. If not given, all particles that ventilated are considered.
        - N_BINS: number of bins for a histogram. Default is 120 for `plot_transit_time` and 365 for `plot_ventilation_season`.
        - label: label for the histogram in `plot_transit_time` and `plot_ventilation_season`.
        - isLat: if given, the histogram in `plot_spatial_distribution` will be of the latitude of the ventilations. Otherwise, it will be of the longitude.

    """

    def plot_ventilations(self, ax, **kwargs):
        """Scatter plot of the positions of first ventilation + contourf of the landmask.
        For parameters, see `PlotableClass`.
        """
        # optionally specify kwargs

        accessible_kwargs = ['vent_definition','mask','vent_bin_size','cmap','vlims','alpha','color']
        accessible_kwarg_presets = {'vent_definition':'first','vent_bin_size':1,'cmap':'viridis','vlims':[1,1000],'alpha':1,'color':'black'}
        for key in kwargs.keys():
            if key in accessible_kwargs:
                globals()[key] = kwargs[key]
                kwargs.pop(key)
            else:
                if key == 'mask':
                    mask = (self.df[vent_definition+'_ventilated']>0)
                else:
                    globals()[key] = accessible_kwarg_presets[key]


        # define bins for histogram
        vent_bins = [np.arange(0,360,vent_bin_size),np.arange(-75,20,vent_bin_size)]
        # plot lon, lat of ventilations
        hist_values = np.histogram2d(self.df[vent_definition+'_ventilated_lon'][mask],self.df[vent_definition+'_ventilated_lat'][mask],bins = vent_bins)
        hist_values[0][hist_values[0]<vmin/2] = np.nan

        lvls = np.logspace(np.log10(vlims[0]),np.log10(vlims[1])+1,int(np.log10(vlims[1])-np.log10(vlims[0])+2))#.astype(int)
        #ax.pcolor(hist_values[1][1:],hist_values[2][1:],hist_values[0].T,cmap = cmap,norm=colors.LogNorm(vmax  = vmax/100,vmin = vmin),alpha = alpha)
        ax.contour(hist_values[1][1:],hist_values[2][1:],hist_values[0].T,cmap = cmap,levels = lvls[:-1],norm=colors.LogNorm(vmax  = vlims[1],vmin = vlims[0]/10),alpha = 1)
        ax.contourf(hist_values[1][1:],hist_values[2][1:],hist_values[0].T,cmap = cmap,levels = lvls[2:],norm=colors.LogNorm(vmax  = vlims[1],vmin = vlims[0]/10),alpha = alpha)

        ax.bar(-100,-100,color = color,label = '%i total particles'%len(self.df[vent_definition+'_ventilated'][mask]))

        # plot landmask
        u_mask_ds = xr.open_dataset('YOUR-DIRECTORY-HERE/data_1990/nest_small_19900101000000u.nc')
        u_mask = np.isnan(u_mask_ds.UVEL[0,0])
        # make nan where water
        u_mask = np.where(u_mask,0,np.nan)
        ax.contourf(u_mask_ds.XU,u_mask_ds.YU,u_mask,alpha = 1,cmap = 'Grays')

        ax.set_xlim([0,360])
        ax.set_ylim([-75,20])
        ax.set_title('Position of ventilations')
    
    def plot_transit_time(self,ax,**kwargs):
        """Histogram of the transit time of the ventilated particles.
        For parameters, see `PlotableClass`.
        """

        # get bins for histogram or default to 120
        if 'N_BINS' in kwargs:
            N_BINS = kwargs['N_BINS']
            kwargs.pop('N_BINS')
        else: 
            N_BINS = 120

        # remove label from kwargs to avoid double labeling
        if 'label' in kwargs:   
            kwargs.pop('label')

        # optionally specify vent_definition (first MLD contact or depth (100m, 10m))
        if 'vent_definition' in kwargs:
            vent_definition = kwargs['vent_definition']
            kwargs.pop('vent_definition')
        else: vent_definition = 'first'

        # read mask for ventilated particles or include all ventilated particles
        if 'mask' in kwargs:
            mask = kwargs['mask']  
            kwargs.pop('mask')
        else: mask= (self.df[vent_definition+'_ventilated']>0)

        # make histogram
        ax.hist(self.df[vent_definition+'_ventilated'][mask]/365,
                bins=np.arange(0,120+120/N_BINS,120/N_BINS),weights=self.df['volume'][mask]/np.average(self.df['volume'][mask]),**kwargs)
        #ax.set_title('Transit time')
        ax.set_xlim([0,120])

    def plot_ventilation_season(self,ax,**kwargs):
        """Histogram of the time in the year of ventilation events.
        For parameters, see `PlotableClass`.
        """

        # get bins for histogram or default to 12 (one for each month)
        if 'N_BINS' in kwargs:
            N_BINS = kwargs['N_BINS']
            kwargs.pop('N_BINS')
        else: 
            N_BINS = 12

        # optionally specify vent_definition (first MLD contact or depth (100m, 10m))
        if 'vent_definition' in kwargs:
            vent_definition = kwargs['vent_definition']
            kwargs.pop('vent_definition')
        else: vent_definition = 'first'

        # read mask for ventilated particles or include all ventilated particles
        if 'mask' in kwargs:
            mask = kwargs['mask']  
            kwargs.pop('mask')
        else: mask= (self.df[vent_definition+'_ventilated']>0)

        # define ticks and labels for the histogram
        months = ['J','F','M','A','M','J','J','A','S','O','N','D']
        seasons = ['DJF','MAM','JJA','SON']
        N_TICKS = min(12,N_BINS)        
        shift = -30 if N_BINS == 4 else 0

        # make histogram and set ticks accordingly       
        ax.hist(365-(self.df[vent_definition+'_ventilated'][mask]+shift)%365,N_BINS,weights=self.df['volume'][mask]/np.average(self.df['volume'][mask]),**kwargs)
        ax.set_xticks(ticks = np.arange(365/(2*N_TICKS),365+365/(2*N_TICKS),365/N_TICKS),
                      labels = months if N_BINS!=4 else seasons,fontweight  ='bold')
        ax.set_title('Season of ventilation',fontsize = 16,fontweight='bold')
        ax.set_yticks([])


    def plot_spatial_distribution(self,ax,**kwargs):
        """Histogram of the spatial distribution (latitude or longitude) of the ventilated particles.
        For parameters, see `PlotableClass`.
        """

        # optionally specify vent_definition (first MLD contact or depth (100m, 10m))
        if 'vent_definition' in kwargs:
            vent_definition = kwargs['vent_definition']
            kwargs.pop('vent_definition')
        else: vent_definition = 'first'
        # read mask for ventilated particles or include all ventilated particles
        if 'mask' in kwargs:
            mask = kwargs['mask']  
            kwargs.pop('mask')
        else: mask= (self.df[vent_definition+'_ventilated']>0)
        
        # make histogram of latitude or longitude
        if 'isLat' in kwargs:
            kwargs.pop('isLat')
            ax.hist(self.df[vent_definition+'_ventilated_lat'][mask],range = [-80,20],bins=50,orientation='horizontal',weights=self.df['volume'][mask]/np.average(self.df['volume'][mask]),**kwargs) 
            ax.set_ylim([-75,20])
            #ax.set_title('Latitude of ventilation')
        else:
            # make histogram from top to bottom
            # make historgram and another with x+360

            ax.hist(self.df[vent_definition+'_ventilated_lon'][mask],range = [0,360],bins=180,weights=self.df['volume'][mask]/np.average(self.df['volume'][mask]),**kwargs)
            ax.hist(self.df[vent_definition+'_ventilated_lon'][mask]+360,range = [360,720],bins=180,weights=self.df['volume'][mask]/np.average(self.df['volume'][mask]),**kwargs)
            ax.set_xlim([200,200+360])
            #ax.set_title('Longitude of ventilation')   


    def show_ventilation_overview(self,ax = None,**kwargs):
        """Shows all plots in one figure.
        For parameters, see `PlotableClass`.
        """

        # optionally specify vent_definition (first MLD contact or depth (100m, 10m))
        if 'vent_definition' in kwargs:
            vent_definition = kwargs['vent_definition']
        else: vent_definition = 'first'
        # if a box is given, use it as mask
        if 'box' in kwargs:
            box = self.ventilation_areas[kwargs['box']]
            kwargs.pop('box')
            mask = ((self.df[vent_definition+'_ventilated_lon'] > box[0][0]) &
                    (self.df[vent_definition+'_ventilated_lon'] < box[0][1]) &
                    (self.df[vent_definition+'_ventilated_lat'] > box[1][0]) &
                    (self.df[vent_definition+'_ventilated_lat'] < box[1][1]) )
            kwargs['mask'] = mask

        # if no box and no mask is given, use all ventilated particles
        elif 'mask' not in kwargs:
            mask = (self.df[vent_definition+'_ventilated'] > 0)
            kwargs['mask'] = mask

        # create figure if ax not given
        if ax is None:
            AX = None
            fig,ax = plt.subplots(1,5,figsize=(30,5),gridspec_kw={'width_ratios':[2,1,1,1,1]})
        else: AX = True
        # plot all plot methods with specified kwargs
        if 'colormap' in kwargs:
            cmap = kwargs['colormap']
            kwargs.pop('colormap')
        else:
            cmap = 'Oranges'
        if 'vlims' in kwargs:
            vlims = kwargs['vlims']
            kwargs.pop('vlims')
        else:
            vlims = [1,1000]
        #self.plot_ventilations(ax[0],s = 1,cmap = cmap, vlims  = vlims,vent_bin_size = 2, **kwargs)
        self.plot_transit_time(ax[3],N_BINS = 60,**kwargs)
        self.plot_ventilation_season(ax[4],N_BINS = 12,**kwargs) 
        self.plot_spatial_distribution(ax[2],**kwargs)
        self.plot_spatial_distribution(ax[1],isLat='lat',**kwargs)        
        
        ax[0].set_title(self.name,fontweight='bold',fontsize=20)

        # if no ax is given, finalize the figure
        if AX is None:
            plt.tight_layout()
            return fig,ax
        else:
            return ax


class AnalyseTrajectory(PlotableClass):
    """Class to analyse the trajectories of advected particles.
    Data worked with is the output of the advection of particles, 
    as carried out in `advection_routines.advection_execution.fullExecution` and merged with `advection_routines.advection_execution.rechunk_and_merge`.
    Aim of this class is to provide memory saving metadata as a pandas dataframe. 
    The meatadata can be produced in parallel, or, if already present, be read from a csv file.

    ## Dataframe columns:
    ### ---- General -----
    - useful: a boolean array to filter particles that are useful for analysis (not getting stuck in the first 60 years)
    - to_end: a boolean array to filter particles that are useful for analysis (not getting stuck in the whole 120 years)
    - start_lon: the longitude of the starting position
    - start_lat: the latitude of the starting position
    - start_depth: the depth of the starting position
    ### ----- Ventilation ------
    [ventilation events are defined differently, here X can be: first: first contact with daily mixed layer; mean: first contact with seasonal mean mixed layer; 10, 100: first contact with depth 10m, 100m]
    - ventilated: a boolean array to filter particles that ventilated
    - X_ventilated: the time of the X ventilation event
    - X_ventilated_lon: the longitude of the X ventilation event
    - X_ventilated_lat: the latitude of the X ventilation event
    - X_ventilated_depth: the depth of the X ventilation event
    - NO_VENTILATED: the number of ventilation events
    ### ----- Pathways ------
    - drake_passage: the time of the first drake passage crossing
    - agulhas: the time of the first agulhas leakage crossing
    - agulhas_extended: the time of the first agulhas leakage crossing, extended to 45°S
    - madagascar_strait: the time of the first madagascar strait crossing
    - indonesian_throughflow: the time of the first indonesian throughflow crossing
    - tasman_sea: the time of the first tasman sea crossing
 
    ## Methods:
    - __init__: initializes the class.
    - __getitem__: creates metadata that are easily accessible.
    - __call__: creates metadata whose calculation is more complex.
    - make_bins: creates bins for the horizontal position of a trajectory.
    - start_array: initializes the array to be filled with metadata.
    - update_array: fills the array with metadata in parallel.
    - collect_ps: collects the metadata from the parallel processes.
    - parallelize_datacollection: the function that is parallelized and produces the metadata.
    - conclude: creates a pandas dataframe from the metadata.
    - save_pandas: saves the pandas dataframe to a csv file.

    
    ## Example:
    ```python
    from tqdm import tqdm
    name='AAIW_'
    A = AnalyseTrajectory(name,update = True)
    A.start_array()
    n = 200
    for i in tqdm(range(0,len(A.full_ds.lon),n)):
        A.update_array(entries = np.arange(i,min(i+n,len(A.full_ds.lon)),1))
    A.conclude()
    A.save_pandas()
    """

    def __init__(self,name,MAX_JOBS = 0.4,update = False):
        """Initializes the class. If update is set to True, the metadata will be updated. Otherwise, it will be read from a csv file.
        If the metadata is to be updated, the class will be used in combination with the `start_array`, `update_array`, and `conclude` methods.
        Here, the setup for the parallelization is done.
        ## Parameters:
        - name: str, the name of the simulation.
        - MAX_JOBS: float, the fraction of the available threads to be used for parallelization.
        - update: bool, whether the metadata should be updated or read from a csv file.
        
        ## Attributes:
        - full_ds: the zarr dataset with the full trajectory data.
        - start_ds: the zarr dataset with the trajectory for the first year(s).
        - mid_ds: the zarr dataset with the trajectory data after 60 years.
        - end_ds: the zarr dataset with the trajectory data after 120 years.
        - name: str, the name of the simulation.
        - update: bool, whether the metadata should be updated or only read from a csv file.

        - ventilation_areas: dict, predefined areas to determine ventilation regions
        

        """
        # open and include zarr datasets, simulation name, update status, and predefined ventilation areas
        self.full_path = 'YOUR-DIRECTORY-HERE'
        try: 
            self.full_ds = zarr.open(self.full_path+'zarr_storage/combined_%s.zarr'%name,mode = 'r')
            self.start_ds = zarr.open(self.full_path+'zarr_storage/%s_0.zarr'%name,mode = 'r')
            self.mid_ds = zarr.open(self.full_path+'zarr_storage/%s_60.zarr'%name,mode = 'r')
            self.end_ds = zarr.open(self.full_path+'zarr_storage/%s_120.zarr'%name,mode = 'r')
        except:
            print('no zarr datasets found')
        self.name = name
        self.update = update
        self.ventilation_areas= {'Drake':[[260,315],[-63,-45]],
                                'Indian Ocean':[[40,115],[-45,-15]],
                                'South Atlantic1':[[305,360],[-40,-20]],
                                'South Atlantic2':[[0,20],[-40,-20]],
                                'South Pacific':[[180,260],[-55,-30]],
                                'Australia':[[115,180],[-55,-30]],
        }
        
        # access pandas dataframe if it exists
        if self.update == False:
            if os.path.exists(self.full_path+'metadata/%s.csv'%self.name):
                self.df = pd.read_csv(self.full_path+'metadata/%s.csv'%self.name)
            else:
                print('[WARNING] no pandas dataframe found, set update to true to create one')
        
        # prepare parallelization if update is set to True
        else:
            if os.path.exists(self.full_path+'metadata/%s.csv'%self.name):
                self.df = pd.read_csv(self.full_path+'metadata/%s.csv'%self.name) 
                print('[WARNING] pandas dataframe found, this configuration allows to update')       
            self.MAXIMUM_JOBS = int(MAX_JOBS*mp.cpu_count())
            self.running_jobs = []       # Processes
            self.open_queues  = []       # Queues
            print('maximum jobs for process:', self.MAXIMUM_JOBS)
        return
    
    def __getitem__(self,arg):
        """Creates metadata that are easily accessible.
        For parameters, see `AnalyseTrajectory`.
        """
        if arg == 'useful': return (self.mid_ds.lon[:,0]!=300)|(self.mid_ds.lat[:,0]!=0)|(self.mid_ds.z[:,0]!=1000)
        elif arg == 'to_end':return (self.end_ds.lon[:,0]!=300)|(self.end_ds.lat[:,0]!=0)|(self.end_ds.z[:,0]!=1000)
        elif arg ==  'start_lon': return self.start_ds.lon[:,0]
        elif arg == 'start_lat': return self.start_ds.lat[:,0]
        elif arg == 'start_depth': return self.start_ds.z[:,0]
        elif arg == 'ventilated': return self.end_ds.n_up[:,-1]
        
        else: return 'not a valid argument'

    def __call__(self,arg,i):
        """Creates metadata whose calculation is more complex.
        For parameters, see `AnalyseTrajectory`.
        """
    
        # for ventilation, the first time step that matches the criteria (ventilation) is returned, or -1 if no ventilation
        if arg == 'first_ventilated': return np.where(self.full_ds.n_up[i]!=0)[0][0] if np.any(self.full_ds.n_up[i]!=0) == True else -1
        elif arg == 'first_ventilated_lon': return self.full_ds.lon[i,np.where(self.full_ds.n_up[i]!=0)[0][0]] if np.any(self.full_ds.n_up[i]!=0) == True else -1
        elif arg == 'first_ventilated_lat': return self.full_ds.lat[i,np.where(self.full_ds.n_up[i]!=0)[0][0]] if np.any(self.full_ds.n_up[i]!=0) == True else -1
        elif arg == 'first_ventilated_depth': return self.full_ds.z[i,np.where(self.full_ds.n_up[i]!=0)[0][0]] if np.any(self.full_ds.n_up[i]!=0) == True else -1
        #elif arg == 'NO_VENTILATED': return len(np.where((self.full_ds.n_up[i,1:]-self.full_ds.n_up[i,:-1]!=0)&(self.full_ds.n_up[i,1:]!=0))[0]) if np.any(self.full_ds.n_up[i]!=0) == True else -1
        
        elif arg == 'mean_ventilated': return np.where(self.full_ds.n_up_mean[i]!=0)[0][0] if np.any(self.full_ds.n_up_mean[i]!=0) == True else -1
        elif arg == 'mean_ventilated_lon': return self.full_ds.lon[i,np.where(self.full_ds.n_up_mean[i]!=0)[0][0]] if np.any(self.full_ds.n_up_mean[i]!=0) == True else -1
        elif arg == 'mean_ventilated_lat': return self.full_ds.lat[i,np.where(self.full_ds.n_up_mean[i]!=0)[0][0]] if np.any(self.full_ds.n_up_mean[i]!=0) == True else -1
        elif arg == 'mean_ventilated_depth': return self.full_ds.z[i,np.where(self.full_ds.n_up_mean[i]!=0)[0][0]] if np.any(self.full_ds.n_up_mean[i]!=0) == True else -1

        elif arg =='10_ventilated': return np.where(self.full_ds.z[i,:]<10)[0][0] if np.any(self.full_ds.z[i,:]<10)==True else -1
        elif arg =='10_ventilated_lon': return self.full_ds.lon[i,np.where(self.full_ds.z[i,:]<10)[0][0]] if np.any(self.full_ds.z[i,:]<10)==True else -1
        elif arg =='10_ventilated_lat': return self.full_ds.lat[i,np.where(self.full_ds.z[i,:]<10)[0][0]] if np.any(self.full_ds.z[i,:]<10)==True else -1
        
        elif arg =='100_ventilated': return np.where(self.full_ds.z[i,:]<100)[0][0] if np.any(self.full_ds.z[i,:]<100)==True else -1
        elif arg =='100_ventilated_lon': return self.full_ds.lon[i,np.where(self.full_ds.z[i,:]<100)[0][0]] if np.any(self.full_ds.z[i,:]<100)==True else -1
        elif arg =='100_ventilated_lat': return self.full_ds.lat[i,np.where(self.full_ds.z[i,:]<100)[0][0]] if np.any(self.full_ds.z[i,:]<100)==True else -1

        # for crossing of straits, the first time step that matches the criteria (crossing) is returned, or -1 if no crossing
        elif arg == 'drake_passage': 
            tmp = np.where((self.full_ds.lon[i,1:] < 295) & (self.full_ds.lon[i,:-1] > 295) & (self.full_ds.lat[i,1:] < -55) & (self.full_ds.lat[i,1:] > -65) & (self.full_ds.lon[i,1:]-self.full_ds.lon[i,:-1] < 180))[0]
            return tmp[0] if len(tmp) > 0 else -1
        elif arg == 'agulhas': 
            tmp= np.where((self.full_ds.lon[i,1:] > 30) & (self.full_ds.lon[i,:-1] < 30) & (self.full_ds.lat[i,1:] < -30) & (self.full_ds.lat[i,1:] > -40) & (self.full_ds.lon[i,1:]-self.full_ds.lon[i,:-1] < 180))[0]
            return tmp[0] if len(tmp) > 0 else -1
        elif arg == 'agulhas_extended': 
            tmp= np.where((self.full_ds.lon[i,1:] > 30) & (self.full_ds.lon[i,:-1] < 30) & (self.full_ds.lat[i,1:] < -30) & (self.full_ds.lat[i,1:] > -45) & (self.full_ds.lon[i,1:]-self.full_ds.lon[i,:-1] < 180))[0]
            return tmp[0] if len(tmp) > 0 else -1
        elif arg == 'madagascar_strait': 
            tmp = np.where((self.full_ds.lon[i,1:] > 35) & (self.full_ds.lon[i,1:] < 45) & (self.full_ds.lat[i,1:] > -20) & (self.full_ds.lat[i,:-1] < -20) & (self.full_ds.lon[i,1:]-self.full_ds.lon[i,:-1] < 180))[0]
            return tmp[0] if len(tmp) > 0 else -1
        elif arg == 'indonesian_throughflow': 
            indonesia = (self.full_ds.lon[i,1:] > 105) & (self.full_ds.lon[i,1:] < 150) & (self.full_ds.lat[i,1:] > -5) & (self.full_ds.lat[i,:-1] < -5) & (self.full_ds.lon[i,1:]-self.full_ds.lon[i,:-1] < 100)
            timor = (self.full_ds.lon[i,1:] > 142) & (self.full_ds.lon[i,:-1] < 142) & (self.full_ds.lat[i,1:] > -15) & (self.full_ds.lat[i,1:] < -5) & (self.full_ds.lon[i,1:]-self.full_ds.lon[i,:-1] < 100)
            trough = indonesia | timor
            tmp =  np.where(trough == True)[0]
            return tmp[0] if len(tmp) > 0 else -1
        elif arg == 'tasman_sea': 
            tmp = np.where((self.full_ds.lon[i,1:] > 150) & (self.full_ds.lon[i,1:] < 175) & (self.full_ds.lat[i,1:] > -37) & (self.full_ds.lat[i,:-1] < -37) & (self.full_ds.lon[i,1:]-self.full_ds.lon[i,:-1] < 100))[0]
            return tmp[0] if len(tmp) > 0 else -1
        
        elif arg == 'make_bins': return self.make_bins(i)

        else: raise ValueError("Argument not valid")

    def make_bins(self,i,bin_size = 1):
        """Creates bins for the horizontal position of a trajectory.
        ## Parameters:
        - i: int, the index of the trajectory.
        - bin_size: int, the size of the bins in degrees.

        ## Returns:
        - hist: a boolean array of the bins that contain the trajectory.
        """
        
        lon_bins = np.arange(0,360+bin_size,bin_size)
        lat_bins = np.arange(-75,20+bin_size,bin_size)
        hist = np.histogram2d(self.full_ds.lon[i,:self.df[self.vent_config][i].astype(int)],
                              self.full_ds.lat[i,:self.df[self.vent_config][i].astype(int)],
                              bins = [lon_bins,lat_bins])
        
        return (hist[0]>0)

    def save_pandas(self):
        """Saves the pandas dataframe to a csv file.
        Only write if update is set to True.
        """
       
        if self.df is None: raise ValueError("No pandas DataFrame to save")
        elif self.update == False:
            print('[WARNING] update was not set to True, overwriting pandas DataFrame will not be possible')
            return
        if not os.path.exists("metadata"):
            os.makedirs("metadata")
        self.df.to_csv(self.full_path+'metadata/%s.csv'%self.name,sep = ',',index = False)
        print('saved')
        return

    def start_array(self,**kwargs):
        """Initializes the array to be filled with metadata.
        Only initialize if update is set to True.
        Can be used to specify the metadata to be collected (define arg_list in kwargs).
        If arg_list is not given, all metadata will be collected.
        """
        if self.update == False:
            print('[WARNING] update was not set to True, writing to pandas dataframe will not be possible')
            return
        
        # define the metadata to be collected, fast and slow calculable metadata
        fast_args = ['useful','to_end','start_lon','start_lat','start_depth','ventilated']
        slow_args = ['first_ventilated','first_ventilated_lon','first_ventilated_lat','first_ventilated_depth',
                     'mean_ventilated','mean_ventilated_lon','mean_ventilated_lat','mean_ventilated_depth',
                     '10_ventilated','10_ventilated_lon','10_ventilated_lat','100_ventilated','100_ventilated_lon','100_ventilated_lat',
                     'drake_passage','agulhas','agulhas_extended','madagascar_strait','indonesian_throughflow','tasman_sea']
        
        # if arg_list is given, use it, otherwise use all metadata
        self.arg_list = kwargs['arg_list'] if 'arg_list' in kwargs.keys() else fast_args+slow_args

            
        # separate fast and slow metadata
        self.parallel_arg_list = []
        for item in self.arg_list:
            if item in slow_args:
                self.parallel_arg_list.append(item)
        
        # remove slow metadata from arg_list
        for item in slow_args:
            if item in self.arg_list:
                self.arg_list.remove(item)

        # initialize the array to be filled with slow metadata, if make_bins is in the parallel_arg_list, only execute this
        if 'make_bins' in self.arg_list:
            self.vent_config = kwargs['vent_config'] if 'vent_config' in kwargs.keys() else 'first'
            self.vent_config += '_ventilated'
            self.parallel_arg_list = ['make_bins']
            self.temp_array = np.array(np.expand_dims(np.expand_dims(self.make_bins(0),axis = 0),axis = 0),dtype=bool)
        else:
            self.temp_array = np.array(self.parallel_arg_list)
        return 
    
    def update_array(self,**kwargs):
        """Fills the array with slow metadata in parallel.
        Only update if update is set to True.
        Collects the metadata from the parallel processes and initializes new processes.
        """

        if self.update == False:
            print('[WARNING] update was not set to True, writing to pandas dataframe will not be possible')
            return
        # collect metadata from parallel processes
        self.collect_ps()
        while len(self.running_jobs) >= self.MAXIMUM_JOBS:
            self.collect_ps()

        # if maximum number of processes is not reached, start a new process (parallelize_datacollection)
        q = mp.Queue()
        kw = kwargs.copy()
        kw["output_queue"] = q
        job = mp.Process(target=self.parallelize_datacollection, kwargs=kw)
        job.start()

        # append the new process and queue to the list of running processes and queues
        self.open_queues.append(q)
        self.running_jobs.append(job)
        return   

    def conclude(self):
        """Creates a pandas dataframe from the collected metadata.
        Only conclude if update is set to True.
        Should be used after the last update_array call.
        Collects remaining metadata from the parallel processes and adds fast metadata to the dataframe.
        """

        if self.update == False:
            print('[ERROR] update was not set to True, no data was collected')
            return
        
        # collect remaining metadata from parallel processes
        while len(self.running_jobs) > 0:
            self.collect_ps() 

        # create pandas dataframe from the collected metadata (temp_array) 
        if 'make_bins' in self.parallel_arg_list:
            #save temp_array as nc file
            ds = xr.Dataset({'particle_trajectories':(['particles','lon','lat'],self.temp_array[1:,0]),'lon':np.arange(0.5,360.5,1),'lat':np.arange(-74.5,20.5,1),'particles':np.arange(0,len(self.full_ds.lon))})
            if not os.path.exists("metadata"):
                os.makedirs("metadata")            
            ds.to_netcdf(self.full_path+'metadata/%s_%s_bins.nc'%(self.name,self.vent_config[:-11]))
        else:
            self.new_df = pd.DataFrame(self.temp_array[1:,:],columns = self.temp_array[0,:])
            
            # add fast metadata to the dataframe
            print(self.arg_list)
            for arg in self.arg_list:
                self.new_df[arg] = self[arg][:len(self.new_df)]
            
            # if the dataframe already exists, update the entires of the new dataframe
            if 'df' in self.__dict__:
                for arg in self.new_df.columns:
                    self.df[arg] = self.new_df[arg]
            else:
                self.df = self.new_df

            return self.df
    
    def collect_ps(self):
        """Collects the metadata from the parallel processes and adds it to temp_array.
        Used in `update_array` and `conclude`.
        """

        # If there are running processes, try to collect metadata from the parallel processes. 
        # If no metadata is available, break the loop.
        while len(self.running_jobs) > 0:
            try :
                data = self.open_queues[0].get(timeout=0.05)
            except:
                break

            self.temp_array = np.vstack([self.temp_array,data])

            # remove the finished job and queue
            self.running_jobs[0].join()
            self.running_jobs.pop(0)
            self.open_queues.pop(0)
        return

    def parallelize_datacollection(self,**kwargs):
        """The function that is parallelized and produces the metadata. 
        Used in `update_array`.
        calls the __call__ method for each entry in the entries list and appends the results to the output_queue.
        """

        # get the entries and output_queue from kwargs
        output_queue = kwargs['output_queue']
        entries = kwargs['entries']
        
        # compute the metadata for each entry in entries
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda i: [self(arg,i) for arg in self.parallel_arg_list], entries))
        results = np.array(results)

        # append the results to the output_queue
        output_queue.put(results)
        return

