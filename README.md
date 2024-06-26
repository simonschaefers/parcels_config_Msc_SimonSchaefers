# Documentation for a OceanParcels setup with POP model velocities for determining the origin of Atlantic AAIW

### Content 

- `advection_routines`: folder that allows for advection and basic analysis of particles in a repeated year framework
  - `advection_execution.py`: file includes the main advection routine `full_execution`, which allows for advection for a given time, `merge_zarr` allows for re-chunking and concatenation of output over multiple iterations.
  - `displacement.py`: file includes an optional setup for a displacement of particles that get stuck along the boundaries due to the Arakawa B-grid configuration of the POP model. The displacement kernel is a 3d interpretation of the displacement routine found in the [OceanParcels Documentation](https://docs.oceanparcels.org/en/latest/examples/documentation_stuck_particles.html) 
  - `parcels_routines.py`: file includes the core parts of the advection routine, following the OceaParcels syntax of fieldset, particle set and kernels. Further, it includes a start set for particles based on density (`find_density`), and a routine to remove stuck particles from advection (`delete_stuck_particles`).
  - `post_process.py`: file includes two classes that allow for creation and plotting of metadata from a dataset produced by `advection_routines.merge_zarr`. `PlotableClass` allows for plotting of ventilation position and transit time distributions, `AnalyseTrajectory` produces metadata, such as start and ventilation position for each trajectory, that are easier to handle than the whole dataset.

- `vizualisation+results`: folder to empathize results and general mechanisms of the particle advection
  
### General Information

- Velocity fields are not included in this documentation, the setup still relies on precise (hard coded) filenames etc.
- The execution is performed with a config file, restarting the `full_execution` after 2 years of advection, due to limited computation time.

### Development

- Upcoming uploads
  - config file and bash scripts for the execution 
  - advanced analysis of output dataset
- extended soft coding
- run time saving adjustments in `advection_routines/parcels_routines.delete_stuck_particles`
