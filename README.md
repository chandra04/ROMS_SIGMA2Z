This script converts the standard ROMS sigma coordinate dataset to a standard Z grid. Every variable is regridded to rho grid.

A user need to do the changes in run_roms2z.py only, roms2z_lib.py sould be unchanged.

It is a parallel program. User can give the numebr of processor, time steps and the vertical depth.
