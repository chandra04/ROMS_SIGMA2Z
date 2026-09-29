#!/usr/bin/env python3
"""
run_roms2z.py -- convert ROMS output (s-coordinate, rho/u/v/psi points) to a
regular longitude / latitude / depth grid for easy analysis.

Edit the SETTINGS block below, then run:

    python run_roms2z.py

Keep roms2z_lib.py in the same folder (or on your PYTHONPATH).
It holds all the functions and never needs editing.
"""

# =============================================================================
#                                   SETTINGS
# =============================================================================

# ---- files ------------------------------------------------------------------
INPUT_FILE  = "/Volumes/CHANDRA7/chapter_04_sensitivity/output/sens_01/his_sens_01_0010.nc"        # ROMS history / average file
OUTPUT_FILE = "ocean_his_z.nc"      # file to create
GRID_FILE   = None                  # ROMS grid file, only if lon/lat/mask/h/angle
                                    # are NOT in INPUT_FILE, e.g. "roms_grd.nc"

# ---- variables --------------------------------------------------------------
# "all"  -> every variable in the file
# list   -> only these, e.g. ["temp", "salt", "u", "v", "zeta"]
VARIABLES = "all"

# ---- depth levels of the output (m, positive downward) ----------------------
# "WOA"  -> World Ocean Atlas standard levels down to the deepest bathymetry
# list   -> your own levels, e.g. [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 500, 1000]
DEPTHS = [0, 5, 10, 15]#, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250,
          #300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 2500, 3000,
          #3500, 4000, 4500, 5000]

# ---- horizontal resolution of the regular grid ------------------------------
# The grid always covers the full model domain (no cropping).
#   "auto"            -> same as the model's own grid spacing (from pm/pn or
#                        the coordinates), e.g. a 4 km model gives ~0.036 deg
#   "5km" / "2500m"   -> spacing in km or m (converted to degrees for
#                        lon/lat grids, exact at the domain's centre latitude)
#   0.05 or "0.05deg" -> degrees (lon/lat grids only)
#   ("0.05deg", "0.04deg") -> different step in lon (x) and lat (y)
# Spherical models (lon_rho/lat_rho) give a lon/lat grid; Cartesian models
# (x_rho/y_rho, spherical = F) give a regular x/y grid in metres.
RESOLUTION = "auto"

# ---- parallel processing ----------------------------------------------------
NCORES = 8                          # number of processes; None = all CPU cores

# ---- optional ---------------------------------------------------------------
TIME_INDICES   = None               # None = all time steps; or e.g. range(0, 10)
ROTATE_VECTORS = True               # u/v, ubar/vbar, sustr/svstr, bustr/bvstr
                                    # -> eastward / northward components
BOTTOM_FILL    = True               # fill between deepest model level and seabed
MIN_WEIGHT     = 0.5                # 0-1: coastal coverage (lower = more coverage)
COMPRESS_LEVEL = 1                  # NetCDF compression 0-9 (0 = fastest, biggest)

# =============================================================================

from roms2z_lib import convert

if __name__ == "__main__":          # required for parallel processing
    convert(input_file=INPUT_FILE, output_file=OUTPUT_FILE, resolution=RESOLUTION,
            variables=VARIABLES, depths=DEPTHS, grid_file=GRID_FILE,
            ncores=NCORES, time_indices=TIME_INDICES,
            rotate_vectors=ROTATE_VECTORS, bottom_fill=BOTTOM_FILL,
            min_weight=MIN_WEIGHT, compress_level=COMPRESS_LEVEL)
