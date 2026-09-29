"""
roms2z_lib.py -- library for converting ROMS output from the native
curvilinear, terrain-following (s / sigma) grid to a regular
longitude / latitude / depth (z-level) grid.

    *** You should not need to edit this file. ***
    All settings go in the run script (run_roms2z.py), which calls convert().

What convert() does
-------------------
* Processes variables in order of their number of dimensions:
  0-D constants -> 1-D -> 2-D -> 3-D -> 4-D.
    - Variables that are NOT on the horizontal grid (Vtransform, hc, s_rho,
      Cs_r, ocean_time, ...) are copied unchanged.
    - Variables on RHO, U, V or PSI points are regridded to the regular
      lon/lat grid; those on s_rho / s_w levels are also interpolated to the
      requested depth levels.
* Works for any staggering: each variable is interpolated from its own
  native points with depths computed at those points.
* Level depths are rebuilt for every time step from zeta, h and the
  s-coordinate parameters (Vtransform 1/2, Vstretching 1-5), as in ROMS.
* Velocity pairs (u/v, ubar/vbar, sustr/svstr, bustr/bvstr) are moved to
  RHO points and rotated to eastward/northward using the grid 'angle'.
* The regular grid covers the full model domain (no cropping); points of the
  regular grid that fall outside the model domain or on land are _FillValue.
* Spherical grids (lon_rho/lat_rho) -> regular lon/lat grid.
  Cartesian grids (x_rho/y_rho, spherical = F) -> regular x/y grid in metres.
* Horizontal resolution is taken from the model grid spacing ("auto") or
  given in degrees, km or m (e.g. 0.05, "0.05deg", "5km", "2500m").
* Time steps are processed in parallel (multiprocessing); the main process
  writes the output and prints variable / time step / ETA on the terminal.

Requirements: numpy, scipy, netCDF4
"""

import os

# one thread per worker process: avoids oversubscription when running in parallel
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import sys
import time as _time
import warnings

import numpy as np
from netCDF4 import Dataset, num2date
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix

__version__ = "2.1"

GRID_OF_XI = {"xi_rho": "rho", "xi_u": "u", "xi_v": "v", "xi_psi": "psi"}
GRID_OF_ETA = {"eta_rho": "rho", "eta_u": "u", "eta_v": "v", "eta_psi": "psi"}
VERT_DIMS = ("s_rho", "s_w")
VECTOR_PAIRS = [("u", "v"), ("ubar", "vbar"), ("sustr", "svstr"), ("bustr", "bvstr")]
COORD_PREFIXES = ("lon_", "lat_", "x_", "y_")   # replaced by the new lon/lat axes

# World Ocean Atlas standard depth levels (m): 102 levels, 0-5500 m
WOA_DEPTHS = np.concatenate([np.arange(0, 100, 5), np.arange(100, 500, 25),
                             np.arange(500, 2000, 50),
                             np.arange(2000, 5501, 100)]).astype(float)

FILL = np.float32(1.0e20)
DROP_ATTRS = ("_FillValue", "missing_value", "coordinates", "grid", "location",
              "field", "valid_min", "valid_max", "valid_range")


# ==========================================================================
# ROMS vertical coordinate
# ==========================================================================
def s_levels(N, Vstretching, kgrid):
    """Nondimensional s at rho (kgrid=0) or w (kgrid=1) levels, -1 <= s <= 0."""
    lev = np.arange(0, N + 1, dtype=float) if kgrid else np.arange(1, N + 1) - 0.5
    if Vstretching == 5:
        return (-(lev * lev - 2 * lev * N + lev + N * N - N) / (N * N - N)
                - 0.01 * (lev * lev - lev * N) / (1.0 - N))
    return (lev - N) / N


def stretching(s, Vstretching, theta_s, theta_b):
    """Stretching function C(s) for ROMS Vstretching = 1..5."""
    s = np.asarray(s, dtype=float)
    if Vstretching == 1:
        if theta_s > 0:
            c1 = 1.0 / np.sinh(theta_s)
            c2 = 0.5 / np.tanh(0.5 * theta_s)
            return ((1 - theta_b) * c1 * np.sinh(theta_s * s)
                    + theta_b * (c2 * np.tanh(theta_s * (s + 0.5)) - 0.5))
        return s.copy()
    if Vstretching == 2:
        if theta_s > 0:
            csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
            if theta_b > 0:
                cbot = -1 + np.sinh(theta_b * (s + 1)) / np.sinh(theta_b)
                w = (s + 1) * (1 - s)
                return w * csur + (1 - w) * cbot
            return csur
        return s.copy()
    if Vstretching == 3:
        hs = 3.0
        cbot = np.log(np.cosh(hs * (s + 1) ** theta_b)) / np.log(np.cosh(hs)) - 1
        ctop = 1 - np.log(np.cosh(hs * np.abs(s) ** theta_s)) / np.log(np.cosh(hs))
        bm = 0.5 * (1 - np.tanh(hs * (s + 0.5)))
        return bm * cbot + (1 - bm) * ctop
    if Vstretching in (4, 5):
        csur = ((1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
                if theta_s > 0 else -s ** 2)
        if theta_b > 0:
            return (np.exp(theta_b * csur) - 1) / (1 - np.exp(-theta_b))
        return csur
    raise ValueError(f"Unsupported Vstretching = {Vstretching}")


def compute_z(h, zeta, s, Cs, hc, Vtransform):
    """Depth (m, negative) of s-levels. h, zeta: (ny, nx) -> (N, ny, nx)."""
    s = np.asarray(s, float).reshape(-1, 1, 1)
    Cs = np.asarray(Cs, float).reshape(-1, 1, 1)
    h = np.maximum(h, 1e-3)
    if Vtransform == 1:
        z0 = hc * s + (h - hc) * Cs
        return z0 + zeta * (1.0 + z0 / h)
    if Vtransform == 2:
        z0 = (hc * s + h * Cs) / (hc + h)
        return zeta + (zeta + h) * z0
    raise ValueError(f"Unsupported Vtransform = {Vtransform}")


def vinterp(v, z, depths, h, bottom_fill=True):
    """
    Linear interpolation of v (N, ny, nx) from level depths z (N, ny, nx;
    negative, increasing with index) to fixed depths (positive down).
    Above the top level -> top value; between the deepest level and the
    seabed -> deepest value (if bottom_fill); below the seabed -> NaN.
    """
    N = v.shape[0]
    shp = v.shape[1:]
    v2 = v.reshape(N, -1)
    z2 = z.reshape(N, -1)
    h2 = h.ravel()
    cols = np.arange(v2.shape[1])
    out = np.full((len(depths), v2.shape[1]), np.nan)
    # k = number of levels below the target; targets visited deepest first so
    # k only moves upward (single pass through the column)
    k = np.zeros(v2.shape[1], dtype=np.intp)
    for i in np.argsort(depths)[::-1]:
        zt = -float(depths[i])
        while True:
            adv = (k < N) & (z2[np.minimum(k, N - 1), cols] < zt)
            if not adv.any():
                break
            k += adv
        ki = np.clip(k, 1, N - 1)
        zlo, zhi = z2[ki - 1, cols], z2[ki, cols]
        vlo, vhi = v2[ki - 1, cols], v2[ki, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            val = vlo + (zt - zlo) / (zhi - zlo) * (vhi - vlo)
        val = np.where(k == N, v2[N - 1], val)
        val = np.where(k == 0, v2[0] if bottom_fill else np.nan, val)
        val = np.where(zt < -h2 - 1e-6, np.nan, val)
        out[i] = val
    return out.reshape((len(depths),) + shp)


# ==========================================================================
# Staggered-grid helpers
# ==========================================================================
def rho2u(a):
    return 0.5 * (a[..., :, :-1] + a[..., :, 1:])


def rho2v(a):
    return 0.5 * (a[..., :-1, :] + a[..., 1:, :])


def rho2psi(a):
    return 0.25 * (a[..., :-1, :-1] + a[..., :-1, 1:] + a[..., 1:, :-1] + a[..., 1:, 1:])


STAGGER = {"rho": lambda a: a, "u": rho2u, "v": rho2v, "psi": rho2psi}


def _nanmean2(a, b):
    va, vb = np.isfinite(a), np.isfinite(b)
    n = va.astype(float) + vb
    s = np.where(va, a, 0.0) + np.where(vb, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n
    out[n == 0] = np.nan
    return out


def u2rho(u):
    out = np.full(u.shape[:-1] + (u.shape[-1] + 1,), np.nan)
    out[..., 1:-1] = _nanmean2(u[..., :-1], u[..., 1:])
    out[..., 0], out[..., -1] = u[..., 0], u[..., -1]
    return out


def v2rho(v):
    out = np.full(v.shape[:-2] + (v.shape[-2] + 1, v.shape[-1]), np.nan)
    out[..., 1:-1, :] = _nanmean2(v[..., :-1, :], v[..., 1:, :])
    out[..., 0, :], out[..., -1, :] = v[..., 0, :], v[..., -1, :]
    return out


# ==========================================================================
# Horizontal interpolation: curvilinear -> regular lon/lat
# ==========================================================================
class HorizInterp:
    """
    Barycentric (linear) interpolation from a structured curvilinear grid to
    a regular lon/lat grid.  Weights are computed once and reused for every
    level and time step.  Only triangles lying inside a single grid cell are
    used, so nothing is interpolated across concave domain edges.  NaN
    (land / below seabed) source values are skipped and the remaining weights
    re-normalised; a target point is kept if its valid weight >= min_weight.
    """

    def __init__(self, lon_src, lat_src, lon_t, lat_t, min_weight=0.5, geographic=True):
        # lon/lat in degrees (geographic=True) or x/y in metres (geographic=False)
        self.min_weight = min_weight
        ny, nx = lon_src.shape
        lat0 = float(np.nanmean(lat_src))
        lon0 = float(np.nanmean(lon_src))
        c = np.cos(np.deg2rad(lat0)) if geographic else 1.0
        pts = np.column_stack([((lon_src - lon0) * c).ravel(), (lat_src - lat0).ravel()])
        tri = Delaunay(pts)
        jj, ii = np.divmod(tri.simplices, nx)
        good = ((jj.max(1) - jj.min(1)) <= 1) & ((ii.max(1) - ii.min(1)) <= 1)
        LON, LAT = np.meshgrid(lon_t, lat_t)
        tp = np.column_stack([((LON - lon0) * c).ravel(), (LAT - lat0).ravel()])
        simp = tri.find_simplex(tp)
        inside = simp >= 0
        inside[inside] = good[simp[inside]]
        sidx = simp[inside]
        T = tri.transform[sidx]
        b = np.einsum("ijk,ik->ij", T[:, :2], tp[inside] - T[:, 2])
        w = np.clip(np.column_stack([b, 1.0 - b.sum(1)]), 0.0, None)
        w /= w.sum(1, keepdims=True)
        self.idx = tri.simplices[sidx]
        self.w = w
        self.inside = np.nonzero(inside)[0]
        self.tshape = LON.shape
        self.ntgt = LON.size
        # sparse weight matrix (n_target x n_source): one fast product per call
        rows = np.repeat(self.inside, 3)
        self.W = csr_matrix((w.ravel(), (rows, self.idx.ravel())),
                            shape=(self.ntgt, pts.shape[0]))

    def __call__(self, field):
        lead = field.shape[:-2]
        f = field.reshape((-1, field.shape[-2] * field.shape[-1])).T   # (nsrc, nlev)
        valid = np.isfinite(f)
        num = self.W @ np.where(valid, f, 0.0)
        den = self.W @ valid.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(den >= self.min_weight - 1e-9, num / den, np.nan)
        return val.T.reshape(lead + self.tshape)


# ==========================================================================
# Small utilities
# ==========================================================================
def _to_float(a):
    return np.ma.filled(np.ma.asarray(a).astype(float), np.nan)


def _hms(sec):
    if sec is None or not np.isfinite(sec):
        return "--:--:--"
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


class _Src:
    """Look variables up in the model output first, then in the grid file."""

    def __init__(self, nc, grd):
        self.nc, self.grd = nc, grd

    def has(self, name):
        return name in self.nc.variables or (self.grd is not None and name in self.grd.variables)

    def get(self, name, default=None):
        for ds in (self.nc, self.grd):
            if ds is not None and name in ds.variables:
                v = ds.variables[name]
                v.set_auto_mask(True)
                return _to_float(v[:])
        if default is None:
            raise KeyError(f"'{name}' not found in the input or grid file")
        return default

    def scalar(self, name, default=None):
        for ds in (self.nc, self.grd):
            if ds is None:
                continue
            if name in ds.variables:
                return float(np.ma.filled(ds.variables[name][:], np.nan).ravel()[0])
            if name in ds.ncattrs():
                return float(np.ravel(ds.getncattr(name))[0])
        return default


KM_PER_DEG = 111.195          # km per degree of latitude (mean Earth radius 6371 km)


def is_spherical(src):
    """
    True  -> geographic grid (lon_rho / lat_rho, degrees)
    False -> Cartesian grid (x_rho / y_rho, metres)
    Uses the ROMS 'spherical' flag when present, otherwise what is available.
    """
    flag = None
    for ds in (src.nc, src.grd):
        if ds is not None and "spherical" in ds.variables:
            v = ds.variables["spherical"]
            v.set_auto_mask(False)
            raw = np.ravel(v[...])
            txt = b"".join(r if isinstance(r, bytes) else str(r).encode() for r in raw)
            txt = txt.decode(errors="ignore").strip().upper()
            flag = txt.startswith(("T", "1"))
            break
    has_ll = src.has("lon_rho") and src.has("lat_rho")
    has_xy = src.has("x_rho") and src.has("y_rho")
    if flag is None:
        if has_ll and np.nanmax(src.get("lon_rho")) != np.nanmin(src.get("lon_rho")):
            flag = True
        elif has_xy:
            flag = False
        else:
            flag = has_ll
    if flag and not has_ll:
        if has_xy:
            return False
        raise KeyError("Grid is spherical but lon_rho/lat_rho were not found (set GRID_FILE?)")
    if not flag and not has_xy:
        raise KeyError("Grid is Cartesian but x_rho/y_rho were not found (set GRID_FILE?)")
    return flag


def native_spacing_km(src, X, Y, geographic):
    """Median model grid spacing in km (from pm/pn if present, else from coordinates)."""
    if src.has("pm") and src.has("pn"):
        pm, pn = src.get("pm"), src.get("pn")
        d = np.concatenate([1.0 / pm[np.isfinite(pm) & (pm > 0)],
                            1.0 / pn[np.isfinite(pn) & (pn > 0)]]) / 1000.0
        if d.size:
            return float(np.median(d))
    if geographic:
        c = np.cos(np.deg2rad(Y))
        dx = KM_PER_DEG * np.hypot(np.diff(X, axis=1) * c[:, 1:], np.diff(Y, axis=1))
        dy = KM_PER_DEG * np.hypot(np.diff(X, axis=0) * c[1:, :], np.diff(Y, axis=0))
    else:
        dx = np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)) / 1000.0
        dy = np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)) / 1000.0
    return float(np.nanmedian(np.concatenate([dx.ravel(), dy.ravel()])))


def _nice(x):
    return float(f"{x:.2g}")


def parse_resolution(val, geographic):
    """
    Returns (unit, value) with unit 'auto', 'deg' or 'km'.
    Accepts "auto", a number, or a string with units: "5km", "2500m", "0.05deg".
    A plain number means degrees on a spherical grid and km on a Cartesian grid.
    """
    if val is None or (isinstance(val, str) and val.strip().lower() == "auto"):
        return "auto", None
    if isinstance(val, (int, float, np.integer, np.floating)):
        return ("deg" if geographic else "km"), float(val)
    s = str(val).strip().lower().replace(" ", "")
    for suffix, unit, factor in (("km", "km", 1.0), ("degrees", "deg", 1.0),
                                 ("degree", "deg", 1.0), ("deg", "deg", 1.0),
                                 ("°", "deg", 1.0), ("m", "km", 1e-3)):
        if s.endswith(suffix):
            return unit, float(s[: -len(suffix)]) * factor
    return ("deg" if geographic else "km"), float(s)


def resolve_resolution(resolution, src, X, Y, geographic):
    """
    Target grid steps (dx, dy): degrees on a spherical grid, metres on a
    Cartesian grid.  'resolution' may be one value or a pair (x, y).
    Returns dx, dy, native spacing (km) and a text description.
    """
    pair = resolution if isinstance(resolution, (tuple, list)) else (resolution, resolution)
    ds_km = native_spacing_km(src, X, Y, geographic)
    lat_c = float(np.nanmean(Y)) if geographic else 0.0
    steps, words = [], []
    for axis, val in zip(("x", "y"), pair):
        unit, v = parse_resolution(val, geographic)
        if unit == "auto":
            unit, v = ("deg", _nice(ds_km / KM_PER_DEG)) if geographic else ("km", _nice(ds_km))
            words.append("auto")
        else:
            words.append(f"{v:g} {unit}")
        if v <= 0:
            raise ValueError(f"Resolution must be > 0 (got {val})")
        if geographic:
            if unit == "km":        # km -> degrees, exact at the domain's centre latitude
                v = v / KM_PER_DEG / (np.cos(np.deg2rad(lat_c)) if axis == "x" else 1.0)
                v = float(f"{v:.4g}")
        else:
            if unit == "deg":
                raise ValueError("Resolution in degrees is not possible on a Cartesian "
                                 "(x_rho/y_rho) grid; give it in km or m")
            v = v * 1000.0          # km -> metres
        steps.append(v)
    how = words[0] if words[0] == words[1] else f"{words[0]} / {words[1]}"
    return steps[0], steps[1], ds_km, how


def _find_time_dim(nc):
    for name, d in nc.dimensions.items():
        if d.isunlimited():
            return name
    for cand in ("ocean_time", "time"):
        if cand in nc.dimensions:
            return cand
    return None


def _classify(var, tdim):
    """
    Return (grid, vert, has_time) for a variable on the horizontal ROMS grid,
    'unsupported' for a grid variable with extra dims, or None if the
    variable is not on the horizontal grid (-> copied unchanged).
    """
    d = var.dimensions
    if len(d) < 2 or d[-1] not in GRID_OF_XI:
        return None
    if GRID_OF_ETA.get(d[-2]) != GRID_OF_XI[d[-1]]:
        return "unsupported"
    rest, vert = list(d[:-2]), None
    if rest and rest[-1] in VERT_DIMS:
        vert = rest.pop()
    if len(rest) > 1 or (rest and rest[0] != tdim):
        return "unsupported"
    return GRID_OF_XI[d[-1]], vert, bool(rest)


# ==========================================================================
# Worker (runs in each parallel process)
# ==========================================================================
_W = {}


def _reset_worker():
    """Forget worker state from an earlier convert() call in this process."""
    if "nc" in _W:
        try:
            _W["nc"].close()
        except Exception:
            pass
    _W.clear()


def _init_worker(infile, G):
    nc = Dataset(infile)
    nc.set_auto_mask(True)
    _W["nc"] = nc
    _W["G"] = G


def _zeta(t):
    G, nc = _W["G"], _W["nc"]
    if t is not None and G["zeta_ok"]:
        return np.nan_to_num(_to_float(nc.variables["zeta"][t]))
    return np.zeros_like(G["h"]["rho"])


def _zlev(g, vert, zeta_r):
    G = _W["G"]
    s, Cs = G["scoord"][vert]
    return compute_z(G["h"][g], STAGGER[g](zeta_r), s, Cs, G["hc"], G["Vtransform"])


def _read(name, t):
    var = _W["nc"].variables[name]
    return _to_float(var[t] if t is not None else var[:])


def _run(task):
    """task = (kind, names, grid, vert, t); returns (t, [float32 arrays])."""
    kind, names, g, vert, t = task
    G = _W["G"]
    depths, bfill = G["depths"], G["bottom_fill"]
    zeta_r = _zeta(t) if vert else None
    outs = []

    if kind == "vector":
        a, b = names
        ga, gb = G["info"][a][0], G["info"][b][0]
        ua = np.where(G["mask"][ga], _read(a, t), np.nan)
        vb = np.where(G["mask"][gb], _read(b, t), np.nan)
        ua = u2rho(ua) if ga == "u" else ua
        vb = v2rho(vb) if gb == "v" else vb
        ca, sa = np.cos(G["angle"]), np.sin(G["angle"])
        comps = [ua * ca - vb * sa, ua * sa + vb * ca]
        z = _zlev("rho", vert, zeta_r) if vert else None
        for c in comps:
            c = np.where(G["mask"]["rho"], c, np.nan)
            if vert:
                c = vinterp(c, z, depths, G["h"]["rho"], bfill)
            outs.append(G["interp"]["rho"](c))
    else:
        data = _read(names[0], t)
        if kind == "field":                       # time-varying: blank land
            data = np.where(G["mask"][g], data, np.nan)
        if vert:
            data = vinterp(data, _zlev(g, vert, zeta_r), depths, G["h"][g], bfill)
        res = G["interp"][g](data)
        if kind == "maskvar":
            res = np.where(np.isfinite(res), (res >= 0.5).astype(float), np.nan)
        outs.append(res)

    return t, [np.where(np.isfinite(o), o, FILL).astype(np.float32) for o in outs]


# ==========================================================================
# Progress display
# ==========================================================================
class _Progress:
    def __init__(self, total_units):
        self.total = max(total_units, 1e-9)
        self.done = 0.0
        self.t_start = None
        self.tty = sys.stdout.isatty()

    def start(self):
        if self.t_start is None:
            self.t_start = _time.time()

    def step(self, units):
        self.done += units

    def eta_total(self):
        if not self.t_start or self.done <= 0:
            return None
        rate = self.done / (_time.time() - self.t_start)
        return (self.total - self.done) / rate

    def line(self, text, final=False):
        if self.tty:
            end = "\n" if final else ""
            sys.stdout.write("\r" + text.ljust(118)[:160] + end)
        else:
            sys.stdout.write(text + "\n")
        sys.stdout.flush()


# ==========================================================================
# Main entry point
# ==========================================================================
def convert(input_file, output_file, resolution="auto", variables="all", depths="WOA",
            grid_file=None, ncores=None, time_indices=None, rotate_vectors=True,
            bottom_fill=True, min_weight=0.5, compress_level=1, dlon=None, dlat=None):
    """
    Convert a ROMS history/average file to a regular lon/lat/depth grid
    (or a regular x/y/depth grid for Cartesian ROMS grids).

    input_file     : ROMS output NetCDF file
    output_file    : NetCDF file to write
    resolution     : horizontal step of the regular grid
                     "auto"     -> same as the model's median grid spacing
                     "5km", "2500m", "0.05deg" -> explicit units
                     a number   -> degrees (spherical grid) or km (Cartesian grid)
                     (x, y)     -> different steps in x/lon and y/lat
    variables      : "all" or a list of variable names
    depths         : "WOA" (World Ocean Atlas levels down to the deepest
                     bathymetry) or a list of depths in m (positive down)
    grid_file      : ROMS grid file, if coordinates/mask/h/angle are not in input
    ncores         : number of parallel processes (None = all CPU cores)
    time_indices   : None (all) or a list/range of 0-based time indices
    rotate_vectors : rotate u/v-type pairs to eastward/northward (x/y)
    bottom_fill    : fill between the deepest model level and the seabed
    min_weight     : 0-1, wet-weight needed at coastal target points
    compress_level : NetCDF zlib level 0-9 (0 = none, fastest)
    dlon, dlat     : optional, degrees; override 'resolution' (older interface)
    """
    T0 = _time.time()
    _reset_worker()
    ncores = int(ncores or os.cpu_count() or 1)

    nc = Dataset(input_file)
    grd = Dataset(grid_file) if grid_file else None
    src = _Src(nc, grd)
    tdim = _find_time_dim(nc)

    print("=" * 78)
    print(f" ROMS s-coordinate -> regular horizontal grid + depth levels   (roms2z_lib v{__version__})")
    print("=" * 78)
    print(f" Input  : {input_file}")
    if grid_file:
        print(f" Grid   : {grid_file}")
    print(f" Output : {output_file}")

    # ------------------------------------------------------------------ plan
    info, plan, unsupported = {}, [], []
    for name, var in nc.variables.items():
        c = _classify(var, tdim)
        if c == "unsupported":
            unsupported.append(name)
            continue
        if c is not None:
            info[name] = c

    if isinstance(variables, str) and variables.lower() == "all":
        selected = list(nc.variables)
    else:
        selected = list(variables)
        missing = [v for v in selected if v not in nc.variables]
        if missing:
            raise KeyError(f"Variables not found in {input_file}: {missing}")
        if tdim and tdim in nc.variables and tdim not in selected:
            selected.insert(0, tdim)             # always keep the time axis
        if rotate_vectors:
            for a, b in VECTOR_PAIRS:
                for x, y in ((a, b), (b, a)):
                    if x in selected and y not in selected and y in nc.variables:
                        selected.append(y)
                        print(f" Note   : added '{y}' so '{x}' can be rotated to east/north")

    used = set()
    for name in selected:
        if name in used:
            continue
        var = nc.variables[name]
        ndim = var.ndim
        if name in unsupported:
            print(f" Skip   : {name} {var.dimensions} (grid variable with extra dimensions)")
            continue
        if name not in info:
            plan.append(dict(ndim=ndim, kind="copy", names=(name,)))
            continue
        if name.startswith(COORD_PREFIXES):
            continue                             # replaced by lon/lat axes
        g, vert, has_t = info[name]
        pair = None
        if rotate_vectors and has_t:
            for a, b in VECTOR_PAIRS:
                if name in (a, b) and a in selected and b in selected \
                        and a in info and b in info and info[a][1:] == info[b][1:]:
                    pair = (a, b)
        if pair:
            plan.append(dict(ndim=ndim, kind="vector", names=pair, grid="rho", vert=vert, time=True))
            used.update(pair)
        else:
            kind = "field" if has_t else ("maskvar" if name.startswith("mask") else "static")
            plan.append(dict(ndim=ndim, kind=kind, names=(name,), grid=g, vert=vert, time=has_t))
            used.add(name)
    plan.sort(key=lambda p: p["ndim"])           # 0-D, 1-D, 2-D, 3-D, 4-D

    # ------------------------------------------------------------------ grid
    spherical = is_spherical(src)
    PX, PY = ("lon", "lat") if spherical else ("x", "y")     # coordinate prefixes
    XN, YN = PX, PY                                          # output axis names
    lon_r, lat_r = src.get(f"{PX}_rho"), src.get(f"{PY}_rho")
    wrap = spherical and (np.nanmax(lon_r) - np.nanmin(lon_r) > 180)
    if wrap:
        lon_r = np.mod(lon_r, 360.0)
    h_r = src.get("h")
    mask_r = np.nan_to_num(src.get("mask_rho", np.ones_like(h_r)))
    angle = src.get("angle", np.zeros_like(h_r)) if src.has("angle") else np.zeros_like(h_r)
    if rotate_vectors and not src.has("angle") and any(p["kind"] == "vector" for p in plan):
        warnings.warn("'angle' not found: assuming the grid is aligned with the x/y axes")

    def grid_mask(g):
        if src.has(f"mask_{g}"):
            return np.nan_to_num(src.get(f"mask_{g}")) > 0.5
        m = {"rho": lambda a: a, "u": lambda a: a[:, :-1] * a[:, 1:],
             "v": lambda a: a[:-1, :] * a[1:, :],
             "psi": lambda a: a[:-1, :-1] * a[:-1, 1:] * a[1:, :-1] * a[1:, 1:]}[g](mask_r)
        return m > 0.5

    def grid_lonlat(g):
        if src.has(f"{PX}_{g}") and src.has(f"{PY}_{g}"):
            lo, la = src.get(f"{PX}_{g}"), src.get(f"{PY}_{g}")
        else:
            lo, la = STAGGER[g](src.get(f"{PX}_rho")), STAGGER[g](lat_r)
        return (np.mod(lo, 360.0) if wrap else lo), la

    # horizontal resolution: given, or taken from the model grid spacing
    if dlon is not None or dlat is not None:
        resolution = (f"{dlon}deg" if dlon is not None else "auto",
                      f"{dlat}deg" if dlat is not None else "auto")
    dlon, dlat, ds_km, how = resolve_resolution(resolution, src, lon_r, lat_r, spherical)

    # regular target grid covering the whole model domain
    lon_t = np.arange(np.floor(np.nanmin(lon_r) / dlon) * dlon,
                      np.ceil(np.nanmax(lon_r) / dlon) * dlon + dlon / 2, dlon)
    lat_t = np.arange(np.floor(np.nanmin(lat_r) / dlat) * dlat,
                      np.ceil(np.nanmax(lat_r) / dlat) * dlat + dlat / 2, dlat)
    lon_t, lat_t = np.round(lon_t, 10), np.round(lat_t, 10)
    ratio = lon_t.size * lat_t.size / lon_r.size
    if ratio > 50:
        unit = "deg" if spherical else "m"
        raise ValueError(
            f"Requested resolution (d{XN}={dlon:g} {unit}, d{YN}={dlat:g} {unit}) gives "
            f"{lon_t.size} x {lat_t.size} points, {ratio:.0f} times more than the model grid "
            f"(spacing ~{ds_km:.3g} km). Check the units of RESOLUTION "
            f"(a plain number means {'degrees' if spherical else 'km'} for this grid), "
            f"or use \"auto\".")

    # vertical coordinate
    Vtransform = int(src.scalar("Vtransform", 1))
    Vstretching = int(src.scalar("Vstretching", 1))
    theta_s = src.scalar("theta_s", 0.0)
    theta_b = src.scalar("theta_b", 0.0)
    hc = src.scalar("hc", None)
    hc = src.scalar("Tcline", 0.0) if hc is None else hc
    scoord = {}
    if "s_rho" in nc.dimensions:
        N = len(nc.dimensions["s_rho"])
        s_r = src.get("s_rho") if src.has("s_rho") else s_levels(N, Vstretching, 0)
        s_w = src.get("s_w") if src.has("s_w") else s_levels(N, Vstretching, 1)
        Cs_r = src.get("Cs_r") if src.has("Cs_r") else stretching(s_r, Vstretching, theta_s, theta_b)
        Cs_w = src.get("Cs_w") if src.has("Cs_w") else stretching(s_w, Vstretching, theta_s, theta_b)
        scoord = {"s_rho": (s_r, Cs_r), "s_w": (s_w, Cs_w)}
    elif any(p.get("vert") for p in plan):
        raise ValueError("3-D (s-level) variables found but no s_rho dimension")

    if isinstance(depths, str):
        hmax = np.nanmax(np.where(mask_r > 0.5, h_r, np.nan))
        n = int(np.searchsorted(WOA_DEPTHS, hmax)) + 1
        depths = WOA_DEPTHS[:min(n, len(WOA_DEPTHS))]
    depths = np.asarray(sorted(set(float(d) for d in depths)))

    nt_all = len(nc.dimensions[tdim]) if tdim else 0
    tlist = list(range(nt_all)) if time_indices is None else list(time_indices)
    nt = len(tlist)

    print(f" Native grid : {lon_r.shape[1]} x {lon_r.shape[0]} (xi x eta), "
          f"{'spherical (lon_rho/lat_rho)' if spherical else 'Cartesian (x_rho/y_rho)'}, "
          f"spacing ~{ds_km:.3g} km")
    if scoord:
        print(f" Vertical    : {len(nc.dimensions['s_rho'])} s-levels  Vtransform={Vtransform} "
              f"Vstretching={Vstretching} theta_s={theta_s:g} theta_b={theta_b:g} hc={hc:g}")
    if spherical:
        kmx = dlon * KM_PER_DEG * np.cos(np.deg2rad(np.nanmean(lat_r)))
        print(f" Resolution  : {how} -> dlon={dlon:g} deg (~{kmx:.3g} km), "
              f"dlat={dlat:g} deg (~{dlat * KM_PER_DEG:.3g} km)")
        print(f" Target grid : {len(lon_t)} lon x {len(lat_t)} lat  "
              f"lon {lon_t[0]:g}..{lon_t[-1]:g}  lat {lat_t[0]:g}..{lat_t[-1]:g}")
    else:
        print(f" Resolution  : {how} -> dx={dlon / 1000:g} km, dy={dlat / 1000:g} km")
        print(f" Target grid : {len(lon_t)} x by {len(lat_t)} y  "
              f"x {lon_t[0] / 1000:g}..{lon_t[-1] / 1000:g} km  y {lat_t[0] / 1000:g}..{lat_t[-1] / 1000:g} km")
    print(f" Depths      : {len(depths)} levels, {depths[0]:g}-{depths[-1]:g} m")
    print(f" Time steps  : {nt}" + (f" of {nt_all}" if nt != nt_all else "")
          + f"   Parallel processes: {ncores}")
    counts = {k: sum(1 for p in plan if p["ndim"] == k) for k in range(5)}
    print(" Variables   : " + "  ".join(f"{k}-D:{counts[k]}" for k in range(5)))

    # interpolation weights for the grids actually needed
    need = {p["grid"] for p in plan if p["kind"] != "copy"}
    interp = {}
    for g in sorted(need):
        t1 = _time.time()
        lo, la = grid_lonlat(g)
        interp[g] = HorizInterp(lo, la, lon_t, lat_t, min_weight, geographic=spherical)
        print(f" Weights     : {g}-points {lo.shape} ready ({_time.time() - t1:.1f} s)")

    G = dict(info=info, interp=interp, angle=angle, scoord=scoord, hc=hc,
             Vtransform=Vtransform, depths=depths, bottom_fill=bottom_fill,
             mask={g: grid_mask(g) for g in ("rho", "u", "v", "psi")},
             h={g: STAGGER[g](h_r) for g in ("rho", "u", "v", "psi")},
             zeta_ok=("zeta" in info and info["zeta"][2]))
    if grd is not None:
        grd.close()

    # ---------------------------------------------------------------- output
    out = Dataset(output_file, "w", format="NETCDF4")
    for a in nc.ncattrs():
        try:
            out.setncattr(a, nc.getncattr(a))
        except Exception:
            pass
    out.setncattr("history", f"{_time.ctime()}: regridded to regular "
                  f"{'lon/lat' if spherical else 'x/y'}/depth by "
                  f"roms2z_lib v{__version__} from {os.path.basename(input_file)}\n"
                  + str(getattr(nc, "history", "")))
    out.setncattr("roms2z_note", ("Velocity pairs rotated to "
                  + ("eastward/northward" if spherical else "x/y directions") + " at rho points")
                  if rotate_vectors else "Velocities left grid-relative")
    out.setncattr("roms2z_resolution", f"d{XN}={dlon:g} {'deg' if spherical else 'm'}, "
                  f"d{YN}={dlat:g} {'deg' if spherical else 'm'} ({how}); "
                  f"model spacing ~{ds_km:.3g} km")
    if tdim:
        out.createDimension(tdim, None)
    out.createDimension("depth", len(depths))
    out.createDimension(YN, len(lat_t))
    out.createDimension(XN, len(lon_t))
    v = out.createVariable("depth", "f8", ("depth",))
    v[:] = depths
    v.setncatts(dict(long_name="depth below sea surface", units="m", positive="down",
                     standard_name="depth", axis="Z"))
    v = out.createVariable(YN, "f8", (YN,))
    v[:] = lat_t
    v.setncatts(dict(long_name="latitude", units="degrees_north", standard_name="latitude",
                     axis="Y") if spherical else
                dict(long_name="y-distance (ROMS y_rho)", units="m",
                     standard_name="projection_y_coordinate", axis="Y"))
    v = out.createVariable(XN, "f8", (XN,))
    v[:] = lon_t
    v.setncatts(dict(long_name="longitude", units="degrees_east", standard_name="longitude",
                     axis="X") if spherical else
                dict(long_name="x-distance (ROMS x_rho)", units="m",
                     standard_name="projection_x_coordinate", axis="X"))

    def ensure_dim(d):
        if d not in out.dimensions:
            out.createDimension(d, None if d == tdim else len(nc.dimensions[d]))

    def new_var(name, src_var, time, vert, extra):
        dims = ((tdim,) if time else ()) + (("depth",) if vert else ()) + (YN, XN)
        chunks = ((1,) if time else ()) + ((1,) if vert else ()) + (len(lat_t), len(lon_t))
        ov = out.createVariable(name, "f4", dims, zlib=compress_level > 0,
                                complevel=max(compress_level, 1), fill_value=FILL,
                                chunksizes=chunks)
        for a in src_var.ncattrs():
            if a not in DROP_ATTRS:
                ov.setncattr(a, src_var.getncattr(a))
        for k, val in extra.items():
            ov.setncattr(k, val)
        return ov

    # time labels for the terminal
    tlabels = [str(t) for t in tlist]
    if tdim and tdim in nc.variables:
        tv = nc.variables[tdim]
        try:
            dates = num2date(tv[tlist], tv.units, getattr(tv, "calendar", "standard"))
            tlabels = [d.strftime("%Y-%m-%d %H:%M") for d in np.atleast_1d(dates)]
        except Exception:
            pass

    # progress bookkeeping (work units ~ number of levels processed)
    def units(p):
        n_lev = len(depths) if p.get("vert") else 1
        return n_lev * len(p["names"])

    prog = _Progress(sum(units(p) * nt for p in plan if p.get("time")))
    pool = None

    # ------------------------------------------------------------ processing
    try:
        current_ndim = -1
        for ip, p in enumerate(plan, 1):
            if p["ndim"] != current_ndim:
                current_ndim = p["ndim"]
                label = {0: "constants"}.get(current_ndim, "variables")
                print("-" * 78)
                print(f" {current_ndim}-D {label}")
                print("-" * 78)
            tag = f"[{ip:>3d}/{len(plan)}]"

            # ---- copied unchanged (not on the horizontal grid)
            if p["kind"] == "copy":
                name = p["names"][0]
                iv = nc.variables[name]
                for d in iv.dimensions:
                    ensure_dim(d)
                try:
                    fv = iv.getncattr("_FillValue") if "_FillValue" in iv.ncattrs() else None
                    ov = out.createVariable(name, iv.datatype, iv.dimensions, fill_value=fv)
                    ov.setncatts({a: iv.getncattr(a) for a in iv.ncattrs() if a != "_FillValue"})
                    iv.set_auto_maskandscale(False)
                    ov.set_auto_maskandscale(False)
                    if iv.ndim == 0:
                        ov[...] = iv[...]
                    elif tdim in iv.dimensions:
                        ax = iv.dimensions.index(tdim)
                        ov[...] = np.take(iv[...], tlist, axis=ax)
                    else:
                        ov[...] = iv[...]
                    print(f" {tag} {name:<14s} copied        {iv.dimensions}")
                except Exception as e:
                    print(f" {tag} {name:<14s} NOT copied ({e})")
                continue

            names, g, vert = p["names"], p["grid"], p["vert"]
            desc = f"{g}-pts" + (f", {vert}" if vert else "")
            if p["kind"] == "vector":
                desc = f"{info[names[0]][0]}/{info[names[1]][0]}-pts -> rho, " \
                       + ("east/north" if spherical else "x/y") + (f", {vert}" if vert else "")
                dirs = ("eastward", "northward") if spherical else ("x-direction", "y-direction")
                extra = [dict(direction=dirs[0], comment=f"rotated from grid-relative {names[0]},{names[1]}"),
                         dict(direction=dirs[1], comment=f"rotated from grid-relative {names[0]},{names[1]}")]
            else:
                extra = [dict(comment=f"interpolated from ROMS {g}-points"
                              + (" to depth levels" if vert else ""))]
            ovs = [new_var(n, nc.variables[n], p["time"], vert, e) for n, e in zip(names, extra)]
            vname = "+".join(names)

            # ---- static field on the grid (h, f, pm, mask_rho, ...): main process
            if not p["time"]:
                if "nc" not in _W:
                    _init_worker(input_file, G)
                _, res = _run((p["kind"], names, g, vert, None))
                for ov, r in zip(ovs, res):
                    ov[...] = r
                print(f" {tag} {vname:<14s} regridded     ({desc})")
                continue

            # ---- time-varying field: parallel over time steps
            if pool is None and ncores > 1:
                methods = mp.get_all_start_methods()
                ctx = mp.get_context("fork" if "fork" in methods else "spawn")
                pool = ctx.Pool(ncores, initializer=_init_worker, initargs=(input_file, G))
            if pool is None and "nc" not in _W:
                _init_worker(input_file, G)
            prog.start()
            tasks = [(p["kind"], names, g, vert, t) for t in tlist]
            it_map = pool.imap(_run, tasks, chunksize=1) if pool else map(_run, tasks)
            tv0 = _time.time()
            for k, (t, res) in enumerate(it_map):
                for ov, r in zip(ovs, res):
                    ov[k] = r
                prog.step(units(p))
                el = _time.time() - tv0
                eta_v = el / (k + 1) * (nt - k - 1)
                final = k + 1 == nt
                prog.line(f" {tag} {vname:<14s} {desc:<24s} time {k + 1:>5d}/{nt} "
                          f"[{tlabels[k]}] {100 * (k + 1) / nt:5.1f}%  "
                          f"var {_hms(el) if final else 'ETA ' + _hms(eta_v)}  "
                          f"| total ETA {_hms(prog.eta_total())}", final=final)
            out.sync()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        out.close()
        nc.close()
        _reset_worker()

    print("=" * 78)
    print(f" Done: {output_file}   total time {_hms(_time.time() - T0)}")
    print("=" * 78)
    return output_file
