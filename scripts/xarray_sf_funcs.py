import dask.delayed
import numpy as np
import xarray as xr
from scipy.special import jv
from scipy.ndimage import gaussian_filter, uniform_filter
import h5py
import dask
from astropy.convolution import convolve_fft, Tophat2DKernel
from copy import copy
import dask.bag as db
import pandas as pd
import glob
import time

# _SWOT_L4_CACHE = None


# def _get_swot_l4_source_ds():
#     global _SWOT_L4_CACHE
#     if _SWOT_L4_CACHE is None:
#         _SWOT_L4_CACHE = xr.open_mfdataset(
#             '/Volumes/Promise Disk/data/validated_swot/SWOT_L4_duacs_v2.0.1/science/*.nc',
#             combine="nested",
#             concat_dim="time",
#             parallel=True,
#             engine="h5netcdf",
#             drop_variables=["longitude_bounds", "latitude_bounds", "ssh", "ugosa", "vgosa", "adt"],
#         )
#     return _SWOT_L4_CACHE


# @dask.delayed
def shift_fields(ds, var, shifts, shift_func):
    shifted_fields = {}
    for direction, shift in shifts.items():
        shifted_key = f"{var}_shifted_{direction}"

        if shift_func == "roll":
            shifted_fields[shifted_key] = ds[var].roll(**shift, roll_coords=False)
        else:
            shifted_fields[shifted_key] = ds[var].shift(**shift)

    return shifted_fields

@dask.delayed
def compute_asf(
    ds,
    shiftnum=1,
    shiftx=None,
    shifty=None,
    full=True,
    uvar="u",
    vvar="v",
    qvar="q",
    xvar="x",
    yvar="y",
    shift_func="roll",
    just_mean=False,
):
    ds_tmp = ds.copy()

    x_len = len(ds_tmp[xvar])
    y_len = len(ds_tmp[yvar])

    if shiftx is None:
        shiftx = shiftnum
    if shifty is None:
        shifty = shiftnum

    shiftdiag = int(np.sqrt(shiftx**2 + shifty**2))

    
    shifts = {
        f"{xvar}_left": {xvar: -shiftx} if shiftx < x_len else None,
        f"{yvar}_down": {yvar: -shifty} if shifty < y_len else None,
        "diag_upleft": {xvar: -shiftdiag, yvar: -shiftdiag}
        if shiftdiag < min(x_len, y_len)
        else None,
        "diag_upright": {xvar: shiftdiag, yvar: -shiftdiag}
        if shiftdiag < min(x_len, y_len)
        else None,
    }

    # Remove None values from shifts
    shifts = {k: v for k, v in shifts.items() if v is not None}

    u_shifts = shift_fields(ds_tmp, uvar, shifts, shift_func)
    v_shifts = shift_fields(ds_tmp, vvar, shifts, shift_func)
    if qvar:
        q_shifts = shift_fields(ds_tmp, qvar, shifts, shift_func)
    

    u_adv_shifts = shift_fields(ds_tmp, f"advection_{uvar}", shifts, shift_func)
    v_adv_shifts = shift_fields(ds_tmp, f"advection_{vvar}", shifts, shift_func)
    if qvar:
        q_adv_shifts = shift_fields(ds_tmp, f"advection_{qvar}", shifts, shift_func)


    asf_shifts = {}
    asfq_shifts = {}
    for direction in shifts.keys():
        asf_shifts[f"asf_shift_{direction}"] = (
            u_adv_shifts[f"advection_{uvar}_shifted_{direction}"]
            - ds_tmp[f"advection_{uvar}"]
        ) * (u_shifts[f"{uvar}_shifted_{direction}"] - ds_tmp[uvar]) + (
            v_adv_shifts[f"advection_{vvar}_shifted_{direction}"]
            - ds_tmp[f"advection_{vvar}"]
        ) * (v_shifts[f"{vvar}_shifted_{direction}"] - ds_tmp[vvar])
        if qvar:
            asfq_shifts[f"asfq_shift_{direction}"] = (
                q_adv_shifts[f"advection_{qvar}_shifted_{direction}"]
                - ds_tmp[f"advection_{qvar}"]
            ) * (q_shifts[f"{qvar}_shifted_{direction}"] - ds_tmp[qvar])

    asf_shift_keys = list(asf_shifts.keys())
    asf_average = sum(asf_shifts[key] for key in asf_shift_keys) / len(asf_shift_keys)


    if qvar:
        asfq_shift_keys = list(asfq_shifts.keys())
        asfq_average = sum(asfq_shifts[key] for key in asfq_shift_keys) / len(
            asfq_shift_keys
        )

    if just_mean:
        if full:
            asf_means = {
                key: asf_shifts[key].mean(dim=[xvar, yvar]).values
                for key in asf_shift_keys
            }
            if qvar:
                asfq_means = {
                    key: asfq_shifts[key].mean(dim=[xvar, yvar]).values
                    for key in asfq_shift_keys
                }
                return asf_means, asfq_means
            else:
                return asf_means
    else:
        if qvar:
            return asf_average, asfq_average
        else:
            return asf_average

@dask.delayed
def compute_ASF_track(
    ds,
    shiftnum=1,
    shiftvar='y',
    full=True,
    uvar="u",
    vvar="v",
    qvar="q",
    xvar="x",
    yvar="y",
    shift_func="roll",
    just_mean=False,
    calc_error=True,
):

    ds_tmp = copy(ds)

    # x_len = len(ds_tmp[xvar])
    # y_len = len(ds_tmp[yvar])

    shifts = {
        f"{shiftvar}": {shiftvar: -shiftnum}
    }

    # decompose velocity vector (u,v) into longitudinal and transverse components along shift direction
    ds_tmp['uL'] = xr.where(shiftvar == xvar, ds_tmp[uvar], ds_tmp[vvar])
    ds_tmp['uT'] = xr.where(shiftvar == xvar, ds_tmp[vvar], ds_tmp[uvar])

    ds_tmp['uLuT'] = ds_tmp['uL'] * ds_tmp['uT']

    dshift = ds_tmp[shiftvar][1] - ds_tmp[shiftvar][0]

    ds_tmp['duLdxL'] = ds_tmp['uL'].differentiate(xvar if shiftvar==xvar else yvar) / dshift
    ds_tmp['duTdxL'] = ds_tmp['uT'].differentiate(xvar if shiftvar==xvar else yvar) / dshift
    ds_tmp['duLuTdxL'] = ds_tmp['uLuT'].differentiate(xvar if shiftvar==xvar else yvar) / dshift

    ds_tmp['uLduLdxL'] = ds_tmp['uL'] * ds_tmp['duLdxL']
    ds_tmp['uLduTdxL'] = ds_tmp['uL'] * ds_tmp['duTdxL']

    uL_shifts = shift_fields(ds_tmp, 'uL', shifts, shift_func)
    uT_shifts = shift_fields(ds_tmp, 'uT', shifts, shift_func)
    uLduLdxL_shifts = shift_fields(ds_tmp, 'uLduLdxL', shifts, shift_func)
    uLduTdxL_shifts = shift_fields(ds_tmp, 'uLduTdxL', shifts, shift_func)
    duLuTdxL_shifts = shift_fields(ds_tmp, 'duLuTdxL', shifts, shift_func)

    for direction in shifts.keys():

        asf_term1 = 2 * (uL_shifts[f"uL_shifted_{direction}"] - ds_tmp['uL']) * (uLduLdxL_shifts[f"uLduLdxL_shifted_{direction}"] - ds_tmp['uLduLdxL'])
        asf_term2 = 2 * (uT_shifts[f"uT_shifted_{direction}"] - ds_tmp['uT']) * (uLduTdxL_shifts[f"uLduTdxL_shifted_{direction}"] - ds_tmp['uLduTdxL'])            
        asf_term3 = (uT_shifts[f"uT_shifted_{direction}"] - ds_tmp['uT']) * (duLuTdxL_shifts[f"duLuTdxL_shifted_{direction}"] - ds_tmp['duLuTdxL'])

        asf_partial = asf_term1 + asf_term2 - asf_term3

        ds[f'ASF_track_{shiftvar}'] = asf_partial
        ds[f'ASF_track_{shiftvar}_term1'] = asf_term1
        ds[f'ASF_track_{shiftvar}_term2'] = asf_term2
        ds[f'ASF_track_{shiftvar}_term3'] = asf_term3

        asf_basic_term1 = (uL_shifts[f"uL_shifted_{direction}"] - ds_tmp['uL']) * (uLduLdxL_shifts[f"uLduLdxL_shifted_{direction}"] - ds_tmp['uLduLdxL'])
        asf_basic_term2 = (uT_shifts[f"uT_shifted_{direction}"] - ds_tmp['uT']) * (uLduTdxL_shifts[f"uLduTdxL_shifted_{direction}"] - ds_tmp['uLduTdxL'])


        asf_basic_total = asf_basic_term1 + asf_basic_term2

        ds[f'ASF_basic_total_{shiftvar}'] = asf_basic_total
        ds[f'ASF_basic_term1_{shiftvar}'] = asf_basic_term1
        ds[f'ASF_basic_term2_{shiftvar}'] = asf_basic_term2

    if calc_error:
        ds_tmp['duLuTdxT'] = ds_tmp['uLuT'].differentiate(xvar if shiftvar==yvar else yvar) / dshift
        ds_tmp['duTdxT'] = ds_tmp['uT'].differentiate(xvar if shiftvar==yvar else yvar) / dshift
        ds_tmp['divergence'] = ds_tmp['duLdxL'] + ds_tmp['duTdxT']
        ds_tmp['uLD'] = ds_tmp['uL'] * ds_tmp['divergence']
        ds_tmp['uTD'] = ds_tmp['uT'] * ds_tmp['divergence']
        ds_tmp['uTduLdxT'] = ds_tmp['uT'] * ds_tmp['duLdxL']
        ds_tmp['uTduTdxT'] = ds_tmp['uT'] * ds_tmp['duTdxT']

        duLuTdxT_shifts = shift_fields(ds_tmp, 'duLuTdxT', shifts, shift_func)
        uLD_shifts = shift_fields(ds_tmp, 'uLD', shifts, shift_func)
        uTD_shifts = shift_fields(ds_tmp, 'uTD', shifts, shift_func)
        uTduLdxT_shifts = shift_fields(ds_tmp, 'uTduLdxT', shifts, shift_func)
        uTduTdxT_shifts = shift_fields(ds_tmp, 'uTduTdxT', shifts, shift_func)

        for direction in shifts.keys():

            asf_error_term1 = (uL_shifts[f"uL_shifted_{direction}"] - ds_tmp['uL']) * (duLuTdxT_shifts[f"duLuTdxT_shifted_{direction}"] - ds_tmp['duLuTdxT'])
            asf_error_term2 = (uL_shifts[f"uL_shifted_{direction}"] - ds_tmp['uL']) * (uLD_shifts[f"uLD_shifted_{direction}"] - ds_tmp['uLD'])            
            asf_error_term3 = (uT_shifts[f"uT_shifted_{direction}"] - ds_tmp['uT']) * (uTD_shifts[f"uTD_shifted_{direction}"] - ds_tmp['uTD'])

            asf_error = asf_error_term1 - asf_error_term2 + asf_error_term3

            ds[f'ASF_track_error_{shiftvar}'] = asf_error
            ds[f'ASF_track_error_{shiftvar}_term1'] = asf_error_term1
            ds[f'ASF_track_error_{shiftvar}_term2'] = asf_error_term2
            ds[f'ASF_track_error_{shiftvar}_term3'] = asf_error_term3

            asf_error_basic_term1 = (uL_shifts[f"uL_shifted_{direction}"] - ds_tmp['uL']) * (uTduLdxT_shifts[f"uTduLdxT_shifted_{direction}"] - ds_tmp['uTduLdxT'])
            asf_error_basic_term2 = (uT_shifts[f"uT_shifted_{direction}"] - ds_tmp['uT']) * (uTduTdxT_shifts[f"uTduTdxT_shifted_{direction}"] - ds_tmp['uTduTdxT'])

            asf_error_basic_total = asf_error_basic_term1 + asf_error_basic_term2
            ds[f'ASF_error_basic_total_{shiftvar}'] = asf_error_basic_total
            ds[f'ASF_error_basic_term1_{shiftvar}'] = asf_error_basic_term1
            ds[f'ASF_error_basic_term2_{shiftvar}'] = asf_error_basic_term2

    return ds



@dask.delayed
def compute_LLL(
    ds,
    shiftnum=1,
    shiftx=None,
    shifty=None,
    shiftdiag=None,
    full=True,
    uvar="u",
    vvar="v",
    xvar="x",
    yvar="y",
    qvar=None,
    shift_func="roll",
    just_mean=False,
):
    ds_tmp = copy(ds)

    x_len = len(ds_tmp[xvar])
    y_len = len(ds_tmp[yvar])

    if shiftx is None:
        shiftx = shiftnum
    if shifty is None:
        shifty = shiftnum

    shiftdiag = int(np.sqrt(shiftx**2 + shifty**2) )

    shifts = {
        f"{xvar}_left": {xvar: -shiftx} if shiftx < x_len else None,
        f"{yvar}_down": {yvar: -shifty} if shifty < y_len else None,
        "diag_upleft": {xvar: -shiftdiag, yvar: -shiftdiag}
        if shiftdiag < min(x_len, y_len)
        else None,
        "diag_upright": {xvar: shiftdiag, yvar: -shiftdiag}
        if shiftdiag < min(x_len, y_len)
        else None,
    }

    # Remove None values from shifts
    shifts = {k: v for k, v in shifts.items() if v is not None}

    u_shifts = shift_fields(ds_tmp, uvar, shifts, shift_func)
    v_shifts = shift_fields(ds_tmp, vvar, shifts, shift_func)
    if qvar:
        q_shifts = shift_fields(ds_tmp, qvar, shifts, shift_func)

    LLL_shifts = {}
    Lqq_shifts = {}

    for direction in shifts.keys():
        if "left" in direction:
            LLL_shifts[f"LLL_shift_{direction}"] = (u_shifts[f"{uvar}_shifted_{direction}"] - ds_tmp[uvar]) ** 3
        if "down" in direction:
            LLL_shifts[f"LLL_shift_{direction}"] = (v_shifts[f"{vvar}_shifted_{direction}"] - ds_tmp[vvar]) ** 3
        if "diag" in direction:
            LLL_shifts[f"LLL_shift_{direction}"] = (
                u_shifts[f"{uvar}_shifted_{direction}"] - ds_tmp[uvar]
            ) ** 2 * (v_shifts[f"{vvar}_shifted_{direction}"] - ds_tmp[vvar])
        if qvar:
            if "left" in direction:
                Lqq_shifts[f"L{qvar}{qvar}_shift_{direction}"] = (u_shifts[f"{uvar}_shifted_{direction}"] - ds_tmp[uvar]) * (q_shifts[f"{qvar}_shifted_{direction}"] - ds_tmp[qvar]) ** 2
            if "down" in direction:
                Lqq_shifts[f"L{qvar}{qvar}_shift_{direction}"] = (v_shifts[f"{vvar}_shifted_{direction}"] - ds_tmp[vvar]) * (q_shifts[f"{qvar}_shifted_{direction}"] - ds_tmp[qvar]) ** 2
            if "diag" in direction:
                Lqq_shifts[f"L{qvar}{qvar}_shift_{direction}"] = (u_shifts[f"{uvar}_shifted_{direction}"] - ds_tmp[uvar]) * (q_shifts[f"{qvar}_shifted_{direction}"] - ds_tmp[qvar]) ** 2

    LLL_shift_keys = list(LLL_shifts.keys())
    LLL_average = sum(LLL_shifts[key] for key in LLL_shift_keys) / len(LLL_shift_keys)

    if qvar:
        Lqq_shift_keys = list(Lqq_shifts.keys())
        Lqq_average = sum(Lqq_shifts[key] for key in Lqq_shift_keys) / len(Lqq_shift_keys)

    if just_mean:
        if full:
            LLL_means = {
                key: LLL_shifts[key].mean(dim=[xvar, yvar]).values
                for key in LLL_shift_keys
            }
            if qvar:
                Lqq_means = {
                    key: Lqq_shifts[key].mean(dim=[xvar, yvar]).values
                    for key in Lqq_shift_keys
                }
                return LLL_means, Lqq_means
            return LLL_means
    else:
        if qvar:
            return LLL_average, Lqq_average
        return LLL_average

@dask.delayed
def compute_LL(
    ds,
    shiftnum=1,
    shiftx=None,
    shifty=None,
    shiftdiag=None,
    uvar="u",
    vvar="v",
    xvar="x",
    yvar="y",
    shift_func="roll",
):
    ds_tmp = copy(ds)

    x_len = len(ds_tmp[xvar])
    y_len = len(ds_tmp[yvar])

    if shiftx is None:
        shiftx = shiftnum
    if shifty is None:
        shifty = shiftnum

    shiftdiag = int(np.sqrt(shiftx**2 + shifty**2) )

    shifts = {
        f"{xvar}_left": {xvar: -shiftx} if shiftx < x_len else None,
        f"{yvar}_down": {yvar: -shifty} if shifty < y_len else None,
        "diag_upleft": {xvar: -shiftdiag, yvar: -shiftdiag}
        if shiftdiag < min(x_len, y_len)
        else None,
        "diag_upright": {xvar: shiftdiag, yvar: -shiftdiag}
        if shiftdiag < min(x_len, y_len)
        else None,
    }

    # Remove None values from shifts
    shifts = {k: v for k, v in shifts.items() if v is not None}

    u_shifts = shift_fields(ds_tmp, uvar, shifts, shift_func)
    v_shifts = shift_fields(ds_tmp, vvar, shifts, shift_func)

    LL_shifts = {}

    for direction in shifts.keys():
        if "left" in direction:
            LL_shifts[f"LL_shift_{direction}"] = (u_shifts[f"{uvar}_shifted_{direction}"] - ds_tmp[uvar]) ** 2
        if "down" in direction:
            LL_shifts[f"LL_shift_{direction}"] = (v_shifts[f"{vvar}_shifted_{direction}"] - ds_tmp[vvar]) ** 2
        if "diag" in direction:
            LL_shifts[f"LLL_shift_{direction}"] = (
                u_shifts[f"{uvar}_shifted_{direction}"] - ds_tmp[uvar]
            ) * (v_shifts[f"{vvar}_shifted_{direction}"] - ds_tmp[vvar])

    LL_shift_keys = list(LL_shifts.keys())
    LL_average = sum(LL_shifts[key] for key in LL_shift_keys) / len(LL_shift_keys)

    return LL_average
    
def add_blur(ds, var, blur_radius=1, xvar="x", yvar="y", min_periods=2):
    ds[f"{var}_blurred"] = (
        ds[var]
        .rolling(
            dim={xvar: blur_radius, yvar: blur_radius},
            min_periods=min_periods,
            center=True,
        )
        .mean()
    )

    return ds

@dask.delayed
def get_bessels(
    ds,
    var,
    xvar,
    yvar=None,
    taper_SF=True,
    periodic=True,
    dx=None,
    dy=None,
    N=None,
    multi_sfs=False,
    Kname="K",
):
    
    ds_tmp = copy(ds)

    if dx is None:
        dx = abs(ds_tmp[f"{xvar}"][0] - ds_tmp[f"{xvar}"][1]).values
    if dy is None:
        dy = abs(ds_tmp[f"{yvar}"][0] - ds_tmp[f"{yvar}"][1]).values

    if N is None:
        if periodic:
            N = len(ds_tmp[f"{xvar}"]) // 2
        else:
            N = len(ds_tmp[f"{xvar}"])
    k_int = 1 / dx
    k = 2 * np.pi * np.arange(-N, N + 1) * (k_int / (2 * N))
    l_int = 1 / dy
    l = 2 * np.pi * np.arange(-N, N + 1) * (l_int / (2 * N))
    k_max_mat = max(np.max(-k[: N + 1]), np.max(l))
    dk = 2 * np.pi / (N * dx)
    dl = 2 * np.pi / (N * dy)
    dkr_mat = np.sqrt(dk**2 + dl**2)
    kr = np.arange(dkr_mat / 2, k_max_mat + dkr_mat, dkr_mat)

    # add kr as a new variable with dimensions shiftnum
    ds_tmp[Kname] = xr.DataArray(
        kr, dims=[Kname], attrs={"units": "1/m", "long_name": "Wavenumber"}
    )

    # Taper the structure function if specified
    if taper_SF:
        sf = ds_tmp[var]
        sf_tapered = sf * np.sin(np.linspace(np.pi / 2, np.pi, len(sf)))[:, np.newaxis]
        ds_tmp[f"{var}_tapered"] = sf_tapered
        var = f"{var}_tapered"

    # Compute Bessel functions
    J1 = xr.DataArray(
        jv(1, ds_tmp[Kname] * ds_tmp[f"{xvar}_diffs"]),
        dims=[Kname, "shiftnum"],
        attrs={"long_name": f"First order Bessel function in {xvar}"},
    )

    ASF = ds_tmp[f"{var}"] * J1
    ds_tmp[f"EFlux_Bessel_{var}"] = (
        ASF.sum(dim="shiftnum") * ds_tmp[f"{xvar}_diffs"][1] * -ds_tmp[Kname] / 2
    )

    return ds_tmp


def load_2d_sims(file, save_netCDF=False):
    f = h5py.File(file, "r")

    # Load sim file into xarray
    x = np.linspace(-f["Lx"][0] / 2, f["Lx"][0] / 2, f["nx"][0])
    y = np.linspace(-f["Ly"][0] / 2, f["Ly"][0] / 2, f["ny"][0])
    u = f["u"][:]
    v = f["v"][:]
    zeta = f["zeta"][:]
    diss_rate_ens = f["Diss_Rate_ENS"][0]
    diss_rate_ke = f["Diss_Rate_KE"][0]
    drag_rate_ens = f["Drag_Rate_ENS"][0]
    drag_rate_ke = f["Drag_Rate_KE"][0]
    ens = f["ENS"][0]
    ke = f["KE"][0]
    work_rate_ens = f["Work_Rate_ENS"][0]
    work_rate_ke = f["Work_Rate_KE"][0]

    # Create xarray dataset
    ds_tmp = xr.Dataset(
        {
            "u": (["y", "x"], u),
            "v": (["y", "x"], v),
            "zeta": (["y", "x"], zeta),
            "diss_rate_ens": ([], diss_rate_ens),
            "diss_rate_ke": ([], diss_rate_ke),
            "drag_rate_ens": ([], drag_rate_ens),
            "drag_rate_ke": ([], drag_rate_ke),
            "ens": ([], ens),
            "ke": ([], ke),
            "work_rate_ens": ([], work_rate_ens),
            "work_rate_ke": ([], work_rate_ke),
        },
        coords={"x": x, "y": y},
    )

    f.close()

    if save_netCDF:
        ds_tmp.to_netcdf(file.replace(".mat", ".nc"))

    else:
        return ds_tmp


def load_multilayer_qg_sims(file, save_netCDF=False):
    f = h5py.File(file, "r")

    # Load sim file into xarray
    x = np.linspace(-f["Lx"][0] / 2, f["Lx"][0] / 2, f["nx"][0])
    y = np.linspace(-f["Ly"][0] / 2, f["Ly"][0] / 2, f["ny"][0])
    u = f["u"][0]
    v = f["v"][0]
    q = f["q"][0]
    KE = f["KE"][0]
    Lat_flux = f["Lat_flux"][0]
    PE = f["PE"][0]
    Vert_flux = f["Vert_flux"][0]

    # get time, which is the last number in the file name, it goes from 0 to 100
    time = int(file.split("_")[-1].split(".")[0])

    # Create xarray dataset
    ds_tmp = xr.Dataset(
        {
            "u": (["y", "x"], u),
            "v": (["y", "x"], v),
            "q": (["y", "x"], q),
            "KE": ([], KE),
            "Lat_flux": ([], Lat_flux),
            "PE": ([], PE),
            "Vert_flux": ([], Vert_flux),
            "time": ([], time),
        },
        coords={"x": x, "y": y},
    )

    f.close()

    if save_netCDF:
        ds_tmp.to_netcdf(file.replace(".mat", ".nc"))

    else:
        return ds_tmp


def load_sqg_sims(file, save_netCDF=False):
    f = h5py.File(file, "r")

    # Load sim file into xarray
    x = np.linspace(-f["Lx"][0] / 2, f["Lx"][0] / 2, f["nx"][0])
    y = np.linspace(-f["Ly"][0] / 2, f["Ly"][0] / 2, f["ny"][0])
    u = f["u"][:]
    v = f["v"][:]
    b = f["b"][:]
    buoyancy_dissipation_hyperviscosity = f["buoyancy_dissipation_hyperviscosity"][0]
    buoyancy_dissipation_hypoviscosity = f["buoyancy_dissipation_hypoviscosity"][0]
    buoyancy_variance = f["buoyancy_variance"][0]
    buoyancy_work = f["buoyancy_work"][0]
    kinetic_energy = f["kinetic_energy"][0]

    # get time, which is the last number in the file name, it goes from 0 to 100
    time = int(file.split("_")[-1].split(".")[0])

    # Create xarray dataset
    ds_tmp = xr.Dataset(
        {
            "u": (["y", "x"], u),
            "v": (["y", "x"], v),
            "b": (["y", "x"], b),
            "buoyancy_dissipation_hyperviscosity": (
                [],
                buoyancy_dissipation_hyperviscosity,
            ),
            "buoyancy_dissipation_hypoviscosity": (
                [],
                buoyancy_dissipation_hypoviscosity,
            ),
            "buoyancy_variance": ([], buoyancy_variance),
            "buoyancy_work": ([], buoyancy_work),
            "kinetic_energy": ([], kinetic_energy),
            "time": ([], time),
        },
        coords={"x": x, "y": y},
    )

    f.close()

    if save_netCDF:
        ds_tmp.to_netcdf(file.replace(".mat", ".nc"))

    else:
        return ds_tmp

# @dask.delayed
def calc_advection(ds, qvar="CALC", uvar="u", vvar="v", xvar="x", yvar="y", dx=1, dy=1):

    ds_tmp = copy(ds)

    ds_tmp[f"d{uvar}d{xvar}"] = ds_tmp[uvar].differentiate(f"{xvar}") / dx
    ds_tmp[f"d{uvar}d{yvar}"] = ds_tmp[uvar].differentiate(f"{yvar}") / dy
    ds_tmp[f"d{vvar}d{xvar}"] = ds_tmp[vvar].differentiate(f"{xvar}") / dx
    ds_tmp[f"d{vvar}d{yvar}"] = ds_tmp[vvar].differentiate(f"{yvar}") / dy

    ds_tmp[f"advection_{uvar}"] = (
        ds_tmp[uvar] * ds_tmp[f"d{uvar}d{xvar}"] + ds_tmp[vvar] * ds_tmp[f"d{uvar}d{yvar}"]
    )
    ds_tmp[f"advection_{vvar}"] = (
        ds_tmp[uvar] * ds_tmp[f"d{vvar}d{xvar}"] + ds_tmp[vvar] * ds_tmp[f"d{vvar}d{yvar}"]
    )

    if qvar == "CALC":
        ds_tmp["q"] = ds_tmp[f"d{vvar}d{xvar}"] - ds_tmp[f"d{uvar}d{yvar}"]
        qvar = "q"

    if qvar:
        ds_tmp[f"d{qvar}d{xvar}"] = ds_tmp[qvar].differentiate(f"{xvar}") / dx
        ds_tmp[f"d{qvar}d{yvar}"] = ds_tmp[qvar].differentiate(f"{yvar}") / dy
        ds_tmp[f"advection_{qvar}"] = (
            ds_tmp[uvar] * ds_tmp[f"d{qvar}d{xvar}"] + ds_tmp[vvar] * ds_tmp[f"d{qvar}d{yvar}"]
        )

    return ds_tmp

# @dask.delayed
def calc_LLC4320_Eta_timemean(filepath, coarsen=None):

    # Load all files matching the filepath pattern into a single xarray dataset
    ds = xr.open_mfdataset(filepath, combine='by_coords', data_vars='all', compat='no_conflicts')

    if coarsen:
        ds = ds.coarsen(i=coarsen, j=coarsen, boundary="trim").mean()

    # Calculate time mean of Eta
    Eta_timemean = ds['Eta'].mean(dim='time')

    return Eta_timemean

def compute_large_scale_mean(ds, var, xvar='num_pixels', yvar='num_lines', stack_dim='swath_num'):

    ds_tmp = copy(ds)

    ds_tmp[f"{var}_{xvar}_mean"] = ds_tmp[var].mean(dim=[xvar, stack_dim], skipna=True)

    coeffs = ds_tmp[f"{var}_{xvar}_mean"].polyfit(dim=yvar, deg=1, skipna=True)["polyfit_coefficients"]
    slope = coeffs.sel(degree=1)
    intercept = coeffs.sel(degree=0)

    fitted_line = slope * ds_tmp[yvar] + intercept
    fitted_full = fitted_line.broadcast_like(ds_tmp[var]).transpose(*ds_tmp[var].dims)

    ds_tmp[f"{var}_{xvar}_mean_fit"] = fitted_full

    return ds_tmp[f"{var}_{xvar}_mean_fit"]

def prep_for_lineplots(ds, xvar="x", yvar="y"):
    ds_spatialmean = ds.mean(dim=[xvar, yvar])

    ds_spatialmean[f"{xvar}d"] = xr.DataArray(
        ds_spatialmean[f"{xvar}_diffs"], dims=[f"{xvar}d"]
    )

    for var in ds_spatialmean.data_vars:
        if "shiftnum" in ds_spatialmean[var].dims:
            ds_spatialmean[var] = ds_spatialmean[var].rename({"shiftnum": f"{xvar}d"})

    return ds_spatialmean

def coarse_grain(
    ds,
    uvar="u",
    vvar="v",
    qvar=None,
    xvar="x",
    yvar="y",
    filter="gaussian",
    kwargs={"sigma": 1},
    dx=1,
    dy=1,
    size=1,
):
    ds_tmp = ds

    if filter == "gaussian":
        func = gaussian_filter
    elif filter == "uniform":
        func = uniform_filter
    elif filter == "astropy":
        func = convolve_fft

    ds_tmp["ul"] = xr.apply_ufunc(
        func,
        ds_tmp[uvar],
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
    )
    ds_tmp["vl"] = xr.apply_ufunc(
        func,
        ds_tmp[vvar],
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
    )

    ds_tmp["uul"] = xr.apply_ufunc(
        func,
        ds_tmp[uvar] ** 2,
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
    )

    ds_tmp["vvl"] = xr.apply_ufunc(
        func,
        ds_tmp[vvar] ** 2,
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
    )

    ds_tmp["uvl"] = xr.apply_ufunc(
        func,
        ds_tmp[uvar] * ds_tmp[vvar],
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
    )

    ds_tmp["tau_uu"] = ds_tmp["uul"] - (
        ds_tmp["ul"] * ds_tmp["ul"]
    )
    ds_tmp["tau_uv"] = ds_tmp["uvl"] - (
        ds_tmp["ul"] * ds_tmp["vl"]
    )
    ds_tmp["tau_vv"] = ds_tmp["vvl"] - (
        ds_tmp["vl"] * ds_tmp["vl"]
    )

    ds_tmp["duldx"] = ds_tmp["ul"].differentiate(xvar) / dx
    ds_tmp["duldy"] = ds_tmp["ul"].differentiate(yvar) / dy
    ds_tmp["dvldx"] = ds_tmp["vl"].differentiate(xvar) / dx
    ds_tmp["dvldy"] = ds_tmp["vl"].differentiate(yvar) / dy

    ds_tmp["EFlux_coarse_grain"] = -(
        ds_tmp["tau_uu"] * ds_tmp["duldx"]
        + ds_tmp["tau_uv"] * (ds_tmp["dvldx"] + ds_tmp["duldy"])
        + ds_tmp["tau_vv"] * ds_tmp["dvldy"]
    )

    if qvar:
        ds_tmp["ql"] = xr.apply_ufunc(
            func,
            ds_tmp[qvar],
            kwargs=kwargs,
            dask="parallelized",
            output_dtypes=[float],
        )

        ds_tmp["uql"] = xr.apply_ufunc(
            func,
            ds_tmp[uvar] * ds_tmp[qvar],
            kwargs=kwargs,
            dask="parallelized",
            output_dtypes=[float],
        )

        ds_tmp["vql"] = xr.apply_ufunc(
            func,
            ds_tmp[vvar] * ds_tmp[qvar],
            kwargs=kwargs,
            dask="parallelized",
            output_dtypes=[float],
        )

        ds_tmp["tau_uq"] = ds_tmp["uql"] - (ds_tmp["ul"] * ds_tmp["ql"])
        ds_tmp["tau_vq"] = ds_tmp["vql"] - (ds_tmp["vl"] * ds_tmp["ql"])

        ds_tmp["dqldx"] = ds_tmp["ql"].differentiate(xvar) / dx
        ds_tmp["dqldy"] = ds_tmp["ql"].differentiate(yvar) / dy

        ds_tmp["qFlux_coarse_grain"] = - (ds_tmp["dqldx"] * ds_tmp["tau_uq"] + ds_tmp["dqldy"] * ds_tmp["tau_vq"])

        return (
            ds_tmp["EFlux_coarse_grain"].mean(dim=[xvar, yvar], skipna=True).values,
            ds_tmp["qFlux_coarse_grain"].mean(dim=[xvar, yvar], skipna=True).values,
        )
    
    return ds_tmp["EFlux_coarse_grain"].mean(dim=[xvar, yvar], skipna=True).values

@dask.delayed
def compute_fourier_flux_finufft(ds, xvar="x", yvar="y", uvar="u", vvar="v", dx=1, dy=1):

    ds_tmp = copy(ds)

    # Compute advection terms if not already computed
    if f"advection_{uvar}" not in ds_tmp:
        ds_tmp = calc_advection(ds_tmp, uvar=uvar, vvar=vvar, xvar=xvar, yvar=yvar, dx=dx, dy=dy)

    # Define wavenumber dimension
    nx = len(ds_tmp[xvar])
    ny = len(ds_tmp[yvar])
    total_points = nx * ny
    ds_tmp = ds_tmp.assign_coords(
        wavenumber=xr.DataArray(
            np.arange(total_points),
            dims=["wavenumber"],
            attrs={"units": "1/m", "long_name": "Wavenumber"},
        )
    )

    # Compute 2D FFTs using finufft
    u_fft = finufft.nufft2d1(
        ds_tmp[xvar].values,
        ds_tmp[yvar].values,
        ds_tmp[uvar].values.flatten(),
        isign=-1,
        eps=1e-6,
    )
    v_fft = finufft.nufft2d1(
        ds_tmp[xvar].values,
        ds_tmp[yvar].values,
        ds_tmp[vvar].values.flatten(),
        isign=-1,
        eps=1e-6,
    )
    advection_u_fft = finufft.nufft2d1(
        ds_tmp[xvar].values,
        ds_tmp[yvar].values,
        ds_tmp[f"advection_{uvar}"].values.flatten(),
        isign=-1,
        eps=1e-10,
    )
    advection_v_fft = finufft.nufft2d1(
        ds_tmp[xvar].values,
        ds_tmp[yvar].values,
        ds_tmp[f"advection_{vvar}"].values.flatten(),
        isign=-1,
        eps=1e-10,
    )

    # Compute EKE gain/loss in wavenumber space where EKE gain/loss = - sum over k of Real(conj(u_fft) * advection_u_fft + conj(v_fft) * advection_v_fft). Then get EKE flux as the sum over Tk up to wavenumber k.
    Tk = -np.real(np.conj(u_fft) * advection_u_fft + np.conj(v_fft) * advection_v_fft)
    EFlux_k = np.cumsum(Tk) * (dx * dy)  # Multiply by area element
    ds_tmp["EFlux_fourier_finufft"] = xr.DataArray(
        EFlux_k,
        dims=["wavenumber"],
        attrs={"units": "m^2/s^3", "long_name": "EKE flux computed using Fourier transforms with finufft"},
    )
    return ds_tmp

@dask.delayed
def compute_fourier_flux(ds, xvar="x", yvar="y", uvar="u", vvar="v", dx=1, dy=1):

    ds_tmp = copy(ds)

    # Apply nine half-overlapping Hamming windows
    windows = []
    nx = len(ds_tmp[xvar])
    ny = len(ds_tmp[yvar])
    window_size_x = nx // 2
    window_size_y = ny // 2
    step_x = window_size_x // 2
    step_y = window_size_y // 2
    for i in range(0, nx - window_size_x + 1, step_x):
        for j in range(0, ny - window_size_y + 1, step_y):
            window = np.outer(
                np.hamming(window_size_y),
                np.hamming(window_size_x)
            )
            windows.append(((i, i + window_size_x), (j, j + window_size_y), window))
    
    # Define wavenumber dimension
    total_points = window_size_x * window_size_y
    ds_tmp = ds_tmp.assign_coords(
        wavenumber=xr.DataArray(
            np.arange(total_points),
            dims=["wavenumber"],
            attrs={"units": "1/m", "long_name": "Wavenumber"},
        )
    )

    # Check if ds_tmp["advection_{uvar}"] exists, if not, compute advection terms
    if f"advection_{uvar}" not in ds_tmp:
        ds_tmp = calc_advection(ds_tmp, uvar=uvar, vvar=vvar, xvar=xvar, yvar=yvar, dx=dx, dy=dy)

    # Compute energy flux for each window and average
    EFlux_list = []
    for (ix_start, ix_end), (iy_start, iy_end), window in windows:
        u_windowed = ds_tmp[uvar].isel({xvar: slice(ix_start, ix_end), yvar: slice(iy_start, iy_end)}) * window
        v_windowed = ds_tmp[vvar].isel({xvar: slice(ix_start, ix_end), yvar: slice(iy_start, iy_end)}) * window

        advection_u_windowed = ds_tmp[f"advection_{uvar}"].isel({xvar: slice(ix_start, ix_end), yvar: slice(iy_start, iy_end)}) * window
        advection_v_windowed = ds_tmp[f"advection_{vvar}"].isel({xvar: slice(ix_start, ix_end), yvar: slice(iy_start, iy_end)}) * window

        # Compute 2D FFTs
        u_fft = np.fft.fft2(u_windowed)
        v_fft = np.fft.fft2(v_windowed)
        advection_u_fft = np.fft.fft2(advection_u_windowed)
        advection_v_fft = np.fft.fft2(advection_v_windowed)

        # Compute EKE gain/loss in wavenumber space where EKE gain/loss = - sum over k of Real(conj(u_fft) * advection_u_fft + conj(v_fft) * advection_v_fft). Then get EKE flux as the sum over Tk up to wavenumber k.
        Tk = -np.real(np.conj(u_fft) * advection_u_fft + np.conj(v_fft) * advection_v_fft)
        EFlux_k = np.cumsum(Tk.flatten()) * (dx * dy)  # Multiply by area element
        EFlux_list.append(EFlux_k)
    EFlux_avg = np.mean(EFlux_list, axis=0)
    ds_tmp["EFlux_fourier"] = xr.DataArray(
        EFlux_avg,
        dims=["wavenumber"],
        attrs={"units": "m^2/s^3", "long_name": "EKE flux computed using Fourier transforms"},
    )
    return ds_tmp

def format_datasets(ds_list_computed, dx, dy, sf_type="asf", xvar="x", yvar="y",dim=["time"]):

    ds_list_left = [
        xr.DataArray(arr[f"{sf_type}_shift_{xvar}_left"], dims=dim)
        for arr in ds_list_computed
        if f"{sf_type}_shift_{xvar}_left" in arr
    ]
    ds_list_down = [
        xr.DataArray(arr[f"{sf_type}_shift_{yvar}_down"], dims=dim)
        for arr in ds_list_computed
        if f"{sf_type}_shift_{yvar}_down" in arr
    ]
    ds_list_diag_upleft = [
        xr.DataArray(arr[f"{sf_type}_shift_diag_upleft"], dims=[dim])
        for arr in ds_list_computed
        if f"{sf_type}_shift_diag_upleft" in arr
    ]
    ds_list_diag_upright = [
        xr.DataArray(arr[f"{sf_type}_shift_diag_upright"], dims=dim)
        for arr in ds_list_computed
        if f"{sf_type}_shift_diag_upright" in arr
    ]

    # print(f"{sf_type}_shift_{xvar}_left")
    # print(dim)
    # print(ds_list_computed[0])
    # print(ds_list_computed[0][f"{sf_type}_shift_{xvar}_left"])

    # Concatenate the DataArray objects along the 'shiftnum' dimension
    ds_left_computed = xr.concat(ds_list_left, dim="shiftnum")
    ds_down_computed = xr.concat(ds_list_down, dim="shiftnum")
    ds_diag_upleft_computed = xr.concat(ds_list_diag_upleft, dim="shiftnum")
    ds_diag_upright_computed = xr.concat(ds_list_diag_upright, dim="shiftnum")

    ds_left_computed[f"{xvar}_diffs"] = xr.DataArray(
        np.arange(0, len(ds_left_computed["shiftnum"])) * dx,
        dims=["shiftnum"],
        attrs={"units": "m", "long_name": f"Separation distance in {xvar}-direction"},
    )
    ds_down_computed[f"{yvar}_diffs"] = xr.DataArray(
        np.arange(0, len(ds_down_computed["shiftnum"])) * dy,
        dims=["shiftnum"],
        attrs={"units": "m", "long_name": f"Separation distance in {yvar}-direction"},
    )
    ds_diag_upleft_computed["diag_upleft_diffs"] = xr.DataArray(
        np.arange(0, len(ds_diag_upleft_computed["shiftnum"]))
        * np.sqrt(dx**2 + dy**2),
        dims=["shiftnum"],
        attrs={
            "units": "m",
            "long_name": "Separation distance in up-left diagonal direction",
        },
    )
    ds_diag_upright_computed["diag_upright_diffs"] = xr.DataArray(
        np.arange(0, len(ds_diag_upright_computed["shiftnum"]))
        * np.sqrt(dx**2 + dy**2),
        dims=["shiftnum"],
        attrs={
            "units": "m",
            "long_name": "Separation distance in up-right diagonal direction",
        },
    )

    longest_length = max(
        len(ds_left_computed),
        len(ds_down_computed),
        len(ds_diag_upleft_computed),
        len(ds_diag_upright_computed),
    )

    ds_left_computed = ds_left_computed.pad(
        {"shiftnum": (0, longest_length - len(ds_left_computed))},
        mode="constant",
        constant_values=None,
    )
    ds_down_computed = ds_down_computed.pad(
        {"shiftnum": (0, longest_length - len(ds_down_computed))},
        mode="constant",
        constant_values=None,
    )
    ds_diag_upleft_computed = ds_diag_upleft_computed.pad(
        {"shiftnum": (0, longest_length - len(ds_diag_upleft_computed))},
        mode="constant",
        constant_values=None,
    )
    ds_diag_upright_computed = ds_diag_upright_computed.pad(
        {"shiftnum": (0, longest_length - len(ds_diag_upright_computed))},
        mode="constant",
        constant_values=None,
    )

    # Combine into a single dataset
    ds_computed = xr.Dataset(
        {
            f"{sf_type}_left": ds_left_computed,
            f"{sf_type}_down": ds_down_computed,
            f"{sf_type}_diag_upleft": ds_diag_upleft_computed,
            f"{sf_type}_diag_upright": ds_diag_upright_computed,
        }
    )

    ds_computed[f"{sf_type}_mean"] = sum(
        ds_computed[var] for var in ds_computed.data_vars
    ) / len(ds_computed.data_vars)

    return ds_computed


def get_region_latlon(region_name):

    filepath = f"/Volumes/Promise Disk/data/mitgcm/{region_name}_data/*.nc"

    ds = xr.open_mfdataset(
        filepath,
        combine="nested",
        concat_dim="time",
        coords="minimal",
        compat="override",
    )

    ds = ds.isel(time=0)[{"XC", "YC"}]

    lat_min = ds["YC"].min().values
    lat_max = ds["YC"].max().compute().item()
    lon_min = ds["XC"].min().compute().item()
    lon_max = ds["XC"].max().compute().item()

    region_latlon = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }

    return region_latlon

def process_one_path(path, transform_func=None):
    # use a context manager, to ensure the file gets closed after use
    with xr.open_dataset(path) as ds:
        # transform_func should do some sort of selection or
        # aggregation
        if transform_func is not None:
            ds = transform_func(ds)
        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        ds.load()
        return ds

def read_netcdfs(files, dim, transform_func=None):

    paths = sorted(glob.glob(files))
    datasets = [process_one_path(p) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined

def read_netcdfs_cutlatlon(files, dim, transform_func=None, lat_south=None, lat_north=None, lon_west=None, lon_east=None):
    """
    Read multiple NetCDF files, apply a transform function, filter by lat/lon, 
    and combine them along a specified dimension. Tracks processing time for each dataset
    and periodically logs progress.
    """
    paths = sorted(glob.glob(files))
    datasets = []
    total_files = len(paths)
    processed_count = 0
    skipped_count = 0
    start_time = time.time()
    last_log_time = start_time

    for i, p in enumerate(paths):
        file_start_time = time.time()
        # print(f"Processing file {i + 1}/{total_files}: {p}")

        try:
            ds = process_one_path(p, transform_func=transform_func)
            # print(ds.ugos.sizes)
            # Apply latitude and longitude filtering
            if lat_south is not None and lat_north is not None:
                lat_indexer = (ds["latitude"] >= lat_south) & (ds["latitude"] <= lat_north)
            else:
                lat_indexer = True

            if lon_west is not None and lon_east is not None:
                lon_indexer = (ds["longitude"] >= lon_west) & (ds["longitude"] <= lon_east)
            else:
                lon_indexer = True

            ds = ds.where(lat_indexer & lon_indexer, drop=True)
            # print(ds.ugos.sizes)
            # print(ds.sizes.get(dim,0))
            # Skip datasets with size 0
            if ds.dims["num_lines"] > 0:
                datasets.append(ds)
                processed_count += 1
                file_elapsed_time = time.time() - file_start_time
                # print(f"File {i + 1}/{total_files} processed in {file_elapsed_time:.2f} seconds.")
            else:
                skipped_count += 1
                # print(f"Skipping file {p} as it has no data after filtering.")

            if i == 0:
                print(f"Total files to process: {total_files}, for an estimated processing time of {file_elapsed_time * total_files:.2f} seconds, or {(file_elapsed_time * total_files) / 60:.2f} minutes.")

        except Exception as e:
            skipped_count += 1
            print(f"Error processing file {p}: {e}")

        # Periodically log progress every 300 seconds
        current_time = time.time()
        if current_time - last_log_time >= 300:  # 5 minutes
            print(f"Progress: {processed_count} files processed, {skipped_count} files skipped. {total_files - processed_count - skipped_count} files remaining. Elapsed time: {current_time - start_time:.2f} seconds. Estimated time remaining: {(total_files - processed_count - skipped_count) * (current_time - start_time) / (processed_count + skipped_count + 1):.2f} seconds, or {(total_files - processed_count - skipped_count) * (current_time - start_time) / (processed_count + skipped_count + 1) / 60:.2f} minutes.")
            print(f"Last file processed: {p}")
            last_log_time = current_time

    # Combine all valid datasets
    if datasets:
        combined = xr.concat(datasets, dim)
        total_elapsed_time = time.time() - start_time
        print(f"All files processed in {total_elapsed_time:.2f} seconds.")
        print(f"Final count: {processed_count} files processed, {skipped_count} files skipped.")
        return combined
    else:
        raise ValueError("No valid datasets to combine after filtering.")

def read_netcdfs_with_dask(files, dim, transform_func=None):
    """
    Read multiple NetCDF files using Dask Bags for lazy and parallel processing.
    """
    paths = sorted(glob.glob(files))

    # Create a Dask Bag to process files lazily
    bag = db.from_sequence(paths, npartitions=10)  # Adjust npartitions based on your system

    # Process each file lazily
    datasets = bag.map(lambda p: process_one_path(p, transform_func=transform_func))

    # Compute and combine the datasets
    combined = xr.concat(datasets.compute(), dim=dim)

    return combined


def _get_dataset_dim_size(path, dim_name):
    with xr.open_dataset(path) as ds_local:
        return ds_local.sizes[dim_name]


def _load_select_pad_dataset(path, variables, line_dim, max_num_lines, pass_attr=None, coarsen=None):
    with xr.open_dataset(path) as ds_local:
        ds_selected = ds_local[list(variables)]


        ds_padded = ds_selected.pad(
            {line_dim: (0, max_num_lines - ds_selected.sizes[line_dim])},
            constant_values=np.nan,
        )

        if coarsen:
            ds_padded = ds_padded.coarsen(num_lines=coarsen, num_pixels=coarsen, boundary="trim").mean()

        ds_padded.load()

        if pass_attr == "GET": 
            # extract pass number from filename, with example filename SWOT_L3_LR_SSH_Expert_001_149_20230726T122756_20230726T131923_v3.0.nc where 149 is the pass number
            filename = path.split("/")[-1]
            pass_value = int(filename.split("_")[6])

        else:
            pass_value = ds_local.attrs.get(pass_attr) if pass_attr else None
        # if pass_value is not None, check if it's a string that can be converted to an integer, and if so, convert it to an integer
        if pass_value is not None and isinstance(pass_value, str) and pass_value.isdigit():
            pass_value = int(pass_value)
        return ds_padded, pass_value


def load_and_pad_swath_datasets(
    file_paths,
    variables,
    line_dim="num_lines",
    concat_dim="swath_num",
    pass_attr=None,
    max_num_lines=9869,
    coarsen=None,
):

    print("Loading datasets and padding them to the maximum number of lines")
    delayed_results = [
        dask.delayed(_load_select_pad_dataset)(
            fp,
            variables,
            line_dim,
            max_num_lines,
            pass_attr,
            coarsen,
        )
        for fp in file_paths
    ]
    computed_results = dask.compute(*delayed_results)
    padded_datasets = [result[0] for result in computed_results]
    pass_numbers = [result[1] for result in computed_results]
    print(f"Prepared {len(padded_datasets)} padded datasets")

    # if longitude and latitude are not in a dataset, drop those datasets and print a warning
    for i, ds in enumerate(padded_datasets):
        if "longitude" not in ds or "latitude" not in ds:
            print(f"Warning: Dataset {file_paths[i]} is missing longitude or latitude. This dataset will be dropped.")
            padded_datasets[i] = None
    padded_datasets = [ds for ds in padded_datasets if ds is not None]

    ds = xr.concat(padded_datasets, dim=concat_dim)
    if pass_attr:
        ds = ds.assign_coords({concat_dim: pass_numbers}).sortby(concat_dim)
    print(f"Concatenated dataset shape: {ds.sizes}")

    return ds

def compute_and_save_llc4320(
    filename="LLC4320",
    region_name="acc",
    child_dir=None,
    date="20111113",
    outpath="data",
    sigma=None,
    timemean_removal_method=None,
    reduce_xd_num=1,
    ASF=True,
    CG=False,
    LLL=False,
    LL=False,
    Bessels=False,
    FFT=False,
    scalar=None,
    taper_SF=False,
    coarsen=None,

):
    # Load data

    filename += f"_{region_name}_{date}"

    if child_dir:
        filepath_pattern = f"/Volumes/Promise Disk/data/mitgcm/{region_name}_data/{child_dir}/*_{date}.nc"
    else:
        filepath_pattern = f"/Volumes/Promise Disk/data/mitgcm/{region_name}_data/*_{date}.nc"


    # Use glob to find the matching file
    matching_files = glob.glob(filepath_pattern)

    if not matching_files:
        raise FileNotFoundError(f"No files found matching pattern: {filepath_pattern}")

    # Open the first matching file
    filepath = matching_files[0]
    ds = xr.open_dataset(filepath)
    print(f"Opened file: {filepath}")

    ds = ds[{"Eta", "XC", "YC", "DXV", "DYU"}]

    if coarsen:
        ds = ds.coarsen(i=coarsen, j=coarsen, boundary="trim").mean()
        new_resolution_string = f"{(48 / coarsen):.1f}"
        filename += f"_coarsen_to_1div{new_resolution_string}deg"
        print(f"Coarsened data by a factor of {coarsen} to approximately 1/{new_resolution_string} degree resolution")
        print(f"Dataset shape after coarsening: {ds.sizes}")

    if timemean_removal_method == "snapshot_mean":
        Eta_timemean = calc_LLC4320_Eta_timemean(f"/Volumes/Promise Disk/data/mitgcm/{region_name}_data/{child_dir}/*.nc", coarsen=coarsen)
        ds["Eta"] = ds["Eta"] - Eta_timemean
        filename += "_timemean_removed_snapshot_mean"
        print("Removed time mean from Eta")



    # Calculate geostrophic velocities and advection
    ds["dEtadx"] = ds["Eta"].differentiate("i") / ds["DXV"]
    ds["dEtady"] = ds["Eta"].differentiate("j") / ds["DYU"]

    omega = 7.2921e-5
    g = 9.81

    ds["f"] = 2 * omega * np.sin(ds.YC * np.pi / 180)

    ds["u_geo"] = -(g / ds["f"]) * ds["dEtady"]
    ds["v_geo"] = (g / ds["f"]) * ds["dEtadx"]

    if sigma:
        ds["u_geo_filtered"] = xr.DataArray(
            gaussian_filter(ds["u_geo"].values, sigma=sigma), dims=["time", "j", "i"]
        )
        ds["v_geo_filtered"] = xr.DataArray(
            gaussian_filter(ds["v_geo"].values, sigma=sigma), dims=["time", "j", "i"]
        )
        uvar = "u_geo_filtered"
        vvar = "v_geo_filtered"

        filename += f"_sigma{sigma}"

    else:
        uvar = "u_geo"
        vvar = "v_geo"

    ds = ds.compute()
    print("Computed geostrophic velocities")

    ecco_dx_est = ds["DXV"].mean().values * coarsen if coarsen else ds["DXV"].mean().values
    ecco_dy_est = ds["DYU"].mean().values * coarsen if coarsen else ds["DYU"].mean().values

    if ASF:

        if scalar:
            qvar_calc = 'CALC'
        else:
            qvar_calc = None

        ds = calc_advection(
            ds,
            qvar=qvar_calc,
            uvar=uvar,
            vvar=vvar,
            xvar="i",
            yvar="j",
            dx=ds["DXV"],
            dy=ds["DYU"],
        )
        print("Calculated advection")
    
        # Calculate ASFs
        ds_list = [
            compute_asf(
                ds,
                shift_func="shift",
                uvar=uvar,
                vvar=vvar,
                qvar=scalar,
                xvar="i",
                yvar="j",
                shiftnum=sh,
                full=True,
                just_mean=True,
            )
            for sh in range(len(ds["i"]) // reduce_xd_num)
        ]
        print("Calculated ASFs")

        if scalar:
            filename += f"_{scalar}"

        if reduce_xd_num > 1:
            filename += f"_reduce_xd_num{reduce_xd_num}"


        # Set up ASF DataArray objects

        if scalar:

            # Compute ASFs
            ds_list_asf = [ds[0] for ds in ds_list]
            ds_list_asfq = [ds[1] for ds in ds_list]

            # Try batch computing to reduce memory usage
            # ds_list_computed_asf = dask.compute(*ds_list_asf)
            batch_size = len(ds_list_asf) // 2 
            ds_list_computed_asf = []
            
            for i in range(0, len(ds_list_asf), batch_size):
                batch = ds_list_asf[i:i+batch_size]
                computed_batch = dask.compute(*batch)
                ds_list_computed_asf.extend(computed_batch)
                print(f"Computed ASF batch {i//batch_size + 1}/{(len(ds_list_asf)-1)//batch_size + 1}")
                
                import gc
                gc.collect()

            print("Computed ASFs")
            
            ds_asf_computed = format_datasets(
                ds_list_computed_asf, ecco_dx_est, ecco_dy_est, sf_type="asf", xvar="i", yvar="j", dim="time"
            )

            # ds_list_computed_asfq = dask.compute(*ds_list_asfq)
            ds_list_computed_asfq = []

            for i in range(0, len(ds_list_asfq), batch_size):
                batch = ds_list_asfq[i:i+batch_size]
                computed_batch = dask.compute(*batch)
                ds_list_computed_asfq.extend(computed_batch)
                print(f"Computed ASFQ batch {i//batch_size + 1}/{(len(ds_list_asfq)-1)//batch_size + 1}")
                
                import gc
                gc.collect()

            print("Computed ASFQ")
            ds_asfq_computed = format_datasets(
                ds_list_computed_asfq, ecco_dx_est, ecco_dy_est, sf_type="asfq", xvar="i", yvar="j", dim="time"
            )
            ds_computed = xr.merge([ds_asf_computed, ds_asfq_computed])
            print("Formatted ASFs with scalar")
        else:

            # Compute ASFs
            ds_list_computed = dask.compute(*ds_list)
            print("Computed ASFs")

            ds_computed = format_datasets(
                ds_list_computed, ecco_dx_est, ecco_dy_est, sf_type="asf", xvar="i", yvar="j", dim="time"
            )
            print("Formatted ASFs")

    if LLL:
        # Calculate LLL SFs
        ds_list_LLL = [
            compute_LLL(
                ds,
                shift_func="shift",
                uvar=uvar,
                vvar=vvar,
                xvar="i",
                yvar="j",
                shiftnum=sh,
                full=True,
                just_mean=True,
            )
            for sh in range(len(ds["i"]) // reduce_xd_num)
        ]

        print("Calculated LLL SFs")
        # Compute LLL SFs
        ds_list_LLL_computed = dask.compute(*ds_list_LLL)
        print("Computed LLL SFs")
        # Set up LLL DataArray objects
        ds_LLL_computed = format_datasets(
            ds_list_LLL_computed, ecco_dx_est, ecco_dy_est, sf_type="LLL", xvar="i", yvar="j", dim="time"
        )
        print("Formatted LLL SFs")

        filename += "_LLL"

    if LL:
        # Calculate LL SFs
        ds_list_LL = [
            compute_LL(
                ds,
                shift_func="shift",
                uvar=uvar,
                vvar=vvar,
                xvar="i",
                yvar="j",
                shiftnum=sh,
            )
            for sh in range(len(ds["i"]) // reduce_xd_num)
        ]

        # Compute LL SFs
        ds_list_LL_computed = dask.compute(*ds_list_LL)

        # Set up LL DataArray objects
        ds_LL_computed = format_datasets(
            ds_list_LL_computed, ecco_dx_est, ecco_dy_est, sf_type="LL"
        )
        print("Computed LL SFs")

        filename += "_LL"

    if Bessels:
        # Calculate fluxes from Bessel functions
        ds_bessels = get_bessels(
            ds_computed,
            var="asf_left",
            xvar="i",
            yvar="j",
            taper_SF=taper_SF,
            periodic=False,
            dx=ecco_dx_est,
            dy=ecco_dy_est,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_down",
            xvar="i",
            yvar="j",
            taper_SF=taper_SF,
            periodic=False,
            dx=ecco_dx_est,
            dy=ecco_dy_est,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upleft",
            xvar="i",
            yvar="j",
            taper_SF=taper_SF,
            periodic=False,
            dx=ecco_dx_est,
            dy=ecco_dy_est,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upright",
            xvar="i",
            yvar="j",
            taper_SF=taper_SF,
            periodic=False,
            dx=ecco_dx_est,
            dy=ecco_dy_est,
            N=len(ds_computed["shiftnum"]),
        )

        if taper_SF:
            taper_var = "tapered"
        else:
            taper_var = ""

        # Compute the delayed Bessel object before assigning to it
        ds_bessels = dask.compute(ds_bessels)[0]

        ds_bessels[f"EFlux_Bessel_ASF_mean_{taper_var}"] = (
            sum(ds_computed[var] for var in ds_computed.data_vars if "Bessel" in var)
            / 4
        )

        # ds_bessels_timemean = ds_bessels.mean(dim="time")

        print("Computed Bessel functions")

        filename += "_Bessels"

        if taper_SF:
            filename += "_tapered"

    if FFT:
        delayed_compute_fourier_flux = dask.delayed(compute_fourier_flux)
        print("Delayed Fourier flux computation")

        ds_fft_list = [
            delayed_compute_fourier_flux(
                ds.isel(time=i),
                xvar="i",
                yvar="j",
                uvar=uvar,
                vvar=vvar,
                dx=ecco_dx_est,
                dy=ecco_dy_est,
            )
            for i in range(len(ds["time"]))
        ]

        print("Set up Fourier flux computation")

        ds_fft_list_computed = dask.compute(*ds_fft_list)

        print("Computed Fourier flux")

        # Extract EFlux_fourier from each computed dataset
        eflux_list = [
            computed_ds["EFlux_fourier"] for computed_ds in ds_fft_list_computed
        ]

        # Concatenate along time dimension
        ds_fft_computed = xr.concat(eflux_list, dim="time")
        ds_fft = xr.Dataset({"EFlux_fourier": ds_fft_computed})

        # Get wavenumber coordinates from first dataset
        ds_fft["wavenumber"] = ds_fft_list_computed[0]["wavenumber"]

        print("Formatted Fourier flux")
        filename += "_FFT"

    if CG:
        # Calculate fluxes by coarse graining
        sizes = np.logspace(0, 2.5, 50)

        # ds = dask.delayed(ds)
        # print('Computed data')

        delayed_coarse_grain = dask.delayed(coarse_grain)
        print("Delayed coarse graining")

        if sigma:
            uvar = "u_geo_filtered"
            vvar = "v_geo_filtered"
        else:
            uvar = "u_geo"
            vvar = "v_geo"

        ds_cg_list = [
            delayed_coarse_grain(
                ds.isel(time=i),
                filter="astropy",
                uvar=uvar,
                vvar=vvar,
                xvar="i",
                yvar="j",
                qvar=scalar,
                dx=ecco_dx_est,
                dy=ecco_dy_est,
                kwargs={"kernel": Tophat2DKernel(s)},
            )
            for s in sizes
            for i in range(len(ds["time"]))
        ]

        print("Set up coarse graining")

        ds_cg_list_computed = dask.compute(*ds_cg_list)

        print("Computed coarse graining")

        if scalar is None:

            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), len(ds["time"])
            )

            ds_cg_list_ = [
                xr.DataArray(reshaped_ds_cg_list[i, :], dims=["time"])
                for i in range(len(sizes))
            ]
            ds_cg_computed = xr.concat(ds_cg_list_, dim="L")
            ds_cg_computed = xr.Dataset({"EFlux_CG": ds_cg_computed})
            ds_cg_computed["L"] = sizes
            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * np.mean([ecco_dx_est, ecco_dy_est]))
            )
        
        else:
            
            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), len(ds["time"]), 2
            )

            ds_cg_list_EFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 0], dims=["time"])
                for i in range(len(sizes))
            ]
            ds_cg_list_QFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 1], dims=["time"])
                for i in range(len(sizes))
            ]

            ds_cg_computed_EFlux = xr.concat(ds_cg_list_EFlux, dim="L")
            ds_cg_computed_QFlux = xr.concat(ds_cg_list_QFlux, dim="L")

            ds_cg_computed = xr.Dataset({
                "EFlux_CG": ds_cg_computed_EFlux, 
                "QFlux_CG": ds_cg_computed_QFlux
                })
            ds_cg_computed["L"] = sizes
            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * np.mean([ecco_dx_est, ecco_dy_est]))
            )

        print("Formatted coarse graining")
        filename += "_CG"

    # Save to netCDF
    # Build list of datasets to merge
    datasets_to_merge = []

    if ASF:
        datasets_to_merge.append(ds_computed)
    if Bessels:
        datasets_to_merge.append(ds_bessels)
    if CG:
        datasets_to_merge.append(ds_cg_computed)
    if FFT:
        datasets_to_merge.append(ds_fft)
    if LLL:
        datasets_to_merge.append(ds_LLL_computed)
    if LL:
        datasets_to_merge.append(ds_LL_computed)

    # Merge all datasets at once
    ds_all = xr.merge(datasets_to_merge) if len(datasets_to_merge) > 1 else datasets_to_merge[0]

    datetime_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    ds_all.to_netcdf(f"{outpath}/{datetime_str}_{filename}.nc")
    print(f"Saved to {outpath}/{datetime_str}_{filename}.nc")        

    return None

def compute_and_save_simswot(
    filename="simSWOT",
    prepadded_mfdataset=None,
    expand_cyclemean_num=None,
    filepath="/Volumes/Promise Disk/data/simulated_swot/full_data/*.nc",
    mean_filepath=None,
    region_name="acc",
    cycle_num='001',
    cycle_group=None,
    lat_north=60,
    lat_south=-60,
    lon_west=-180,
    lon_east=180,
    outpath="data/simSWOT/",
    cleaned=False,
    reduce_xd_num=1,
    timemean_removal_method="cycle_mean",
    ASF=True,
    CG=False,
    LLL=False,
    Bessels=False,
    taper_SF=False,
    scalar=None,
    flip_swath=False,
    coarsen=None,
):
    # Load data


    if prepadded_mfdataset is None:
        print(f"Loading data from {filepath}")
        filename += f"_{region_name}_cycle_{cycle_num}"

        file_paths = glob.glob(filepath)

        # Drop filepaths that do not contain the cycle number
        file_paths = [fp for fp in file_paths if f"Expert_{cycle_num}_" in fp]

        ds = load_and_pad_swath_datasets(
            file_paths=file_paths,
            variables=[
                "time",
                "ssh_karin",
                "ssh_karin_2",
                "simulated_true_ssh_karin",
                "simulated_error_karin",
                "simulated_error_baseline_dilation",
                "simulated_error_roll",
                "simulated_error_phase",
                "simulated_error_timing",
                "simulated_error_orbital",
                "simulated_error_troposphere",
            ],
            line_dim="num_lines",
            concat_dim="swath_num",
            pass_attr="pass_number",
        )
    
    else:
        print("Using pre-padded dataset which should already have the desired cycles.")
        filename += f"_{region_name}_cycles_{cycle_group[0]}-{cycle_group[-1]}"
        ds = prepadded_mfdataset

    if coarsen:
        ds = ds.coarsen(num_lines=coarsen, num_pixels=coarsen, boundary="trim").mean()
        new_resolution_string = f"{(48 / coarsen):.1f}"
        filename += f"_coarsen_to_1div{new_resolution_string}deg"
        print(f"Coarsened data by a factor of {coarsen} to approximately 1/{new_resolution_string} degree resolution")
        print(f"Dataset shape after coarsening: {ds.sizes}")


    if timemean_removal_method == "cycle_mean":
        print("Loading cycle-meaned .nc files and padding them to the maximum number of lines.")
        mean_file_paths = glob.glob(mean_filepath)

        ds_cyclemean = load_and_pad_swath_datasets(
            file_paths=mean_file_paths,
            variables=[
                "time",
                "ssh_karin",
                "ssh_karin_2",
                "simulated_true_ssh_karin",
                "simulated_error_karin",
                "simulated_error_baseline_dilation",
                "simulated_error_roll",
                "simulated_error_phase",
                "simulated_error_timing",
                "simulated_error_orbital",
                "simulated_error_troposphere",
            ],
            line_dim="num_lines",
            concat_dim="swath_num",
            pass_attr="pass_number",
            coarsen=coarsen,
        )

        if expand_cyclemean_num is not None:
            # expand ds_cyclemean to have expand_cyclemean_num x swath_num, where each cycle-mean swath is repeated expand_cyclemean_num times
            ds_cyclemean = xr.concat([ds_cyclemean] * expand_cyclemean_num, dim="swath_num")

        # print shape of both datasets to confirm they match
        # print(f"Original dataset shape: {ds.sizes}")
        # print(f"Cycle-mean dataset shape: {ds_cyclemean.sizes}")

        # Drop ambiguous coordinate/variable names that can cause xarray MergeError
        # ambiguous = [name for name in ("latitude_nadir", "longitude_nadir") if name in ds_cyclemean]
        # if ambiguous:
        #     ds_cyclemean = ds_cyclemean.drop_vars(ambiguous, errors="ignore")
        if coarsen:
            print("Cyclemean dataset shapes after coarsening: ds_cyclemean:", ds_cyclemean.sizes)

        # Align datasets along common coordinates before subtraction
        ds, ds_cyclemean = xr.align(ds, ds_cyclemean, join="inner", exclude=["num_lines", "num_pixels"])

        print(f"Aligned datasets along swath_num dimension. Number of ds swaths after alignment: {len(ds['swath_num'])}, number of cycle-mean swaths after alignment: {len(ds_cyclemean['swath_num'])}")

        ds["simulated_true_ssh_karin"] = ds["simulated_true_ssh_karin"] - ds_cyclemean["simulated_true_ssh_karin"]


        filename += "_timemean_removed_cycle_mean"
        print("Removed timemean from SSH with cycle-mean files.")

    # Filter by latitude and longitude
    lat_indexer = ((ds["latitude"] >= lat_south) & (ds["latitude"] <= lat_north)).compute()
    lon_indexer = ((ds["longitude"] <= lon_east) & (ds["longitude"] >= lon_west)).compute()
    ds = ds.where(lat_indexer & lon_indexer, drop=True)
    
    print(f"Dataset shape after filtering: {ds.sizes}")

    # Determine if satellite swath leans toward the northeast or northwest
    correlation = xr.corr(ds["latitude"], ds["longitude"], dim=["num_lines", "num_pixels"])
    ds["direction"] = xr.where(correlation >= 0, "northeast", "southwest")

    # Preserve per-swath metadata through processing steps
    direction_by_swath = ds["direction"].copy()
    time_by_swath = ds["time"].copy() if "time" in ds else None

    swot_dx = 2000 * coarsen if coarsen else 2000
    # Calculate geostrophic velocities
    omega = 7.2921e-5
    g = 9.81 
    ds['f'] = 2 * omega * np.sin(ds.latitude * np.pi/180)

    if cleaned:

        ds['swot_total_error'] = (
            ds.simulated_error_phase + 
            ds.simulated_error_roll + 
            ds.simulated_error_timing + 
            ds.simulated_error_baseline_dilation + 
            # ds_cut.simulated_error_karin + 
            # ds_cut.simulated_error_troposphere + 
            ds.simulated_error_orbital
            )

        ds['ssh_karin_all_error_removed'] = ds.ssh_karin - ds['swot_total_error']

        ds['ssh_karin_all_error_removed_ddnum_lines'] = ds['ssh_karin_all_error_removed'].differentiate('num_lines') / swot_dx
        ds['ssh_karin_all_error_removed_ddnum_pixels'] = ds['ssh_karin_all_error_removed'].differentiate('num_pixels') / swot_dx
        ds['u_geo_all_error_removed'] = - (g / ds['f']) * ds['ssh_karin_all_error_removed_ddnum_lines']
        ds['v_geo_all_error_removed'] = (g / ds['f']) * ds['ssh_karin_all_error_removed_ddnum_pixels']

        uvar = 'u_geo_all_error_removed'
        vvar = 'v_geo_all_error_removed'

        filename += "_cleaned"
    else:

        ds['simulated_true_ssh_karin_ddnum_lines'] = ds['simulated_true_ssh_karin'].differentiate('num_lines') / swot_dx
        ds['simulated_true_ssh_karin_ddnum_pixels'] = ds['simulated_true_ssh_karin'].differentiate('num_pixels') / swot_dx
        ds['u_geo'] = - (g / ds['f']) * ds['simulated_true_ssh_karin_ddnum_lines']
        ds['v_geo'] = (g / ds['f']) * ds['simulated_true_ssh_karin_ddnum_pixels']

        # THIS IS A BAD METHOD, USE NEWER SORT METHOD, DONT JUST FLIP VELOCITIES
        # if flip_swath:
        #     ds['u_geo'] = xr.where(ds['direction'] == 'northeast', ds['u_geo'], -ds['u_geo'])
        #     ds['v_geo'] = xr.where(ds['direction'] == 'northeast', ds['v_geo'], -ds['v_geo'])
        #     filename += "_flipped_swath"

        uvar = 'u_geo'
        vvar = 'v_geo'
        print("Calculated geostrophic velocities")

    if timemean_removal_method == "linear_fit":

        ds[uvar] = ds[uvar] - compute_large_scale_mean(ds, var=uvar, xvar='num_pixels', yvar='num_lines', stack_dim='swath_num')
        ds[vvar] = ds[vvar] - compute_large_scale_mean(ds, var=vvar, xvar='num_pixels', yvar='num_lines', stack_dim='swath_num')
        filename += "_timemean_removed_linear_fit"
        print("Removed time mean from geostrophic velocities using linear fit")

    if timemean_removal_method == "bulk_mean":

        ds[uvar] = ds[uvar] - ds[uvar].mean(dim="swath_num")
        ds[vvar] = ds[vvar] - ds[vvar].mean(dim="swath_num")
        filename += "_timemean_removed"
        print("Removed time mean from geostrophic velocities")



    if len(ds['num_pixels']) < len(ds['num_lines']):
        larger_dim = 'num_lines'
    else:
        larger_dim = 'num_pixels'

    if ASF:

        if scalar:
            qvar_calc = 'CALC'
        else:
            qvar_calc = None

        ds = calc_advection(ds, qvar=qvar_calc, uvar=uvar, vvar=vvar, xvar='num_pixels', yvar='num_lines', dx=swot_dx, dy=swot_dx)
        ds["direction"] = direction_by_swath
        if time_by_swath is not None:
            ds["time"] = time_by_swath

        print("Calculated advection")

        # Calculate ASFs
        ds_list = [
            compute_asf(
                ds,
                shift_func="shift",
                uvar=uvar,
                vvar=vvar,
                qvar=scalar,
                xvar='num_pixels',
                yvar='num_lines',
                shiftnum=sh,
                full=True,
                just_mean=True,
            )
            for sh in range(len(ds[larger_dim]) // reduce_xd_num)
        ]
        print("Calculated ASFs")

        # Set up ASF DataArray objects

        if scalar:

            ds_list_asf = [ds[0] for ds in ds_list]
            ds_list_asfq = [ds[1] for ds in ds_list]

            ds_list_computed_asf = dask.compute(*ds_list_asf)
            
            print("Computed ASFs")

            ds_asf_computed = format_datasets(
                ds_list_computed_asf, swot_dx, swot_dx, sf_type="asf", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )

            ds_list_computed_asfq = dask.compute(*ds_list_asfq)
            print("Computed ASFQs")

            ds_asfq_computed = format_datasets(
                ds_list_computed_asfq, swot_dx, swot_dx, sf_type="asfq", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )
            ds_computed = xr.merge([ds_asf_computed, ds_asfq_computed])

            filename += f"_{scalar}"

            print("Formatted ASFs with scalar")

        else:

            ds_list_computed = dask.compute(*ds_list)
            print("Computed ASFs")

            ds_computed = format_datasets(
                ds_list_computed, swot_dx, swot_dx, sf_type="asf", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )
            print("Formatted ASFs")
    

    if LLL:
        # Calculate LLL SFs
        ds_list_LLL = [
            compute_LLL(
                ds,
                shift_func="shift",
                uvar=uvar,
                vvar=vvar,
                qvar=scalar,
                xvar='num_pixels',
                yvar='num_lines',
                shiftnum=sh,
                full=True,
                just_mean=True,
            )
            for sh in range(len(ds[larger_dim]) // reduce_xd_num)
        ]

        # Compute LLL SFs
        ds_list_LLL_computed = dask.compute(*ds_list_LLL)
        print("Computed LLL SFs")

        print("Shape of ds_list_LLL_computed:", np.shape(ds_list_LLL_computed))
        print("Length of ds_list_LLL_computed:", len(ds_list_LLL_computed))


        # Set up LLL DataArray objects
        if scalar:
            ds_list_computed_LLL = [ds[0] for ds in ds_list_LLL_computed]
            ds_list_computed_Lqq = [ds[1] for ds in ds_list_LLL_computed]

            ds_computed_LLL = format_datasets(
                ds_list_computed_LLL, swot_dx, swot_dx, sf_type="LLL", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )

            ds_computed_Lqq = format_datasets(
                ds_list_computed_Lqq, swot_dx, swot_dx, sf_type="Lqq", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )

            ds_LLL_computed = xr.merge([ds_computed_LLL, ds_computed_Lqq])
            filename += f"_L{scalar}{scalar}"
            print("Formatted LLL SFs with scalar")
        else:
            ds_LLL_computed = format_datasets(
                ds_list_LLL_computed, swot_dx, swot_dx, sf_type="LLL", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )
            filename += "_LLL"
            print("Formatted LLL SFs without scalar")

    if Bessels:
        # Calculate fluxes from Bessel functions
        ds_bessels = get_bessels(
            ds_computed,
            var="asf_left",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_down",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upleft",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upright",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )

        if taper_SF:
            taper_var = "tapered"
        else:
            taper_var = ""

        # Compute the delayed Bessel object before assigning to it
        ds_bessels = dask.compute(ds_bessels)[0]

        ds_bessels[f"EFlux_Bessel_ASF_mean_{taper_var}"] = (
            sum(ds_computed[var] for var in ds_computed.data_vars if "Bessel" in var)
            / 4
        )

        # ds_bessels_timemean = ds_bessels.mean(dim="time")

        print("Computed Bessel functions")

        filename += "_Bessels"

        if taper_SF:
            filename += "_tapered"


    if CG: 
        sizes = np.logspace(0, 2.5, 50)
        delayed_coarse_grain = dask.delayed(coarse_grain)
        print("Delayed coarse graining")

        ds_cg_list = [
            delayed_coarse_grain(
                ds.isel(swath_num=i),
                filter="astropy",
                uvar=uvar,
                vvar=vvar,
                xvar='num_pixels',
                yvar='num_lines',
                qvar=scalar,
                dx=swot_dx,
                dy=swot_dx,
                kwargs={"kernel": Tophat2DKernel(s)},
            )
            for s in sizes
            for i in range(len(ds['swath_num']))
        ]

        print("Set up coarse graining")

        ds_cg_list_computed = dask.compute(*ds_cg_list)
        print("Computed coarse graining")


        if scalar is None:

            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), len(ds['swath_num'])
            )   

            ds_cg_list_ = [
                xr.DataArray(reshaped_ds_cg_list[i, :], dims=["swath_num"])
                for i in range(len(sizes))
            ]
            ds_cg_computed = xr.concat(ds_cg_list_, dim="L")
            ds_cg_computed = xr.Dataset({"EFlux_CG": ds_cg_computed})
            ds_cg_computed["L"] = sizes 

            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * swot_dx)
            )

        else:
            
            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), len(ds['swath_num']), 2
            )   

            ds_cg_list_EFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 0], dims=["swath_num"])
                for i in range(len(sizes))
            ]
            ds_cg_list_QFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 1], dims=["swath_num"])
                for i in range(len(sizes))
            ]

            ds_cg_computed_EFlux = xr.concat(ds_cg_list_EFlux, dim="L")
            ds_cg_computed_QFlux = xr.concat(ds_cg_list_QFlux, dim="L")

            ds_cg_computed = xr.Dataset({
                "EFlux_CG": ds_cg_computed_EFlux, 
                "QFlux_CG": ds_cg_computed_QFlux
                })
            ds_cg_computed["L"] = sizes 

            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * swot_dx)
            )

        filename += "_CG"


    # Save to netCDF
    # Build list of datasets to merge
    datasets_to_merge = []

    if ASF:
        datasets_to_merge.append(ds_computed)
    if Bessels:
        datasets_to_merge.append(ds_bessels)
    if CG:
        datasets_to_merge.append(ds_cg_computed)
    # if FFT:
    #     datasets_to_merge.append(ds_fft)
    if LLL:
        datasets_to_merge.append(ds_LLL_computed)
    # if LL:
    #     datasets_to_merge.append(ds_LL_computed)

    # Merge all datasets at once
    ds_all = xr.merge(datasets_to_merge) if len(datasets_to_merge) > 1 else datasets_to_merge[0]

    ds_all["direction"] = direction_by_swath
    if time_by_swath is not None:
        ds_all["time"] = time_by_swath

    assert "direction" in ds_all.variables, "Missing 'direction' in output dataset before NetCDF write"
    assert "time" in ds_all.variables, "Missing 'time' in output dataset before NetCDF write"


    datetime_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    ds_all.to_netcdf(f"{outpath}/{region_name}/{datetime_str}_{filename}.nc")
    print(f"Saved to {outpath}/{region_name}/{datetime_str}_{filename}.nc")

    return None
    
def compute_and_save_SWOT_L3(
    filename="SWOT_L3",
    filepath="/Volumes/Promise Disk/data/validated_swot/SWOT_L3_LR_SSH/*.nc",
    mean_filepath=None,
    region_name="acc",
    cycle_num="001",
    lat_north=60,
    lat_south=-60,
    lon_west=-180,
    lon_east=180,
    outpath="data/SWOT_L3/",
    reduce_xd_num=1,
    timemean_removal_method=None,
    CG=False,
    LLL=False,
    Bessels=False,
    taper_SF=False,
    scalar=None,
    coarsen=None,
    filter_vels=False,
    flip_swath=False,    
):
    # Load data

    filename += f"_{region_name}_cycle_{cycle_num}"

    print(f"Loading data from {filepath}")

    file_paths = glob.glob(filepath)
    print("Defined file paths with glob")

    uvar = "ugos_filtered" if filter_vels else "ugos_unfiltered"
    vvar = "vgos_filtered" if filter_vels else "vgos_unfiltered"

    print(f"Number of datasets: {len(file_paths)}")
    ds = load_and_pad_swath_datasets(
        file_paths=file_paths,
        variables=["time", uvar, vvar],
        line_dim="num_lines",
        concat_dim="swath_num",
        pass_attr="GET",
        coarsen=coarsen
    )

    if coarsen:
        new_resolution_string = f"{(48 / coarsen):.1f}"
        filename += f"_coarsen_to_1div{new_resolution_string}deg"
        print(f"Coarsened data by a factor of {coarsen} to approximately 1/{new_resolution_string} degree resolution")


    if timemean_removal_method == "cycle_mean":
        print("Loading cycle-meaned .nc files and padding them to the maximum number of lines.")
        mean_file_paths = glob.glob(mean_filepath)
        print(f"Number of cycle-mean datasets: {len(mean_file_paths)}")

        ds_cyclemean = load_and_pad_swath_datasets(
            file_paths=mean_file_paths,
            variables=[
                "time",
                uvar,
                vvar
            ],
            line_dim="num_lines",
            concat_dim="swath_num",
            pass_attr="pass_number",
            coarsen=coarsen,
        )

        # ds and ds_cyclemean may have different swath_nums so we need to align them along the swath_num dimension before subtraction, and we want to keep only the swath_nums that are present in both datasets
        ds, ds_cyclemean = xr.align(ds, ds_cyclemean, join="inner", exclude=["num_lines", "num_pixels"])

        print(f"Aligned datasets along swath_num dimension. Number of ds swaths after alignment: {len(ds['swath_num'])}, number of cycle-mean swaths after alignment: {len(ds_cyclemean['swath_num'])}")

        ds[uvar] = ds[uvar] - ds_cyclemean[uvar]
        ds[vvar] = ds[vvar] - ds_cyclemean[vvar]

        filename += "_timemean_removed_cycle_mean"
        print("Removed timemean from geostrophic velocities with cycle-mean files.")


    lat_indexer = ((ds["latitude"] >= lat_south) &
                   (ds["latitude"] <= lat_north)).compute()
    lon_indexer = ((ds["longitude"] <= lon_east) &
                   (ds["longitude"] >= lon_west)).compute()
    
    ds = ds.where(lat_indexer & lon_indexer, drop=True)
    print("Filtered datasets by region")

    # Determine if satellite swath leans toward the northeast or northwest
    
    correlation = xr.corr(ds["latitude"], ds["longitude"], dim=['num_lines', 'num_pixels'])

    # If correlation is positive, swath leans northeast; if negative, leans northwest, but the line above calculates correlation for each swath_num, so each swath_num can have different leaning direction
    # We will determine leaning direction for each swath_num individually and set it as a variable in the dataset so it should be the same length as swath_num dimension
    ds['direction'] = xr.where(correlation >= 0, 'northeast', 'southwest')

    # If swath leans southwest, flip it to lean northeast so that all swaths have the same orientation for later calculations. Iterate through swath_num dimension and flip those that lean southwest
    if flip_swath:
        reverse = ds["direction"] == "southwest"

        n_lines = ds.sizes["num_lines"]
        n_pixels = ds.sizes["num_pixels"]

        line_idx = xr.DataArray(np.arange(n_lines), dims="num_lines")
        pix_idx  = xr.DataArray(np.arange(n_pixels), dims="num_pixels")

        line_idx = xr.where(reverse, line_idx[::-1], line_idx)
        pix_idx  = xr.where(reverse, pix_idx[::-1], pix_idx)

        ds = ds.isel(
            num_lines=line_idx,
            num_pixels=pix_idx,
        )
        ds["direction"] = xr.where(reverse, "northeast", ds["direction"])
        filename += "_flipped_swath"
        print("Flipped swaths to have consistent orientation")

    # Preserve per-swath metadata through processing steps
    direction_by_swath = ds["direction"].copy()
    time_by_swath = ds["time"].copy() if "time" in ds else None

    swot_dx = 2000 * coarsen if coarsen else 2000

    # Calculate advection
    if scalar:
        qvar_calc = 'CALC'
    else:
        qvar_calc = None

    ds = calc_advection(ds, qvar=qvar_calc, uvar=uvar, vvar=vvar, xvar='num_pixels', yvar='num_lines', dx=swot_dx, dy=swot_dx)
    ds["direction"] = direction_by_swath
    if time_by_swath is not None:
        ds["time"] = time_by_swath
    print("Calculated advection")

    if len(ds['num_pixels']) < len(ds['num_lines']):
        larger_dim = 'num_lines'
    else:
        larger_dim = 'num_pixels'

    # Calculate ASFs
    ds_list = [
        compute_asf(
            ds,
            shift_func="shift",
            uvar=uvar,
            vvar=vvar,
            qvar=scalar,
            xvar='num_pixels',
            yvar='num_lines',
            shiftnum=sh,
            full=True,
            just_mean=True,
        )
        for sh in range(len(ds[larger_dim]) // reduce_xd_num)
    ]

    # Compute ASFs
    ds_list_computed = dask.compute(*ds_list)
    print("Computed ASFs")
    # Set up ASF DataArray objects

    if scalar:

        ds_list_computed_asf = [ds[0] for ds in ds_list_computed]
        ds_list_computed_asfq = [ds[1] for ds in ds_list_computed]
        ds_computed = format_datasets(
            ds_list_computed_asf, swot_dx, swot_dx, sf_type="asf", xvar='num_pixels', yvar='num_lines', dim='swath_num'
    )

        ds_computed_asfq = format_datasets(
            ds_list_computed_asfq, swot_dx, swot_dx, sf_type="asfq", xvar='num_pixels', yvar='num_lines', dim='swath_num'
        )
        ds_computed = xr.merge([ds_computed, ds_computed_asfq])
        filename += f"_{scalar}"
        print("Formatted ASFs with scalar")

    else:
        
        ds_computed = format_datasets(
            ds_list_computed, swot_dx, swot_dx, sf_type="asf", xvar='num_pixels', yvar='num_lines', dim='swath_num'
        )
        print("Formatted ASFs without scalar")
    
    if LLL:
        # Calculate LLL SFs
        ds_list_LLL = [
            compute_LLL(
                ds,
                shift_func="shift",
                uvar=uvar,
                vvar=vvar,
                qvar=scalar,
                xvar='num_pixels',
                yvar='num_lines',
                shiftnum=sh,
                full=True,
                just_mean=True,
            )
            for sh in range(len(ds[larger_dim]) // reduce_xd_num)
        ]

        # Compute LLL SFs
        ds_list_LLL_computed = dask.compute(*ds_list_LLL)
        print("Computed LLL SFs")

        print("Shape of ds_list_LLL_computed:", np.shape(ds_list_LLL_computed))
        print("Length of ds_list_LLL_computed:", len(ds_list_LLL_computed))


        # Set up LLL DataArray objects
        if scalar:
            ds_list_computed_LLL = [ds[0] for ds in ds_list_LLL_computed]
            ds_list_computed_Lqq = [ds[1] for ds in ds_list_LLL_computed]

            ds_computed_LLL = format_datasets(
                ds_list_computed_LLL, swot_dx, swot_dx, sf_type="LLL", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )

            ds_computed_Lqq = format_datasets(
                ds_list_computed_Lqq, swot_dx, swot_dx, sf_type="Lqq", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )

            ds_LLL_computed = xr.merge([ds_computed_LLL, ds_computed_Lqq])
            filename += f"_L{scalar}{scalar}"
            print("Formatted LLL SFs with scalar")
        else:
            ds_LLL_computed = format_datasets(
                ds_list_LLL_computed, swot_dx, swot_dx, sf_type="LLL", xvar='num_pixels', yvar='num_lines', dim='swath_num'
            )
            filename += "_LLL"
            print("Formatted LLL SFs without scalar")


    if Bessels:
        # Calculate fluxes from Bessel functions
        ds_bessels = get_bessels(
            ds_computed,
            var="asf_left",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_down",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upleft",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upright",
            xvar="num_pixels",
            yvar="num_lines",
            taper_SF=taper_SF,
            periodic=False,
            dx=swot_dx,
            dy=swot_dx,
            N=len(ds_computed["shiftnum"]),
        )

        if taper_SF:
            taper_var = "tapered"
        else:
            taper_var = ""

        # Compute the delayed Bessel object before assigning to it
        ds_bessels = dask.compute(ds_bessels)[0]

        ds_bessels[f"EFlux_Bessel_ASF_mean_{taper_var}"] = (
            sum(ds_computed[var] for var in ds_computed.data_vars if "Bessel" in var)
            / 4
        )

        # ds_bessels_timemean = ds_bessels.mean(dim="time")

        print("Computed Bessel functions")

        filename += "_Bessels"

        if taper_SF:
            filename += "_tapered"

    if CG: 
        sizes = np.logspace(0, 2.5, 50)
        delayed_coarse_grain = dask.delayed(coarse_grain)                                       
        print("Delayed coarse graining")

        ds_cg_list = [
            delayed_coarse_grain(
                ds.isel(swath_num=i),
                filter="astropy",
                uvar=uvar,
                vvar=vvar,
                xvar='num_pixels',
                yvar='num_lines',
                qvar=scalar,
                dx=swot_dx,
                dy=swot_dx,
                kwargs={"kernel": Tophat2DKernel(s)},
            )
            for s in sizes
            for i in range(len(ds['swath_num']))
        ]

        print("Set up coarse graining")

        ds_cg_list_computed = dask.compute(*ds_cg_list)
        print("Computed coarse graining")            

        if scalar is None:
            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                        len(sizes), len(ds['swath_num'])
                    )

            ds_cg_list_ = [
                xr.DataArray(reshaped_ds_cg_list[i, :], dims=["swath_num"])
                for i in range(len(sizes))
            ]
            ds_cg_computed = xr.concat(ds_cg_list_, dim="L")
            ds_cg_computed = xr.Dataset({"EFlux_CG": ds_cg_computed})
            ds_cg_computed["L"] = sizes 

            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * swot_dx)
            )

        else:
            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), len(ds['swath_num']), 2
            )
            
            ds_cg_list_EFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 0], dims=["swath_num"])
                for i in range(len(sizes))
            ]
            ds_cg_list_QFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 1], dims=["swath_num"])
                for i in range(len(sizes))
            ]

            ds_cg_computed_EFlux = xr.concat(ds_cg_list_EFlux, dim="L")
            ds_cg_computed_QFlux = xr.concat(ds_cg_list_QFlux, dim="L")
            ds_cg_computed = xr.Dataset({
                "EFlux_CG": ds_cg_computed_EFlux,
                "QFlux_CG": ds_cg_computed_QFlux,
            })
            ds_cg_computed["L"] = sizes
            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * swot_dx)
            )

        
        print("Coarse graining formatted")
        filename += "_CG"

    # Save to netCDF and use xr.merge to combine datasets if they exist
    datasets_to_merge = [ds_computed]
    if LLL:
        datasets_to_merge.append(ds_LLL_computed)
    if CG:
        datasets_to_merge.append(ds_cg_computed)
    if Bessels:
        datasets_to_merge.append(ds_bessels)

    ds_all = xr.merge(datasets_to_merge) if len(datasets_to_merge) > 1 else datasets_to_merge[0]

    # Add direction and time variable for each swath_num based on original ds
    ds_all["direction"] = direction_by_swath
    if time_by_swath is not None:
        ds_all["time"] = time_by_swath

    assert "direction" in ds_all.variables, "Missing 'direction' in output dataset before NetCDF write"
    assert "time" in ds_all.variables, "Missing 'time' in output dataset before NetCDF write"

    filename += "_filtered_vels" if filter_vels else "_unfiltered_vels"

    datetime_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    ds_all.to_netcdf(f"{outpath}/{region_name}/{datetime_str}_{filename}.nc")
    print(f"Saved to {outpath}/{region_name}/{datetime_str}_{filename}.nc")

def compute_and_save_NEMO(
    filename="NEMO",
    inputpath="/Volumes/Promise Disk/data/NEMO/2019-2021_data/cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_180.00W-179.92E_80.00S-90.00N_0.49m_2019-06-30-2021-06-30.nc",
    region_name="acc",
    outpath="data",
    lat_north=60,
    lat_south=-60,
    lon_west=-180,
    lon_east=180,
    reduce_xd_num=1,
    CG=False,
    LLL=False,
    Bessels=False,
    scalar=None,
):
    # Load data
    filepath = inputpath

    ds = xr.open_dataset(filepath).squeeze(dim='depth', drop=True)

    dx = 0.083 # 0.083 degrees in longitude
    dy = 0.083 # 0.083 degrees in latitude

    dx = dx * 111320  # Convert degrees to meters (approximate)
    dy = dy * 111320  # Convert degrees to meters (approximate)

    # Filter by region
    lat_indexer = ((ds["latitude"] >= lat_south) &
                   (ds["latitude"] <= lat_north)).compute()
    lon_indexer = ((ds["longitude"] <= lon_east) &
                   (ds["longitude"] >= lon_west)).compute()
    ds = ds.where(lat_indexer & lon_indexer, drop=True)
    print(f"Filtered datasets by region. Dataset shape: {ds.sizes}")

    if scalar:
        qvar_calc = 'CALC'
    else:
        qvar_calc = None


    # Calculate advection
    ds = calc_advection(
        ds,
        qvar=qvar_calc,
        uvar="uo",
        vvar="vo",
        xvar="longitude",
        yvar="latitude",
        dx=dx,
        dy=dy,
    )

    if len(ds["longitude"]) < len(ds["latitude"]):
        larger_dim = "latitude"
    else:
        larger_dim = "longitude"

    # Calculate ASFs
    ds_list = [
        compute_asf(
            ds,
            shift_func="shift",
            uvar="uo",
            vvar="vo",
            qvar=scalar,
            xvar="longitude",
            yvar="latitude",
            shiftnum=sh,
            full=True,
            just_mean=True,
        )
        for sh in range(len(ds[larger_dim]) // reduce_xd_num)
    ]

    # Set up ASF DataArray objects

    if scalar:

        ds_list_asf = [ds[0] for ds in ds_list]
        ds_list_asfq = [ds[1] for ds in ds_list]

        ds_list_computed_asf = dask.compute(*ds_list_asf)
        print(f"Computed {len(ds_list_asf)} ASFs")

        ds_asf_computed = format_datasets(
            ds_list_computed_asf, dx, dy, sf_type="asf", xvar="longitude", yvar="latitude", dim="time"
        )

        ds_list_computed_asfq = dask.compute(*ds_list_asfq)
        print(f"Computed {len(ds_list_asfq)} ASFQs")

        ds_asfq_computed = format_datasets(
            ds_list_computed_asfq, dx, dy, sf_type="asfq", xvar="longitude", yvar="latitude", dim="time"
        )
        ds_computed = xr.merge([ds_asf_computed, ds_asfq_computed])
        filename += f"_{scalar}"
        print("Formatted ASFs with scalar")

    else:

        ds_list_computed = dask.compute(*ds_list)
        print("Computed ASFs")

        ds_computed = format_datasets(
            ds_list_computed, dx, dy, sf_type="asf", xvar="longitude", yvar="latitude", dim="time"
        )
        print("Formatted ASFs")

    # Coarse grain if requested
    if CG:
        sizes = np.logspace(0, 2.5, 50)
        delayed_coarse_grain = dask.delayed(coarse_grain)
        print("Delayed coarse graining")

        ds_cg_list = [
            delayed_coarse_grain(
                ds.isel(time=i),
                filter="astropy",
                uvar="uo",
                vvar="vo",
                xvar="longitude",
                yvar="latitude",
                qvar=scalar,
                dx=dx,
                dy=dy,
                kwargs={"kernel": Tophat2DKernel(s)},
            )
            for s in sizes
            for i in range(len(ds["time"]))
        ]

        print("Set up coarse graining")

        ds_cg_list_computed = dask.compute(*ds_cg_list)
        print("Computed coarse graining")

        if scalar is None:

            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), len(ds["time"])
            )

            ds_cg_list_ = [
                xr.DataArray(reshaped_ds_cg_list[i, :], dims=["time"])
                for i in range(len(sizes))
            ]
            ds_cg_computed = xr.concat(ds_cg_list_, dim="L")
            ds_cg_computed = xr.Dataset({"EFlux_CG": ds_cg_computed})
            ds_cg_computed["L"] = sizes

            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * np.mean([dx, dy]))
            )

        else:
            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), len(ds["time"]), 2
            )

            ds_cg_list_EFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 0], dims=["time"])
                for i in range(len(sizes))
            ]
            ds_cg_list_QFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 1], dims=["time"])
                for i in range(len(sizes))
            ]

            ds_cg_computed_EFlux = xr.concat(ds_cg_list_EFlux, dim="L")
            ds_cg_computed_QFlux = xr.concat(ds_cg_list_QFlux, dim="L")

            ds_cg_computed = xr.Dataset({
                "EFlux_CG": ds_cg_computed_EFlux,
                "QFlux_CG": ds_cg_computed_QFlux
            })
            ds_cg_computed["L"] = sizes

            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * np.mean([dx, dy]))
            )

        filename += "_CG"

    # Save to netCDF
    if CG:
        ds_all = xr.merge([ds_computed, ds_cg_computed])
    else:
        ds_all = ds_computed

    datetime_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    ds_all.to_netcdf(f"{outpath}/NEMO/{region_name}/{datetime_str}_{filename}.nc")
    print(f"Saved to {outpath}/NEMO/{region_name}/{datetime_str}_{filename}.nc")
    return None

def compute_and_save_SWOT_L4(
    ds_og,
    ds_timemean,
    filename="SWOT_L4",
    lat_north=60,
    lat_south=-60,
    lon_west=-180,
    lon_east=180,
    outpath="data",
    CG=False,
    LLL=False,
    LL=False,
    Bessels=False,
    FFT=False,
    scalar=None,
    batch_reduce_num=1,
    taper_SF=False,
):

    ds = ds_og.copy()

    # SWOT L4 spatial resolution (1/10 degree)
    dx = 0.1  # degrees in longitude
    dy = 0.1  # degrees in latitude

    dx = dx * 111320  # Convert degrees to meters (approximate)
    dy = dy * 111320  # Convert degrees to meters (approximate)

    # Filter by region
    lat_indexer = ((ds["latitude"] >= lat_south) &
                   (ds["latitude"] <= lat_north)).compute()
    lon_indexer = ((ds["longitude"] <= lon_east) &
                   (ds["longitude"] >= lon_west)).compute()
    ds = ds.where(lat_indexer & lon_indexer, drop=True)
    print(f"Filtered dataset by region. Dataset shape: {ds.sizes}")

    ds["ugos"] = ds["ugos"] - ds_timemean['ugos']
    ds["vgos"] = ds["vgos"] - ds_timemean['vgos']
    filename += "_timemean_removed"
    print("Demeaned velocities")

    if scalar:
        qvar_calc = 'CALC'
    else:
        qvar_calc = None

    # Calculate advection
    ds = calc_advection(
        ds,
        qvar=qvar_calc,
        uvar="ugos",
        vvar="vgos",
        xvar="longitude",
        yvar="latitude",
        dx=dx,
        dy=dy,
    )
    print("Calculated advection")

    if ds.sizes["longitude"] < ds.sizes["latitude"]:
        larger_dim = "latitude"
    else:
        larger_dim = "longitude"
    print(f"Larger dim length: {ds.sizes[larger_dim]}")

    ds_list = [
        compute_asf(
            ds,
            shift_func="shift",
            uvar="ugos",
            vvar="vgos",
            qvar=scalar,
            xvar="longitude",
            yvar="latitude",
            shiftnum=sh,
            full=True,
            just_mean=True,
        )
        for sh in range(ds.sizes[larger_dim])
    ]
    print("Set up ASF computations")

    # Set up ASF DataArray objects
    # SWOT_L4 has no time dimension, so just merge the computed ASFs directly

    if scalar:

        # Compute ASFs
        ds_list_asf = [ds[0] for ds in ds_list]
        ds_list_asfq = [ds[1] for ds in ds_list]

        # Try batch computing to reduce memory usage
        # ds_list_computed_asf = dask.compute(*ds_list_asf)
        batch_size = len(ds_list_asf) // batch_reduce_num  # Adjust batch size as needed 
        ds_list_computed_asf = []
        
        for i in range(0, len(ds_list_asf), batch_size):
            batch = ds_list_asf[i:i+batch_size]
            computed_batch = dask.compute(*batch)
            ds_list_computed_asf.extend(computed_batch)
            print(f"Computed ASF batch {i//batch_size + 1}/{(len(ds_list_asf)-1)//batch_size + 1}")
            
            import gc
            gc.collect()

        print("Computed ASFs")
        
        ds_asf_computed = format_datasets(
            ds_list_computed_asf, dx, dy, sf_type="asf", xvar="longitude", yvar="latitude", dim="time"
        )

        # ds_list_computed_asfq = dask.compute(*ds_list_asfq)
        ds_list_computed_asfq = []

        for i in range(0, len(ds_list_asfq), batch_size):
            batch = ds_list_asfq[i:i+batch_size]
            computed_batch = dask.compute(*batch)
            ds_list_computed_asfq.extend(computed_batch)
            print(f"Computed ASFQ batch {i//batch_size + 1}/{(len(ds_list_asfq)-1)//batch_size + 1}")
            
            import gc
            gc.collect()

        print("Computed ASFQ")
        ds_asfq_computed = format_datasets(
            ds_list_computed_asfq, dx, dy, sf_type="asfq", xvar="longitude", yvar="latitude", dim="time"
        )
        ds_computed = xr.merge([ds_asf_computed, ds_asfq_computed])
        print("Formatted ASFs with scalar")

    else:

        ds_list_computed = dask.compute(*ds_list)
        print("Computed ASFs")

        ds_computed = format_datasets(
            ds_list_computed, dx, dy, sf_type="asf", xvar="longitude", yvar="latitude", dim="time"
        )
        print("Formatted ASFs")


    if LLL:
        # Calculate LLL SFs
        ds_list_LLL = [
            compute_LLL(
                ds,
                shift_func="shift",
                uvar="ugos",
                vvar="vgos",
                xvar="longitude",
                yvar="latitude",
                shiftnum=sh,
                full=True,
                just_mean=True,
            )
            for sh in range(ds.sizes[larger_dim])
        ]

        # Compute LLL SFs
        ds_list_LLL_computed = dask.compute(*ds_list_LLL)

        # Set up LLL DataArray objects
        ds_LLL_computed = format_datasets(
            ds_list_LLL_computed, dx, dy, sf_type="LLL", xvar="longitude", yvar="latitude", dim="time"
        )
        print("Computed LLL SFs")
        filename += "_LLL"

    if LL:
        # Calculate LL SFs
        ds_list_LL = [
            compute_LL(
                ds,
                shift_func="shift",
                uvar="ugos",
                vvar="vgos",
                xvar="longitude",
                yvar="latitude",
                shiftnum=sh,
            )
            for sh in range(ds.sizes[larger_dim])
        ]

        # Compute LL SFs
        ds_list_LL_computed = dask.compute(*ds_list_LL)

        # Set up LL DataArray objects
        ds_LL_computed = format_datasets(
            ds_list_LL_computed, dx, dy, sf_type="LL", xvar="longitude", yvar="latitude", dim="time"
        )
        print("Computed LL SFs")
        filename += "_LL"

    if Bessels:
        # Calculate fluxes from Bessel functions
        ds_bessels = get_bessels(
            ds_computed,
            var="asf_left",
            xvar="longitude",
            yvar="latitude",
            taper_SF=taper_SF,
            periodic=False,
            dx=dx,
            dy=dy,
            N=ds_computed.sizes["shiftnum"],
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_down",
            xvar="longitude",
            yvar="latitude",
            taper_SF=taper_SF,
            periodic=False,
            dx=dx,
            dy=dy,
            N=ds_bessels.sizes["shiftnum"],
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upleft",
            xvar="longitude",
            yvar="latitude",
            taper_SF=taper_SF,
            periodic=False,
            dx=dx,
            dy=dy,
            N=ds_bessels.sizes["shiftnum"],
        )
        ds_bessels = get_bessels(
            ds_bessels,
            var="asf_diag_upright",
            xvar="longitude",
            yvar="latitude",
            taper_SF=taper_SF,
            periodic=False,
            dx=dx,
            dy=dy,
            N=ds_bessels.sizes["shiftnum"],
        )

        if taper_SF:
            taper_var = "tapered"
        else:
            taper_var = ""

        # Compute the delayed Bessel object before assigning to it
        ds_bessels = dask.compute(ds_bessels)[0]

        ds_bessels[f"EFlux_Bessel_ASF_mean_{taper_var}"] = (
            sum(ds_computed[var] for var in ds_computed.data_vars if "Bessel" in var)
            / 4
        )

        # ds_bessels_timemean = ds_bessels.mean(dim="time")

        print("Computed Bessel functions")

        filename += "_Bessels"

        if taper_SF:
            filename += "_tapered"

    if FFT:
        # delayed_compute_fourier_flux = dask.delayed(compute_fourier_flux)
        # print("Delayed Fourier flux computation")

        ds_fft_list = [
            compute_fourier_flux(
                ds.isel(time=i),
                xvar="longitude",
                yvar="latitude",
                uvar="ugos",
                vvar="vgos",
                dx=dx,
                dy=dy,
            )
            for i in range(ds.sizes["time"])
        ]

        print("Set up Fourier flux computation")

        ds_fft_list_computed = dask.compute(*ds_fft_list)

        print("Computed Fourier flux")

        # Extract EFlux_fourier from each computed dataset
        eflux_list = [
            computed_ds["EFlux_fourier"] for computed_ds in ds_fft_list_computed
        ]

        # Concatenate along time dimension
        ds_fft_computed = xr.concat(eflux_list, dim="time")
        ds_fft = xr.Dataset({"EFlux_fourier": ds_fft_computed})

        # Get wavenumber coordinates from first dataset
        ds_fft["wavenumber"] = ds_fft_list_computed[0]["wavenumber"]

        print("Formatted Fourier flux")
        filename += "_FFT"

    if CG:
        # Calculate fluxes by coarse graining
        sizes = np.logspace(0, 2.5, 50)

        # ds = dask.delayed(ds)
        # print('Computed data')

        delayed_coarse_grain = dask.delayed(coarse_grain)
        print("Delayed coarse graining")

        ds_cg_list = [
            delayed_coarse_grain(
                ds.isel(time=i),
                filter="astropy",
                uvar="ugos",
                vvar="vgos",
                xvar="longitude",
                yvar="latitude",
                qvar=scalar,
                dx=dx,
                dy=dy,
                kwargs={"kernel": Tophat2DKernel(s)},
            )
            for s in sizes
            for i in range(ds.sizes["time"])
        ]

        print("Set up coarse graining")

        ds_cg_list_computed = dask.compute(*ds_cg_list)

        print("Computed coarse graining")

        if scalar is None:

            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), ds.sizes["time"]
            )

            ds_cg_list_ = [
                xr.DataArray(reshaped_ds_cg_list[i, :], dims=["time"])
                for i in range(len(sizes))
            ]
            ds_cg_computed = xr.concat(ds_cg_list_, dim="L")
            ds_cg_computed = xr.Dataset({"EFlux_CG": ds_cg_computed})
            ds_cg_computed["L"] = sizes
            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * np.mean([dx, dy]))
            )
        
        else:
            
            reshaped_ds_cg_list = np.asarray(ds_cg_list_computed).reshape(
                len(sizes), ds.sizes["time"], 2
            )

            ds_cg_list_EFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 0], dims=["time"])
                for i in range(len(sizes))
            ]
            ds_cg_list_QFlux = [
                xr.DataArray(reshaped_ds_cg_list[i, :, 1], dims=["time"])
                for i in range(len(sizes))
            ]

            ds_cg_computed_EFlux = xr.concat(ds_cg_list_EFlux, dim="L")
            ds_cg_computed_QFlux = xr.concat(ds_cg_list_QFlux, dim="L")

            ds_cg_computed = xr.Dataset({
                "EFlux_CG": ds_cg_computed_EFlux, 
                "QFlux_CG": ds_cg_computed_QFlux
                })
            ds_cg_computed["L"] = sizes
            ds_cg_computed["K_coarse_grain"] = (
                2 * np.pi / (ds_cg_computed["L"] * np.mean([dx, dy]))
            )

        print("Formatted coarse graining")
        filename += "_CG"

    # Save to netCDF
    # Build list of datasets to merge
    datasets_to_merge = [ds_computed]

    if Bessels:
        datasets_to_merge.append(ds_bessels)
    if CG:
        datasets_to_merge.append(ds_cg_computed)
    if FFT:
        datasets_to_merge.append(ds_fft)
    if LLL:
        datasets_to_merge.append(ds_LLL_computed)
    if LL:
        datasets_to_merge.append(ds_LL_computed)

    # Merge all datasets at once
    ds_all = xr.merge(datasets_to_merge) if len(datasets_to_merge) > 1 else datasets_to_merge[0]

    datetime_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    coord_str = f"{lat_south}_{lat_north}_{lon_west}_{lon_east}"

    ds_all.to_netcdf(f"{outpath}/SWOT_L4/{datetime_str}__{coord_str}__{filename}.nc")
    print(f"Saved to {outpath}/SWOT_L4/{datetime_str}__{coord_str}__{filename}.nc")        

    return None

def generate_snapshots_NEMO(
    filename="NEMO",
    region_name="acc",
    outpath="data",
    lat_north=60,
    lat_south=-60,
    lon_west=-180,
    lon_east=180,
):
    """
    Generate snapshots of NEMO data for a specified region.
    """
    filepath = "/Volumes/Promise Disk/data/NEMO/2019-2021_data/cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_180.00W-179.92E_80.00S-90.00N_0.49m_2019-06-30-2021-06-30.nc"
    
    ds = xr.open_dataset(filepath).squeeze(dim='depth', drop=True)

    # Filter by region
    lat_indexer = ((ds["latitude"] >= lat_south) &
                   (ds["latitude"] <= lat_north)).compute()
    lon_indexer = ((ds["longitude"] <= lon_east) &
                   (ds["longitude"] >= lon_west)).compute()
    ds = ds.where(lat_indexer & lon_indexer, drop=True)
    
    # Save snapshots
    snapshot = ds.isel(time=-1)
    snapshot.to_netcdf(f"{outpath}/NEMO/{region_name}/{filename}_snapshot.nc")
    print(f"Saved snapshot to {outpath}/NEMO/{region_name}/{filename}_snapshot.nc")

    return None

def generate_snapshots_SWOT_L3(
    filename="SWOT_L3",
    region_name="acc",
    cycle_num="001",
    outpath="data",
    lat_north=60,
    lat_south=-60,
    lon_west=-180,
    lon_east=180,
):
    """
    Generate snapshots of SWOT L3 data for a specified region.
    """
    filename += f"_{region_name}_cycle_{cycle_num}"

    filepath = f"/Volumes/Promise Disk/data/validated_swot/SWOT_L3_LR_SSH/SWOT_L3_LR_SSH_2.0/Expert/cycle_{cycle_num}/*.nc"
    print(f"Loading data from {filepath}")

    file_paths = glob.glob(filepath)
    print("Defined file paths with glob")

    ds = load_and_pad_swath_datasets(
        file_paths=file_paths,
        variables=["time", "ugos", "vgos"],
        line_dim="num_lines",
        concat_dim="swath_num",
    )

    lat_indexer = ((ds["latitude"] >= lat_south) &
                   (ds["latitude"] <= lat_north)).compute()
    lon_indexer = ((ds["longitude"] <= lon_east) &
                   (ds["longitude"] >= lon_west)).compute()
    
    ds = ds.where(lat_indexer & lon_indexer, drop=True)
    print("Filtered datasets by region")

    # Save snapshots
    snapshots = ds
    snapshots.to_netcdf(f"{outpath}/SWOT_L3/{region_name}/{filename}_snapshots.nc")
    print(f"Saved snapshot to {outpath}/SWOT_L3/{region_name}/{filename}_snapshots.nc")

    return None