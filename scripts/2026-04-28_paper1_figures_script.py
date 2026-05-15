import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib_inline
import seaborn as sns
from scipy.special import jv
import glob
import os
import re
import traceback
# import xarray_sf_funcs as xsfuncs
import importlib
# importlib.reload(xsfuncs)
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

sns.set_style(style="white")
sns.set_context("notebook")

# plt.rcParams["figure.figsize"] = [4,3]
# plt.rcParams['figure.dpi'] = 100

# matplotlib_inline.backend_inline.set_matplotlib_formats("retina")

plt.rcParams["text.usetex"] =False
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

# use latex fonts for math text in plots
plt.rcParams["mathtext.fontset"] = "stix"

# force ticks to go inward for all plots
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

def _format_date_range(ds, time_dim="time"):

    start = ds[time_dim].min(skipna=True).dt.strftime("%b %d, %Y").item()
    end = ds[time_dim].max(skipna=True).dt.strftime("%b %d, %Y").item()

    # print(f"Formatting date range for dataset with time dimension of shape {ds.time.shape}")
    # print(f"Dataset time range: {start} to {end}")

    return f"{start} – {end}"

# Function to convert wavenumber to separation distance
def wavenumber_to_distance(wavenumber):
    return 1 / wavenumber

# Function to convert separation distance to wavenumber
def distance_to_wavenumber(distance):
    return 1 / distance

# Function to process datasets
def process_dataset(ds, KEflux_key, QFlux_key, quantile_upper, quantile_lower, dim):
    if ds is not None:
        ds = ds.where(
            (ds[KEflux_key] < ds[KEflux_key].quantile(quantile_upper, dim))
            & (ds[KEflux_key] > ds[KEflux_key].quantile(quantile_lower, dim))
            # & (ds[QFlux_key] < ds[QFlux_key].quantile(quantile_upper, dim))
            # & (ds[QFlux_key] > ds[QFlux_key].quantile(quantile_lower, dim))
        )
        ds["EFlux_CG"] = ds["EFlux_CG"].mean(dim="shiftnum")
        ds["K_coarse_grain"] = ds["K_coarse_grain"].mean(dim="shiftnum")

        try:
            ds["QFlux_CG"] = ds["QFlux_CG"].mean(dim="shiftnum")
        except KeyError:
            pass
    return ds

def load_dataset(file_key, params, region, preprocess_func=None):
    try:
        file_path = params[file_key]
        if not file_path.endswith(".nc"):
            raise ValueError(f"Invalid file for {file_key} in region {region}: {file_path}")
        # Keep lazy loading behavior; arrays are read only when downstream
        # operations (e.g., mean/quantile/plot conversion) require values.
        ds = xr.open_dataset(file_path)
        if preprocess_func:
            ds = preprocess_func(ds)
        return ds
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"Skipping {file_key} dataset for region {region} due to error: {e}")
        return None

def preprocess_LLC4320(ds, region):
    # ds_og = xr.open_mfdataset(
    #     f'/Volumes/Promise Disk/data/mitgcm/{region}_data/*.nc',
    #     combine="nested",
    #     concat_dim="time",
    #     coords="minimal",
    #     compat="override",
    # )
    ds['asf_flux_mean'] = -0.5 * ds['asf_mean'] * (1318 / 2000) ** 2
    # ds = ds.assign_coords({'time': ds_og['time']})
    return ds

def preprocess_NEMO(ds, mindate=None, maxdate=None, snapshot_frequency=1, og_filename='/Volumes/Promise Disk/data/NEMO/2019-2021_data/cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_180.00W-179.92E_80.00S-90.00N_0.49m_2019-06-30-2021-06-30.nc'):

    ds_og = xr.open_dataset(og_filename)
    ds = ds.assign_coords({'time': ds_og['time']})
    ds = ds.sel(time=slice(mindate, maxdate, snapshot_frequency))
    ds['asf_flux_mean'] = -0.5 * ds['asf_mean']
    return ds

# Function to plot a dataset
def plot_dataset(ax, ax_sec, ds, dim, KEflux_key, dwAw_key, LLL_key, Lww_key, x_key, scale_factor, quantile_alpha, quantile_upper, quantile_lower, confint, SF, CG, Ens, KE, dwAw, LLL, Ens_LLL, Lww, ke_flux_label, enstrophy_flux_label, dwdAw_label, CG_label, CG_label_q, LLL_label, Ens_LLL_label, Lww_label, convert_KEFlux_to_wattperkm2permeter, convert_KEFlux_to_Wperm3, colors):

    blue, lightblue, red, lightred = colors
    if Ens and not KE:
        ax_sec = ax

    if ds is not None:

        if convert_KEFlux_to_wattperkm2permeter:
            # Convert from m^2/s^3 to W/km^2/m assuming density = 1025 kg/m^3
            conversion_scale_factor = 1025 / 1e-6
        if convert_KEFlux_to_Wperm3:
            conversion_scale_factor = 1025
        else:
            conversion_scale_factor = 1.0

        if SF:

            if KE: 
                ax.semilogx(2 * np.pi / ds[x_key],
                    -0.5 * ds[KEflux_key].mean(dim) * scale_factor * conversion_scale_factor,
                    color=blue, zorder=1, label='Structure function: ' + ke_flux_label)
                
                if LLL:
                    ax.semilogx(2 * np.pi / ds[x_key],
                        -2 * ds[LLL_key].mean(dim) * scale_factor * conversion_scale_factor / (3 * ds[x_key]),
                        color=blue, linestyle="--", zorder=1, label='Structure function: ' + LLL_label)

            if Ens:
                ax_sec.semilogx(2 * np.pi / ds[x_key],
                    2 * ds[KEflux_key].mean(dim) * scale_factor / ds[x_key]**2,
                    color=red, zorder=1, label='Structure function: ' + enstrophy_flux_label)
                
                if Lww:
                    ax_sec.semilogx(2 * np.pi / ds[x_key],
                        -0.5 * ds[Lww_key].mean(dim) * scale_factor / ds[x_key],
                        color=red, linestyle="--", zorder=1, label='Structure function: ' + Lww_label)
                
                if dwAw:
                    ax_sec.semilogx(2 * np.pi / ds[x_key],
                        -0.5 * ds[dwAw_key].mean(dim) * scale_factor,
                        color=red, linestyle=":", zorder=1, label='Structure function: ' + dwdAw_label)
                
                if Ens_LLL:
                    ax_sec.semilogx(2 * np.pi / ds[x_key],
                        8 * ds[LLL_key].mean(dim) * scale_factor / ds[x_key]**3,
                        color=blue, linestyle=(0, (3, 5, 1, 5)), zorder=1, label='Structure function: ' + Ens_LLL_label)
            
            if confint:
                if KE:
                    ax.fill_between(2 * np.pi / ds[x_key],
                        -0.5 * ds[KEflux_key].quantile(quantile_lower, dim=dim) * scale_factor * conversion_scale_factor,
                        -0.5 * ds[KEflux_key].quantile(quantile_upper, dim=dim) * scale_factor * conversion_scale_factor,
                        color=blue, alpha=quantile_alpha)
                    
                    if LLL:
                        ax.fill_between(2 * np.pi / ds[x_key],
                            -2 * ds[LLL_key].quantile(quantile_lower, dim=dim) * scale_factor * conversion_scale_factor / (3 * ds[x_key]),
                            -2 * ds[LLL_key].quantile(quantile_upper, dim=dim) * scale_factor * conversion_scale_factor / (3 * ds[x_key]),
                            color=blue, alpha=quantile_alpha)
                
                if Ens:
                    ax_sec.fill_between(2 * np.pi / ds[x_key],
                        2 * ds[KEflux_key].quantile(quantile_lower, dim=dim) * scale_factor / ds[x_key]**2,
                        2 * ds[KEflux_key].quantile(quantile_upper, dim=dim) * scale_factor / ds[x_key]**2,
                        color=red, alpha=quantile_alpha)
                    
                    if dwAw:
                        ax_sec.fill_between(2 * np.pi / ds[x_key],
                            -0.5 * ds[dwAw_key].quantile(quantile_lower, dim=dim) * scale_factor,
                            -0.5 * ds[dwAw_key].quantile(quantile_upper, dim=dim) * scale_factor,
                            color=red, alpha=quantile_alpha)
                        
                    if Lww:
                        ax_sec.fill_between(2 * np.pi / ds[x_key],
                            -0.5 * ds[Lww_key].quantile(quantile_lower, dim=dim) * scale_factor / ds[x_key],
                            -0.5 * ds[Lww_key].quantile(quantile_upper, dim=dim) * scale_factor / ds[x_key],
                            color=red, alpha=quantile_alpha)
                        
                    if Ens_LLL:
                        ax_sec.fill_between(2 * np.pi / ds[x_key],
                            8 * ds[LLL_key].quantile(quantile_lower, dim=dim) * scale_factor / ds[x_key]**3,
                            8 * ds[LLL_key].quantile(quantile_upper, dim=dim) * scale_factor / ds[x_key]**3,
                            color=blue, alpha=quantile_alpha)

        if CG:

            if KE:
        
                ax.semilogx(ds["K_coarse_grain"].mean(dim),
                    ds["EFlux_CG"].mean(dim) * scale_factor * conversion_scale_factor,
                    color=lightblue, linestyle="-.", label=CG_label, zorder=0)
                
                if confint:
                    ax.fill_between(ds["K_coarse_grain"].mean(dim),
                        ds["EFlux_CG"].quantile(quantile_lower, dim=dim) * scale_factor * conversion_scale_factor,
                        ds["EFlux_CG"].quantile(quantile_upper, dim=dim) * scale_factor * conversion_scale_factor,
                        color=lightblue, alpha=quantile_alpha)

            if Ens:
                ax_sec.semilogx(ds["K_coarse_grain"].mean(dim),
                    ds["QFlux_CG"].mean(dim) * scale_factor,
                    color=lightred, linestyle="-.", label=CG_label_q, zorder=0)
                if confint:
                    ax_sec.fill_between(ds["K_coarse_grain"].mean(dim),
                        ds["QFlux_CG"].quantile(quantile_lower, dim=dim) * scale_factor,
                        ds["QFlux_CG"].quantile(quantile_upper, dim=dim) * scale_factor,
                        color=lightred, alpha=quantile_alpha)


def set_common_properties(axes, axes_sec, datasets, params, Ens, KE, ke_flux_label, enstrophy_flux_label, convert_KEFlux_to_wattperkm2permeter, convert_KEFlux_to_Wperm3, KE_lims_scale, Q_lims_scale, draw_Rossby_radius, colors, KE_ymin_override=None, KE_ymax_override=None, Q_ymin_override=None, Q_ymax_override=None, legend_outside=False):
    blue, lightblue, red, lightred = colors

    if Ens and KE:
        enums = enumerate(zip(axes, axes_sec, datasets.keys()))
        
    else:
        enums = enumerate(zip(axes, datasets.keys()))

    for i, vars in enums:
        if Ens and KE:
            ax, ax_sec, key = vars
        else:
            ax, key = vars
        ds = datasets[key]
        ax.set_xlabel("Wavenumber [m$^{-1}$]")
        ax.set_xlim(1e-6, 1e-2)  # Set x-limits for all axes

        if ds is not None:
            # Set ymin and ymax to be symmetric around zero
            asf_mean = abs(-0.5 * ds_SWOT["asf_down"].mean("swath_num")).mean()

            if convert_KEFlux_to_wattperkm2permeter:
                scale_factor = 1025 / 1e-6
            if convert_KEFlux_to_Wperm3:
                scale_factor = 1025
            else:
                scale_factor = 1.0

            asf_mean *= scale_factor

            ax.set_ylim(-asf_mean * KE_lims_scale, asf_mean * KE_lims_scale)

            ax.hlines(0, 1e-6, 1e-2, colors="k", lw=1, zorder=0)

            if draw_Rossby_radius:
                R = params["Rossby Radius"]  # units of kilometers
                if R is not None:
                    k_Rossby = distance_to_wavenumber(R * 1e3)  # convert to meters and then to wavenumber
                    # ax.axvline(k_Rossby, color="grey", linestyle="--", label="Rossby Radius", lw=1)
                    # ax.text(k_Rossby * 1.1, ax.get_ylim()[1] * 0.8, f"Ro {R} km", color="grey")

                    # Use fill between to shade R+/-
                    ax.fill_betweenx(
                        ax.get_ylim(),
                        k_Rossby * 0.5,
                        k_Rossby * 1.5,
                        color="grey",
                        alpha=0.3,
                    )
                    # put text in the lower middle of the shaded area
                    ax.text(k_Rossby, ax.get_ylim()[0] * 0.8, "Ro", color="k", ha="center")

            # Set y-label for primary y-axis
            if i == 0:  # Only set for the first axis
                if convert_KEFlux_to_wattperkm2permeter:
                    ke_flux_units = " W km$^{-2}$ m$^{-1}$"
                if convert_KEFlux_to_Wperm3:
                    ke_flux_units = " W m$^{-3}$"
                else:
                    ke_flux_units = " m$^2$ s$^{-3}$"
                
                enstrophy_flux_units = " s$^{-3}$"
                
                if KE and not Ens:
                    ax.set_ylabel(rf"KE flux [{ke_flux_units}]")
                elif not KE and Ens:
                    ax.set_ylabel(rf"Enstrophy flux [{enstrophy_flux_units}]")
            
            ax.tick_params(axis='x', direction="in", which="both", top=False)

            t = ax.yaxis.get_offset_text()
            t.set_x(-0.05)  # Adjust the position of the offset text

            # Add secondary x-axis for separation distance
            secax = ax.secondary_xaxis('top', functions=(wavenumber_to_distance, distance_to_wavenumber))
            secax.set_xlabel("Separation Distance [m]")
            secax.set_xlim(wavenumber_to_distance(1e-2), wavenumber_to_distance(1e-6))  # Match limits to primary x-axis
            secax.tick_params(direction="in", which="both", bottom=False)

            # Set properties for secondary y-axis
            if Ens:
                if KE:
                    if i == len(axes) - 1:  # Only set for the last axis
                        ax_sec.set_ylabel(rf"Enstrophy flux [{enstrophy_flux_units}]"
                                        f"\n {enstrophy_flux_label}",
                                        color=red)
                    ax.tick_params(axis='y', labelcolor=blue, direction="in")
                    ax_sec.tick_params(axis='y', labelcolor=red, direction="in")
                    t = ax_sec.yaxis.get_offset_text()
                    t.set_x(1.1)  # Adjust the position of the offset text

                if not KE:
                    ax_sec = ax

                enstrophy_mean = abs((2 * ds_SWOT["asf_down"].mean("swath_num") / ds_SWOT["num_lines_diffs"]**2)).mean()
                ax_sec.set_ylim(-enstrophy_mean * Q_lims_scale, enstrophy_mean * Q_lims_scale)

        # Add box with dataset name in the bottom left corner of each panel
        text = key if ds is not None else f"No {key} data"
        # text = 'SimSWOT no noise' if key == 'SimSWOT' else text
        # text = 'SimSWOT with noise' if key == 'SimSWOT_cleaned' else text
        ax.text(
            0.05, 0.05, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgrey', edgecolor='k', alpha=0.8)
        )

    # Add box with full region name in the top left corner of the first axis
    axes[0].text(0.05, 0.95, params["Full name"], transform=axes[0].transAxes,
                 fontsize=10, verticalalignment='top', fontweight='bold', zorder=0,
                 bbox=dict(boxstyle='round', facecolor='lightgrey', edgecolor='k', alpha=0.8))

    # Add legends
    axes[len(axes) - 1].legend(
        loc='upper left',  # Position the legend at the top center
        # # bbox_to_anchor=(0.5, 1.5),  # Adjust the position to be above the plot
        # ncol=2,  # Set the legend to have two columns
        # frameon=False  # Optional: Remove the legend box frame
        # # zorder=5
    )
    if legend_outside:
        axes[len(axes) - 1].legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False
        )

    if axes_sec is not None:
        axes_sec[len(axes_sec) - 1].legend(
            loc='upper right',  # Position the secondary legend at the upper center
            # bbox_to_anchor=(0.5, 1.2),  # Adjust the position to match the primary legend
            # ncol=2,  # Set the legend to have two columns
            # frameon=False  # Optional: Remove the legend box frame
            # # zorder=5
        )
        
        if legend_outside:
            axes_sec[len(axes_sec) - 1].legend(
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=False
            )

    if KE_ymin_override is not None and KE_ymax_override is not None:
        if KE:
            for ax in axes:
                ax.set_ylim(KE_ymin_override, KE_ymax_override)
    if Q_ymin_override is not None and Q_ymax_override is not None:
        if Ens:
            if not KE:
                axes_sec = axes
            for ax_sec in axes_sec:
                ax_sec.set_ylim(Q_ymin_override, Q_ymax_override)

def get_most_recent_file(region_name, file_pattern):
    """
    Get the most recent file for a specific region based on the datetime prefix in the filename.
    """
    # print(region_name, file_pattern)
    files = glob.glob(file_pattern)

    region_files = [f for f in files if ".nc" in os.path.basename(f)]
    if not region_files:
        raise FileNotFoundError(f"No files found for region: {region_name}")
    # Sort files by their datetime prefix (assuming the datetime is at the start of the filename)
    region_files.sort(key=lambda f: os.path.basename(f).split("_")[0], reverse=True)
    return region_files[0]

def merge_files_and_savenetcdf(region_output_dir, region, min_cycle=None, max_cycle=None, drop_cycles=None, cycle_list=None, startswith='20250730', date=False, mindate='', maxdate='',SWOT_name='SWOT_L3', suffix='CG'):

    if date and not (min_cycle or max_cycle):
        file_list = [f for f in os.listdir(region_output_dir) if f.endswith(".nc")]

        if not file_list:
            raise FileNotFoundError(f"No .nc files found in {region_output_dir} for region {region}.")
        datasets = [xr.open_dataset(os.path.join(region_output_dir, f)) for f in file_list if f.startswith(startswith) and region in f and f.endswith(f"_{suffix}.nc")]
        if not datasets:
            raise FileNotFoundError(f"No datasets found for region {region} starting with {startswith} and suffix {suffix}.")
        merged_dataset = xr.concat(datasets, dim='time')
        output_file = os.path.join(region_output_dir, f"LLC4320_{region}_merged_dates_{mindate}-{maxdate}_{suffix}.nc")

        merged_dataset.to_netcdf(output_file, mode='w', format='NETCDF4', engine='netcdf4')

        print(f"Merged dataset saved to: {output_file}")


    if not date:
        if (min_cycle is None or max_cycle is None) and cycle_list is None:
            raise ValueError("Both min_cycle and max_cycle must be specified when date is False.")
        
        if cycle_list is not None:
            cycle_files = [f for f in os.listdir(region_output_dir) if SWOT_name in f and f.startswith(startswith) and f.endswith(".nc") and any(cycle in f for cycle in cycle_list)]
            if not cycle_files:
                raise FileNotFoundError(f"No cycle files found for region {region} with cycles {cycle_list}.")
        else:

            cycle_files = [f for f in os.listdir(region_output_dir) if SWOT_name in f and f.startswith(startswith) and f.endswith(".nc")]

            # Filter out cycles that are not in the specified range
            cycle_files = [f for f in cycle_files if re.match(rf".*{SWOT_name}_{region}_cycle_(\d+)_{suffix}\.nc", f) and min_cycle <= re.search(rf"{SWOT_name}_{region}_cycle_(\d+)_{suffix}\.nc", f).group(1) <= max_cycle]

            if not cycle_files:
                # Try regex that looks for files formatted like "*_simSWOT_{region}_cycles_*-*_{suffix}.nc without specifying the cycle min and max
                cycle_files = [f for f in os.listdir(region_output_dir) if SWOT_name in f and f.startswith(startswith) and f.endswith(".nc") and re.match(rf".*_{SWOT_name}_{region}_cycles_\d+-\d+_{suffix}\.nc", f)]
                print(f"Cycle files: {cycle_files}")
        
        if not cycle_files:
            raise FileNotFoundError(f"No cycle files found for region {region} with cycles {min_cycle}-{max_cycle} or cycle_list {cycle_list}.")
        
        if drop_cycles is not None:
            cycle_files = [f for f in cycle_files if not any(drop_cycle in f for drop_cycle in drop_cycles)]
            if not cycle_files:
                raise FileNotFoundError(f"No cycle files left for region {region} after dropping cycles {drop_cycles}.")

        datasets = [xr.open_dataset(os.path.join(region_output_dir, f)) for f in cycle_files]

        try:
            merged_dataset = xr.concat(datasets, dim='swath_num')

        except ValueError:
            # Only keep variables that all to-be-merged datasets contain
            common_vars = set.intersection(*(set(ds.data_vars) for ds in datasets))
            if not common_vars:
                raise ValueError("No common variables found across datasets to concatenate.")

            filtered_datasets = [ds[sorted(common_vars)] for ds in datasets]
            merged_dataset = xr.concat(filtered_datasets, dim='swath_num')
            print(f"ValueError during concat; merged using only common variables: {sorted(common_vars)}")


        # Add cycle number as a new coordinate
        # cycle_numbers = [int(re.search(rf"{SWOT_name}_{region}_cycle_(\d+)_CG\.nc", f).group(1)) for f in cycle_files]
        # merged_dataset = merged_dataset.assign_coords(cycle_num=("swath_num", cycle_numbers))
        
        # datetime_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        if cycle_list is not None:
            output_file = os.path.join(region_output_dir, f"{SWOT_name}_{region}_cycles_{'-'.join(cycle_list)}_{suffix}.nc")
        else:
            output_file = os.path.join(region_output_dir, f"{SWOT_name}_{region}_cycles_{min_cycle}-{max_cycle}_{suffix}.nc")
        merged_dataset.to_netcdf(output_file, mode='w', format='NETCDF4', engine='netcdf4')
    
        print(f"Merged dataset saved to: {output_file}")

region_dict = {
    "acc": {
        "lat_north": -53,
        "lat_south": -57.5,
        "lon_east": 158,
        "lon_west": 148,
        "reduce_xd_num": 2,
        "LLC4320_file": "",
        "simSWOT_file": "",
        "NEMO_file": "",
        "Full name": "Antarctic Circumpolar Current",
        "Mean Latitude": " (55°S)",
        "Rossby Radius": 15,
    },
    "nwpacific": {
        "lat_north": 23,
        "lat_south": 19,
        "lon_east": 137,
        "lon_west": 132,
        "reduce_xd_num": 1,
        "LLC4320_file": "",
        "simSWOT_file": "",
        "NEMO_file": "",
        "Full name": "North Pacific",
        "Mean Latitude": " (21°N)",
        "Rossby Radius": 60,
    },
    "capebasin": {
        "lat_north": -41.01421,
        "lat_south": -44.99279,
        "lon_east": 16,
        "lon_west": 11,
        "reduce_xd_num": 1,
        "LLC4320_file": "",
        "simSWOT_file": "",
        "NEMO_file": "",
        "Full name": "Cape Basin",
        "Mean Latitude": " (43°S)",
        "Rossby Radius": 25,
    },
    "nwaustralia": {
        "lat_north": -11,
        "lat_south": -15,
        "lon_east": 125,
        "lon_west": 120,
        "reduce_xd_num": 1,
        "LLC4320_file": "",
        "simSWOT_file": "",
        "NEMO_file": "",
        "Full name": "Northwest Australia",
        "Mean Latitude": " (13°S)",
        "Rossby Radius": 100,
    },
    "westatlantic": {
        "lat_north": 38.7,
        "lat_south": 32.7,
        "lon_east": -73,
        "lon_west": -76,
        "reduce_xd_num": 1,
        "LLC4320_file": "",
        "simSWOT_file": "",
        "NEMO_file": "",
        "Full name": "West Atlantic",
        "Mean Latitude": " (35°N)",
        "Rossby Radius": 35,
    },
    "newcaledonia": {
        "lat_north": -22,
        "lat_south": -26,
        "lon_east": 171,
        "lon_west": 166,
        "reduce_xd_num": 1,
        "LLC4320_file": "",
        "simSWOT_file": "",
        "NEMO_file": "",
        "Full name": "New Caledonia",
        "Mean Latitude": " (24°S)",
        "Rossby Radius": "NA",
    },
    "labradorsea": {
        "lat_north": 63.79824,
        "lat_south": 59.5549,
        "lon_east": -58.52856,
        "lon_west": -63.61784,
        "reduce_xd_num": 1,
        "LLC4320_file": "",
        "simSWOT_file": "",
        "NEMO_file": "",
        "Full name": "Labrador Sea",
        "Mean Latitude": " (62°N)",
        "Rossby Radius": "NA",
    },
    # "Florida": {"lat_north": 29, "lat_south": 24, "lon_east": -77, "lon_west": -82, "reduce_xd_num": 1, "LLC4320_file": "", "SimSWOT_file": "", "NEMO_file": "", "Full name": "Florida", "Mean Latitude": " (26.5°N)", "Rossby Radius": 45},
    # "GrandBanks": {"lat_north": 40, "lat_south": 35, "lon_east": -45, "lon_west": -50, "reduce_xd_num": 1, "LLC4320_file": "", "SimSWOT_file": "", "NEMO_file": "", "Full name": "Grand Banks", "Mean Latitude": " (37.5°N)", "Rossby Radius": 30},
    # "Arbic": {"lat_north": 43, "lat_south": 27.5, "lon_east": -40, "lon_west": -60, "reduce_xd_num": 1, "LLC4320_file": "", "SimSWOT_file": "", "NEMO_file": "", "Full name": "Arbic", "Mean Latitude": " (35°N)", "Rossby Radius": 30},
    # "Equator": {"lat_north": 5, "lat_south": -5, "lon_east": -10, "lon_west": -35, "reduce_xd_num": 1, "LLC4320_file": "", "SimSWOT_file": "", "NEMO_file": "", "Full name": "Atlantic Equator", "Mean Latitude": " (0°)", "Rossby Radius": 200},
    # "Interior": {"lat_north": 32, "lat_south": 22, "lon_east": -30, "lon_west": -42, "reduce_xd_num": 1, "LLC4320_file": "", "SimSWOT_file": "", "NEMO_file": "", "Full name": "Atlantic Interior", "Mean Latitude": " (27°N)", "Rossby Radius": 40},
    # "GulfStream": {"southwest": (-80, 25), "southeast": (-35, 25), "northwest": (-80, 39), "northeast": (-35, 49), "reduce_xd_num": 1, "LLC4320_file": "", "SimSWOT_file": "", "NEMO_file": "", "Full name": "Gulf Stream", "Mean Latitude": " (37°N)", "Rossby Radius": 35},
}


# Update the region_dict dynamically
for region, params in region_dict.items():
    try:
        # Merge files if data/MITgcm/{region}/*merged_dates_{mindate}-{maxdate}.nc" does not exist
        region_output_dir = f"data/MITgcm/{region}/{region}_all"
        mindate = "20110913"
        maxdate = "20121114"
        suffix = "timemean_removed_Bessels_tapered_CG"
        merged_file = os.path.join(
            region_output_dir,
            f"LLC4320_{region}_merged_dates_{mindate}-{maxdate}_{suffix}.nc",
        )
        run_new = False 
        if not os.path.exists(merged_file) or run_new:
            if os.path.exists(merged_file):
                os.remove(merged_file)
                print(f"Deleted existing file: {merged_file}")
                
            merge_files_and_savenetcdf(
                region_output_dir,
                region,
                date=True,
                mindate=mindate,
                maxdate=maxdate,
                startswith="2026",
                suffix=suffix,
            )
        else:
            print(f"Merged file already exists for region {region}: {merged_file}")

        # Get the most recent LLC4320 file
        params["LLC4320_file"] = get_most_recent_file(
            region_name=region,
            file_pattern=f"data/MITgcm/{region}/{region}_all/*_{mindate}-{maxdate}_{suffix}.nc",
        )
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping LLC4320 file for region: {region}")

    # try:
    #     # Get the most recent SimSWOT file the new way
    #     params["SimSWOT_file"] = get_most_recent_file(
    #         region_name=region,
    #         file_pattern=f"data/simSWOT/{region}/SimSWOT_{region}_all_cycles*.nc",
    #     )
    # except FileNotFoundError:
    #     print(f"No new SimSWOT file for region: {region}.")

    # try:
    #     # Get the most recent NEMO file
    #     params["NEMO_file"] = get_most_recent_file(
    #         region_name=region, file_pattern=f"data/NEMO/{region}/*_NEMO_*.nc"
    #     )
    # except FileNotFoundError as e:
    #     print(f"Warning: {e}. Skipping NEMO file for region: {region}")

    try:
        min_cycle = "001"
        max_cycle = "040"

        make_new = False  # Set to True to force new file creation, False to skip if file exists

        # Merge files if data/SWOT_L3/{region}/*_allcycles_{region}.nc" does not exist
        region_output_dir = f"data/SWOT_L3/SWOT_L3_LR_SSH_3.0/2026-04-06/{region}"
        suffix = "timemean_removed_Bessels_tapered_CG"
        all_cycles_file = os.path.join(
            region_output_dir,
            f"SWOT_L3_{region}_cycles_{min_cycle}-{max_cycle}_{suffix}.nc",
        )
        print(all_cycles_file)
        if not os.path.exists(all_cycles_file) or make_new:
            merge_files_and_savenetcdf(
                region_output_dir,
                region,
                min_cycle=min_cycle,
                drop_cycles=["032"],
                max_cycle=max_cycle,
                startswith="2026",
                suffix=suffix,
            )
        else:
            print(
                f"All cycles file already exists for region {region}: {all_cycles_file}"
            )

        # Get most recently merged SWOT file
        params["SWOT_file"] = get_most_recent_file(
            region_name=region,
            file_pattern=f"data/SWOT_L3/SWOT_L3_LR_SSH_3.0/2026-04-06/{region}/*_{min_cycle}-{max_cycle}_{suffix}.nc",
        )
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping SWOT file for region: {region}")


    try:
        cycle_list = None
        min_cycle = "001"
        max_cycle = "017"
        # Merge files if data/simSWOT/{region}/*_allcycles_{region}.nc" does not exist
        region_output_dir = f"data/simSWOT/2026-04-07/{region}"

        if region == "acc":
            suffix = "timemean_removed_cycle_mean_flipped_swath_LLL_Bessels_tapered_CG"
        else:
            suffix = "timemean_removed_linear_fit_Bessels_tapered_CG"

        if cycle_list:
            all_cycles_file = os.path.join(
                region_output_dir,
                f"simSWOT_{region}_cycles_{'-'.join(cycle_list)}_{suffix}.nc",
            )
            file_pattern = (
                f"data/simSWOT/2026-04-07/{region}/*_{'-'.join(cycle_list)}_{suffix}.nc"
            )
        elif min_cycle and max_cycle:
            all_cycles_file = os.path.join(
                region_output_dir,
                f"simSWOT_{region}_cycles_{min_cycle}-{max_cycle}_{suffix}.nc",
            )
            file_pattern = (
                f"data/simSWOT/2026-04-07/{region}/*_{min_cycle}-{max_cycle}_{suffix}.nc"
            )
        else:
            raise ValueError(
                "Either cycle_list or both min_cycle and max_cycle must be provided."
            )

        run_new = False  # Set to True to force new file creation, False to skip if file exists
        if run_new and os.path.exists(all_cycles_file):
            os.remove(all_cycles_file)
            print(f"Deleted existing file: {all_cycles_file}")

            # If the file was deleted, we should also set the file_pattern to look for the individual cycle files that will be merged to create the all_cycles_file
            if cycle_list:
                file_pattern = f"data/simSWOT/2026-04-07/{region}/*_{'-'.join(cycle_list)}_{suffix}.nc"
            elif min_cycle and max_cycle:
                file_pattern = f"data/simSWOT/2026-04-07/{region}/*_{min_cycle}-{max_cycle}_{suffix}.nc"
            else:
                raise ValueError(
                    "Either cycle_list or both min_cycle and max_cycle must be provided."
                )

        if not os.path.exists(all_cycles_file):

            if cycle_list:
                merge_files_and_savenetcdf(
                    region_output_dir,
                    region,
                    cycle_list=cycle_list,
                    startswith="2026",
                    SWOT_name="simSWOT",
                    suffix=suffix,
                )
            elif min_cycle and max_cycle:
                merge_files_and_savenetcdf(
                    region_output_dir,
                    region,
                    min_cycle=min_cycle,
                    max_cycle=max_cycle,
                    startswith="2026",
                    SWOT_name="simSWOT",
                    suffix=suffix,
                )
            else:
                raise ValueError(
                    "Either cycle_list or both min_cycle and max_cycle must be provided."
                )
        else:
            print(
                f"All cycles file already exists for region {region}: {all_cycles_file}"
            )

        # Get most recently merged SWOT file
        params["simSWOT_file"] = get_most_recent_file(
            region_name=region, file_pattern=file_pattern
        )
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping simSWOT file for region: {region}")

# merge fast_phase files for  simSWOT
for region, params in region_dict.items():

    # try:

    #     # Merge simSWOT files for fast_phase if not already merged
    #     region_output_dir = f"data/simSWOT/fast_phase/2026-04-19/{region}"
    #     min_cycle = "001"
    #     max_cycle = None

    #     if max_cycle is None:
    #         # search for max cycle in the directory and set max_cycle to max-1
    #         filenames = [f for f in os.listdir(region_output_dir) if f.startswith("2026") and f.endswith(".nc")]
    #         cycle_numbers = []
    #         for filename in filenames:
    #             match = re.search(rf"simSWOT_{region}_cycle_(\d+)_flipped_swath_LLL_Bessels_tapered_CG\.nc", filename)
    #             if match:
    #                 cycle_numbers.append(int(match.group(1)))
    #         if cycle_numbers:
    #             max_cycle_found = f"{max(cycle_numbers):03d}"
    #             print(f"Set max_cycle to {int(max_cycle_found) - 1:03d} based on files in {region_output_dir}")
    #             max_cycle = f"{int(max_cycle_found) - 1:03d}"
    #         else:
    #             raise FileNotFoundError(f"No cycle files found in {region_output_dir} to determine max_cycle for region {region}.")

    #     suffix = "flipped_swath_LLL_Bessels_tapered_CG"
    #     all_cycles_file = os.path.join(
    #         region_output_dir,
    #         f"simSWOT_{region}_cycles_{min_cycle}-{max_cycle}_{suffix}.nc",
    #     )
    #     run_new = True  # Set to True to force new file creation, False to skip if file exists
    #     if not os.path.exists(all_cycles_file) or run_new:
    #         if os.path.exists(all_cycles_file):
    #             os.remove(all_cycles_file)
    #             print(f"Deleted existing file: {all_cycles_file}")
    #         merge_files_and_savenetcdf(
    #             region_output_dir,
    #             region,
    #             min_cycle=min_cycle,
    #             max_cycle=max_cycle,
    #             startswith="2026",
    #             SWOT_name="simSWOT",
    #             suffix=suffix,
    #         )
    #     else:
    #         print(
    #             f"All cycles file already exists for region {region}: {all_cycles_file}"
    #         )          # Get most recently merged SWOT file 
    #     params["simSWOT_fast_phase_file"] = get_most_recent_file(
    #         region_name=region,
    #         file_pattern=f"data/simSWOT/fast_phase/2026-04-19/{region}/*_{min_cycle}-{max_cycle}_{suffix}.nc",
    #     )
    # except FileNotFoundError as e:
    #     print(f"Warning: {e}. Skipping simSWOT fast_phase file for region: {region}")

    try:
        # example file for cycle chunks 20260422_145420_simSWOT_acc_cycles_001-016_LLL_Bessels_tapered_CG.nc

        # Merge simSWOT files for fast_phase if not already merged
        region_output_dir = f"data/simSWOT/fast_phase/2026-04-19/{region}"
        min_cycle = "001"
        max_cycle = None
        suffix = "timemean_removed_linear_fit_LLL_Bessels_tapered_CG"

        if max_cycle is None:
            # search for max cycle in the directory and set max_cycle to max-1
            filenames = [f for f in os.listdir(region_output_dir) if f.startswith("2026") and f.endswith(".nc")]
            cycle_numbers = []
            for filename in filenames:
                match = re.search(rf"_simSWOT_{region}_cycles_(\d+)-(\d+)_{suffix}\.nc", filename)
                if match:
                    cycle_numbers.append(int(match.group(2)))
            if cycle_numbers:
                max_cycle_found = f"{max(cycle_numbers):03d}"
                print(f"Set max_cycle to {int(max_cycle_found):03d} based on files in {region_output_dir}")
                max_cycle = f"{int(max_cycle_found):03d}"
            else:
                raise FileNotFoundError(f"No cycle files found in {region_output_dir} to determine max_cycle for region {region}.")

        all_cycles_file = os.path.join(
            region_output_dir,
            f"simSWOT_{region}_cycles_{min_cycle}-{max_cycle}_{suffix}.nc",
        )
        run_new = False  # Set to True to force new file creation, False to skip if file exists
        if not os.path.exists(all_cycles_file) or run_new:
            if os.path.exists(all_cycles_file):
                os.remove(all_cycles_file)
                print(f"Deleted existing file: {all_cycles_file}")
            merge_files_and_savenetcdf(
                region_output_dir,
                region,
                min_cycle=min_cycle,
                max_cycle=max_cycle,
                startswith="2026",
                SWOT_name="simSWOT",
                suffix=suffix,
            )
        else:
            print(
                f"All cycles file already exists for region {region}: {all_cycles_file}"
            )          # Get most recently merged SWOT file 
        params["simSWOT_fast_phase_file"] = get_most_recent_file(
            region_name=region,
            file_pattern=f"data/simSWOT/fast_phase/2026-04-19/{region}/simSWOT_{region}_cycles_{min_cycle}-{max_cycle}_{suffix}.nc",
        )
    except FileNotFoundError as e:
        print(f"Warning: {e}. Skipping simSWOT fast_phase file for region: {region}")

def plot_1x3_allmethods(region_dict, region_idx=0, confint=True, Bessels=False, LLL=False, KE=True, KE_ymin=None, KE_ymax=None, Bessel_mean_bug=True, filter_outliers=False,use_km=False, convert_KEFlux_to_wattperm3=False, season=None, simSWOT_fastphase=False):
    def _validate_xy_shapes(x, y, label):
        x_arr = np.asarray(x).squeeze()
        y_arr = np.asarray(y).squeeze()

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(
                f"{label}: expected 1D x/y but got ndim x={x_arr.ndim}, y={y_arr.ndim}."
            )
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"{label}: x/y length mismatch x={x_arr.shape[0]}, y={y_arr.shape[0]}."
            )

    def _time_range_or_na(ds, time_name):
        if time_name not in ds.coords:
            return "N/A"
        if ds.sizes.get(time_name, 0) == 0:
            return "N/A"
        return f"{ds[time_name].min(skipna=True).dt.strftime('%Y').item()} - {ds[time_name].max(skipna=True).dt.strftime('%Y').item()}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    region, params = list(region_dict.items())[region_idx]

    ds_LLC4320 = load_dataset("LLC4320_file", params, region)

    if simSWOT_fastphase:
        ds_simSWOT = load_dataset("simSWOT_fast_phase_file", params, region)
    else:
        ds_simSWOT = load_dataset("simSWOT_file", params, region)
    ds_SWOT = load_dataset("SWOT_file", params, region)

    if any(ds is None for ds in [ds_LLC4320, ds_simSWOT, ds_SWOT]):
        print(f"Skipping plot for region {region}: one or more datasets failed to load.")
        plt.close(fig)
        return None

    try:

        if season is not None:
            ds_LLC4320 = ds_LLC4320.assign_coords(time=pd.date_range(start="2011-09-13", periods=ds_LLC4320.sizes['time'], freq='h'))
            ds_simSWOT = ds_simSWOT.assign_coords(time_coord=ds_simSWOT['time'].mean(["num_lines", "num_pixels"]))
            ds_SWOT = ds_SWOT.assign_coords(time_coord=ds_SWOT['time'].mean(["num_lines", "num_pixels"]))

            # cut datasets to season months where season is a list of month integers, e.g. [12, 1, 2] for DJF
            ds_LLC4320 = ds_LLC4320.sel(time=ds_LLC4320['time'].dt.month.isin(season))
            ds_simSWOT = ds_simSWOT.sel(time_coord=ds_simSWOT['time_coord'].dt.month.isin(season))
            ds_SWOT = ds_SWOT.sel(time_coord=ds_SWOT['time_coord'].dt.month.isin(season))

            # Skip this season cleanly if any dataset has no samples after filtering.
            if ds_LLC4320.sizes.get('time', 0) == 0:
                print(f"Skipping season {season}: LLC4320 has 0 snapshots after month filter.")
                try:
                    ds_LLC4320.close()
                    ds_simSWOT.close()
                    ds_SWOT.close()
                except Exception:
                    pass
                plt.close(fig)
                return None
            if ds_simSWOT.sizes.get('swath_num', 0) == 0:
                print(f"Skipping season {season}: simSWOT has 0 swaths after month filter.")
                try:
                    ds_LLC4320.close()
                    ds_simSWOT.close()
                    ds_SWOT.close()
                except Exception:
                    pass
                plt.close(fig)
                return None
            if ds_SWOT.sizes.get('swath_num', 0) == 0:
                print(f"Skipping season {season}: SWOT has 0 swaths after month filter.")
                try:
                    ds_LLC4320.close()
                    ds_simSWOT.close()
                    ds_SWOT.close()
                except Exception:
                    pass
                plt.close(fig)
                return None

        # Set flux variables
        if KE is True:

            ds_LLC4320_KE_asf = -0.5 * ds_LLC4320['asf_mean'] * (1318 / 2000) ** 2
            ds_simSWOT_KE_asf = -0.5 * ds_simSWOT['asf_mean']
            ds_SWOT_KE_asf = -0.5 * ds_SWOT['asf_mean']

            # ds_LLC4320_enst_asf = 2 * ds_LLC4320['asf_mean'] * (1318 / 2000) ** 2 / ds_LLC4320['i_diffs']**2
            # ds_simSWOT_enst_asf = 2 * ds_simSWOT['asf_down'] / ds_simSWOT['num_lines_diffs']**2
            # ds_SWOT_enst_asf = 2 * ds_SWOT['asf_down'] / ds_SWOT['num_lines_diffs']**2

            ds_LLC4320_KE_CG = ds_LLC4320['EFlux_CG'] * (1318 / 2000) ** 2
            ds_simSWOT_KE_CG = ds_simSWOT['EFlux_CG'] 
            ds_SWOT_KE_CG = ds_SWOT['EFlux_CG']

            if Bessel_mean_bug:

                ds_LLC4320["EFlux_Bessel_ASF_mean_tapered"] = (sum(ds_LLC4320[var] for var in ds_LLC4320.data_vars if "Bessel_asf" in var)/4)
                ds_simSWOT["EFlux_Bessel_ASF_mean_tapered"] = (sum(ds_simSWOT[var] for var in ds_simSWOT.data_vars if "Bessel_asf" in var)/4)
                ds_SWOT["EFlux_Bessel_ASF_mean_tapered"] = (sum(ds_SWOT[var] for var in ds_SWOT.data_vars if "Bessel_asf" in var)/4)


            ds_LLC4320_KE_Bessel = ds_LLC4320['EFlux_Bessel_ASF_mean_tapered'] * (1318 / 2000) ** 2
            ds_simSWOT_KE_Bessel = ds_simSWOT['EFlux_Bessel_ASF_mean_tapered'] 
            ds_SWOT_KE_Bessel = ds_SWOT['EFlux_Bessel_ASF_mean_tapered']

            # ds_LLC4320_KE_LLL = (2/(3 * ds_LLC4320.i_diffs)) * ds_LLC4320['LLL_mean'] * (1318 / 2000) ** 2
            ds_simSWOT_KE_LLL = (2/(3 * ds_simSWOT.num_lines_diffs)) * ds_simSWOT['LLL_mean']
            # ds_SWOT_KE_LLL = (2/(3 * ds_SWOT.num_lines_diffs)) * ds_SWOT['LLL_mean']

            if convert_KEFlux_to_wattperm3:
                ds_LLC4320_KE_asf *= 1025
                ds_simSWOT_KE_asf *= 1025
                ds_SWOT_KE_asf *= 1025

                ds_LLC4320_KE_CG *= 1025
                ds_simSWOT_KE_CG *= 1025
                ds_SWOT_KE_CG *= 1025

                ds_LLC4320_KE_Bessel *= 1025
                ds_simSWOT_KE_Bessel *= 1025
                ds_SWOT_KE_Bessel *= 1025

                # ds_LLC4320_KE_LLL *= 1025
                ds_simSWOT_KE_LLL *= 1025
                # ds_SWOT_KE_LLL *= 1025

        # TODO LATER
        # if Q is True:

        #     ds_LLC4320_enst_dwAw = -0.5 * ds_LLC4320['asfq_mean'] * (1318 / 2000) ** 2
        #     ds_simSWOT_enst_dwAw = -0.5 * ds_simSWOT['asfq_down']
        #     ds_SWOT_enst_dwAw = -0.5 * ds_SWOT['asfq_down']

        #     ds_LLC4320_enst_CG = ds_LLC4320['QFlux_CG'] * (1318 / 2000) ** 2
        #     ds_simSWOT_enst_CG = ds_simSWOT['QFlux_CG']
        #     ds_SWOT_enst_CG = ds_SWOT['QFlux_CG']



        # Set x/K variables
        ds_LLC4320_Kvar_asf = 1 / ds_LLC4320['i_diffs']
        ds_simSWOT_Kvar_asf = 1 / ds_simSWOT['num_lines_diffs']
        ds_SWOT_Kvar_asf = 1 / ds_SWOT['num_lines_diffs']

        ds_LLC4320_Kvar_CG = ds_LLC4320['K_coarse_grain'].mean('time') / np.pi
        ds_simSWOT_Kvar_CG = ds_simSWOT['K_coarse_grain'].mean('swath_num') / np.pi
        ds_SWOT_Kvar_CG = ds_SWOT['K_coarse_grain'].mean('swath_num') / np.pi

        ds_LLC4320_Kvar_Bessel = ds_LLC4320['K']
        ds_simSWOT_Kvar_Bessel = ds_simSWOT['K']
        ds_SWOT_Kvar_Bessel = ds_SWOT['K']

        if use_km:
            ds_LLC4320_Kvar_asf *= 1e3
            ds_simSWOT_Kvar_asf *= 1e3
            ds_SWOT_Kvar_asf *= 1e3

            ds_LLC4320_Kvar_CG *= 1e3
            ds_simSWOT_Kvar_CG *= 1e3
            ds_SWOT_Kvar_CG *= 1e3

            ds_LLC4320_Kvar_Bessel *= 1e3
            ds_simSWOT_Kvar_Bessel *= 1e3
            ds_SWOT_Kvar_Bessel *= 1e3

        if filter_outliers:
        # Filter out outliers based on percentiles of the KE values for each dataset and set the newly filtered KE values to the original variable so that the plotting code doesn't need to be changed
            for ds, KE_var in zip([ds_LLC4320_KE_asf, ds_simSWOT_KE_asf, ds_SWOT_KE_asf], [ds_LLC4320_KE_asf, ds_simSWOT_KE_asf, ds_SWOT_KE_asf]):
                lower_bound = KE_var.quantile(0.1)
                upper_bound = KE_var.quantile(0.9)
                KE_var_filtered = KE_var.where((KE_var >= lower_bound) & (KE_var <= upper_bound))
                KE_var.data = KE_var_filtered.data



        # Plotting parameters
        asf_color = 'tab:blue'
        asf_linewidth = 2
        asf_linestyle = 'solid'

        CG_color = 'k'
        CG_linewidth = 3
        CG_linestyle = 'solid'

        Bessel_color = 'tab:blue'
        Bessel_linewidth = 2
        Bessel_linestyle = 'solid'

        LLL_color = 'tab:red'
        LLL_linestyle = '--'
        LLL_linewidth = 2

        _validate_xy_shapes(ds_LLC4320_Kvar_asf, ds_LLC4320_KE_asf.mean('time'), 'LLC4320 ASF')
        _validate_xy_shapes(ds_LLC4320_Kvar_CG, ds_LLC4320_KE_CG.mean('time'), 'LLC4320 CG')
        _validate_xy_shapes(ds_simSWOT_Kvar_asf, ds_simSWOT_KE_asf.mean('swath_num'), 'simSWOT ASF')
        _validate_xy_shapes(ds_simSWOT_Kvar_CG, ds_simSWOT_KE_CG.mean('swath_num'), 'simSWOT CG')
        _validate_xy_shapes(ds_SWOT_Kvar_asf, ds_SWOT_KE_asf.mean('swath_num'), 'SWOT ASF')
        _validate_xy_shapes(ds_SWOT_Kvar_CG, ds_SWOT_KE_CG.mean('swath_num'), 'SWOT CG')

        # Plot LLC4320 variables
        axes[0].semilogx(ds_LLC4320_Kvar_asf, ds_LLC4320_KE_asf.mean('time'),c=asf_color,ls=asf_linestyle, lw=asf_linewidth)
        axes[0].semilogx(ds_LLC4320_Kvar_CG, ds_LLC4320_KE_CG.mean('time'),c=CG_color,ls=CG_linestyle,lw=CG_linewidth)
        axes[0].semilogx(ds_LLC4320_Kvar_Bessel, ds_LLC4320_KE_Bessel.mean('time'),c=Bessel_color,ls=Bessel_linestyle, lw=Bessel_linewidth) if Bessels else None
        # axes[0].semilogx(ds_LLC4320_Kvar_asf, ds_LLC4320_KE_LLL.mean('time'),c=LLL_color,ls=LLL_linestyle, lw=LLL_linewidth) if LLL else None

        # Plot simSWOT variables
        axes[1].semilogx(ds_simSWOT_Kvar_asf, ds_simSWOT_KE_asf.mean('swath_num'),c=asf_color,ls=asf_linestyle, lw=asf_linewidth, label='ASF')
        axes[1].semilogx(ds_simSWOT_Kvar_CG, ds_simSWOT_KE_CG.mean('swath_num'),c=CG_color,ls=CG_linestyle,lw=CG_linewidth, label='CG')
        axes[1].semilogx(ds_simSWOT_Kvar_Bessel, ds_simSWOT_KE_Bessel.mean('swath_num'),c=Bessel_color,ls=Bessel_linestyle, lw=Bessel_linewidth, label='Bessel') if Bessels else None
        axes[1].semilogx(ds_simSWOT_Kvar_asf, ds_simSWOT_KE_LLL.mean('swath_num'),c=LLL_color,ls=LLL_linestyle, lw=LLL_linewidth, label='LLL') if LLL else None

        # Plot SWOT variables
        axes[2].semilogx(ds_SWOT_Kvar_asf, ds_SWOT_KE_asf.mean('swath_num'),c=asf_color,ls=asf_linestyle, lw=asf_linewidth, label='ASF')
        axes[2].semilogx(ds_SWOT_Kvar_CG, ds_SWOT_KE_CG.mean('swath_num'),c=CG_color,ls=CG_linestyle,lw=CG_linewidth, label='CG')
        axes[2].semilogx(ds_SWOT_Kvar_Bessel, ds_SWOT_KE_Bessel.mean('swath_num'),c=Bessel_color,ls=Bessel_linestyle, lw=Bessel_linewidth, label='Bessel') if Bessels else None
        # axes[2].semilogx(ds_SWOT_Kvar_asf, ds_SWOT_KE_LLL.mean('swath_num'),c=LLL_color,ls=LLL_linestyle, lw=LLL_linewidth, label='LLL') if LLL else None    

        if confint:
            upper_quantile = 0.90
            lower_quantile = 0.1

            # Plot LLC4320 confidence intervals
            axes[0].fill_between(
                ds_LLC4320_Kvar_asf, 
                ds_LLC4320_KE_asf.quantile(lower_quantile, dim='time'),
                ds_LLC4320_KE_asf.quantile(upper_quantile, dim='time'),
                color=asf_color, alpha=0.2
                                )
            axes[0].fill_between(
                ds_LLC4320_Kvar_CG, 
                ds_LLC4320_KE_CG.quantile(lower_quantile, dim='time'),
                ds_LLC4320_KE_CG.quantile(upper_quantile, dim='time'),
                color=CG_color, alpha=0.2
                                )
            axes[0].fill_between(
                ds_LLC4320_Kvar_Bessel, 
                ds_LLC4320_KE_Bessel.quantile(lower_quantile, dim='time'),
                ds_LLC4320_KE_Bessel.quantile(upper_quantile, dim='time'),
                color=Bessel_color, alpha=0.2
                                ) if Bessels else None      

            # axes[0].fill_between(
            #     ds_LLC4320_Kvar_asf, 
            #     ds_LLC4320_KE_LLL.quantile(lower_quantile, dim='time'),
            #     ds_LLC4320_KE_LLL.quantile(upper_quantile, dim='time'),
            #     color='tab:green', alpha=0.2
            #                     ) if LLL else None

            # Plot simSWOT confidence intervals
            axes[1].fill_between(
                ds_simSWOT_Kvar_asf,
                ds_simSWOT_KE_asf.quantile(lower_quantile, dim='swath_num'),
                ds_simSWOT_KE_asf.quantile(upper_quantile, dim='swath_num'),
                color=asf_color, alpha=0.2
            )
            axes[1].fill_between(
                ds_simSWOT_Kvar_CG,
                ds_simSWOT_KE_CG.quantile(lower_quantile, dim='swath_num'),
                ds_simSWOT_KE_CG.quantile(upper_quantile, dim='swath_num'),
                color=CG_color, alpha=0.2
            )
            axes[1].fill_between(
                ds_simSWOT_Kvar_Bessel,
                ds_simSWOT_KE_Bessel.quantile(lower_quantile, dim='swath_num'),
                ds_simSWOT_KE_Bessel.quantile(upper_quantile, dim='swath_num'),
                color=Bessel_color, alpha=0.2
            ) if Bessels else None

            axes[1].fill_between(
                ds_simSWOT_Kvar_asf,
                ds_simSWOT_KE_LLL.quantile(lower_quantile, dim='swath_num'),
                ds_simSWOT_KE_LLL.quantile(upper_quantile, dim='swath_num'),
                color='tab:green', alpha=0.2
            ) if LLL else None

            # Plot SWOT confidence intervals
            axes[2].fill_between(
                ds_SWOT_Kvar_asf,
                ds_SWOT_KE_asf.quantile(lower_quantile, dim='swath_num'),
                ds_SWOT_KE_asf.quantile(upper_quantile, dim='swath_num'),
                color=asf_color, alpha=0.2
            )
            axes[2].fill_between(
                ds_SWOT_Kvar_CG,
                ds_SWOT_KE_CG.quantile(lower_quantile, dim='swath_num'),
                ds_SWOT_KE_CG.quantile(upper_quantile, dim='swath_num'),
                color=CG_color, alpha=0.2
            )
            axes[2].fill_between(
                ds_SWOT_Kvar_Bessel,
                ds_SWOT_KE_Bessel.quantile(lower_quantile, dim='swath_num'),
                ds_SWOT_KE_Bessel.quantile(upper_quantile, dim='swath_num'),
                color=Bessel_color, alpha=0.2
            ) if Bessels else None

            # axes[2].fill_between(
            #     ds_SWOT_Kvar_asf,
            #     ds_SWOT_KE_LLL.quantile(lower_quantile, dim='swath_num'),
            #     ds_SWOT_KE_LLL.quantile(upper_quantile, dim='swath_num'),
            #     color='tab:green', alpha=0.2
            # ) if LLL else None

        # Add text with full region name and mean latitude in the top left corner of the first column
        axes[0].text(0.05, 0.95, f"{params['Full name']} {params['Mean Latitude']}", transform=axes[0].transAxes,
                        fontsize=10, verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgrey', edgecolor='k', alpha=0.8))
        
        # Add text with dataset name and date range in the bottom left corner of each panel

        # define season_str based on season list, e.g. [12, 1, 2] -> DJF, [3, 4, 5] -> MAM, etc. Numbers like [2,5,7] that don't correspond to a standard season should just be joined with commas, e.g. "Feb, May, Jul"
        if season is not None:
            if season == [12, 1, 2]:
                season_str = "DJF"
            elif season == [3, 4, 5]:
                season_str = "MAM"
            elif season == [6, 7, 8]:
                season_str = "JJA"
            elif season == [9, 10, 11]:
                season_str = "SON"
            else:
                season_str = ', '.join([pd.to_datetime(month, format='%m').strftime('%b') for month in season])
        else:
            season_str = "Full Year"

        axes[0].text(0.05, 0.05, f"LLC4320 - {ds_LLC4320.time.size} snapshots \n {season_str} {_time_range_or_na(ds_LLC4320, 'time')}", transform=axes[0].transAxes,
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightgrey', edgecolor='k', alpha=0.8))
        
        axes[1].text(0.05, 0.05, f"SimSWOT - {ds_simSWOT.swath_num.size} swaths\n{season_str} {_time_range_or_na(ds_simSWOT, 'time_coord')}", transform=axes[1].transAxes,
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightgrey', edgecolor='k', alpha=0.8))
        
        axes[2].text(0.05, 0.05, f"SWOT - {ds_SWOT.swath_num.size} swaths\n{season_str} {_time_range_or_na(ds_SWOT, 'time_coord')}", transform=axes[2].transAxes,
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightgrey', edgecolor='k', alpha=0.8))

        # Set ylim/ymax to be the same for all panels in a row based on the ylim/ymax of LLC4320 only
        for i in range(3):
            if any(v is not None for v in [KE_ymin, KE_ymax]):
                axes[i].set_ylim(KE_ymin, KE_ymax)
            # elif any(v is not None for v in [Q_ymin, Q_ymax]) and 'Enstrophy flux' in plotting_variable:
            #     axes[i].set_ylim(Q_ymin, Q_ymax)
            else:
                # set ylim based on the max abs val of the LLC4320 ylim to make it symmetric around 0
                max_abs_ylim = max(abs(axes[0].get_ylim()[0]), abs(axes[0].get_ylim()[1]))
                axes[i].set_ylim(-max_abs_ylim, max_abs_ylim)
                

            # if i == 0:
            #     # if convert_Wm3:
            #     #     plotting_variable = plotting_variable + r" [W m$^{-3}$]"
            #     # else:
            #     plotting_variable = plotting_variable + r" [m$^2$ s$^{-3}$]"

            # elif 'Enstrophy flux' in plotting_variable and i == 0:
            #     plotting_variable = plotting_variable + r" [s$^{-3}$]"
            axes[0].set_ylabel(r"Kinetic energy flux [m$^2$ s$^{-3}$]")
            axes[i].set_xlabel(r"Wavenumber [m$^{-1}$]")
            axes[i].set_xlim(1e-6, 1e-2)  # Set x-limits for all axes
            axes[i].hlines(0, 1e-6, 1e-2, colors="k", lw=1, zorder=0)
            secax = axes[i].secondary_xaxis('top', functions=(wavenumber_to_distance, distance_to_wavenumber))
            secax.set_xlabel("Separation Distance [m]")
            secax.set_xlim(wavenumber_to_distance(1e-2), wavenumber_to_distance(1e-6))  # Match limits to primary x-axis
            secax.tick_params(direction="in", which="both", bottom=False)

            if use_km:
                axes[i].set_xlabel(r"Wavenumber [km$^{-1}$]")
                secax.set_xlabel("Separation Distance [km]")
                axes[i].set_xlim(1e-3, 1e1)  # Set x-limits for all axes in km
                secax.set_xlim(wavenumber_to_distance(1e1), wavenumber_to_distance(1e-3))  # Match limits to primary x-axis in km
                axes[i].hlines(0, 1e-3, 1e1, colors="k", lw=1, zorder=0)

                # draw vertical line at 30km for acc region
                # if region == "acc":
                #     axes[i].axvline(x=1/30, color='k', ls='--', lw=1) 
        
        axes[1].legend(loc='upper right')
            
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error occurred while plotting for season {season}: {e}")
        print(f"LLC4320 dims: {dict(ds_LLC4320.sizes)}")
        print(f"simSWOT dims: {dict(ds_simSWOT.sizes)}")
        print(f"SWOT dims: {dict(ds_SWOT.sizes)}")
        if 'ds_LLC4320_KE_asf' in locals():
            print(f"LLC4320 asf dims: {ds_LLC4320_KE_asf.dims}, shape: {ds_LLC4320_KE_asf.shape}")
        if 'ds_LLC4320_Kvar_asf' in locals():
            print(f"LLC4320 x dims: {ds_LLC4320_Kvar_asf.dims}, shape: {ds_LLC4320_Kvar_asf.shape}")
        traceback.print_exc()
        plt.close(fig)
        return None
    finally:
        for ds_name, ds in [("LLC4320", ds_LLC4320), ("simSWOT", ds_simSWOT), ("SWOT", ds_SWOT)]:
            try:
                if ds is not None:
                    ds.close()
            except Exception as close_error:
                print(f"Warning: failed to close {ds_name} dataset cleanly: {close_error}")

for season in [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]:
    fig = plot_1x3_allmethods(region_dict, region_idx=0, confint=False, LLL=True, KE=True, KE_ymin=None, KE_ymax=None, Bessel_mean_bug=False, filter_outliers=False, use_km=False, convert_KEFlux_to_wattperm3=False, season=season, simSWOT_fastphase=True)
    if fig is None:
        print(f"Skipping display for season {season} because plotting failed.")
        continue
    plt.show()
    plt.close(fig)