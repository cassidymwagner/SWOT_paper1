import importlib
import xarray_sf_funcs as xsfuncs
from dask.distributed import Client
import dask
import os
import re
import glob
import warnings
import logging
import gc
import signal
import xarray as xr

warnings.filterwarnings("ignore")

region_dict = {
    'acc': {'lat_north': -53, 'lat_south': -57.5, 'lon_east': 158, 'lon_west': 148, 'reduce_xd_num': 1},
    'nwpacific': {'lat_north': 23, 'lat_south': 19, 'lon_east': 137, 'lon_west': 132, 'reduce_xd_num': 1},
    'capebasin': {'lat_north': -41.01421, 'lat_south': -44.99279, 'lon_east': 16, 'lon_west': 11, 'reduce_xd_num': 1},
    'newcaledonia': {'lat_north': -22, 'lat_south': -26, 'lon_east': 171, 'lon_west': 166, 'reduce_xd_num': 1},
    'nwaustralia': {'lat_north': -11, 'lat_south': -15, 'lon_east': 125, 'lon_west': 120, 'reduce_xd_num': 1},
    'westatlantic': {'lat_north': 38.7, 'lat_south': 32.7, 'lon_east': -73, 'lon_west': -76, 'reduce_xd_num': 1},
    'labradorsea': {'lat_north': 63.79824, 'lat_south': 59.5549, 'lon_east': -58.52856, 'lon_west': -63.61784, 'reduce_xd_num': 1},
    # 'Florida': {'lat_north': 29, 'lat_south': 24, 'lon_east': -77, 'lon_west': -82, 'reduce_xd_num': 1},
    # 'GrandBanks': {'lat_north': 40, 'lat_south': 35, 'lon_east': -45, 'lon_west': -50, 'reduce_xd_num': 1},
    # 'Arbic': {'lat_north': 43, 'lat_south': 27.5, 'lon_east': -40, 'lon_west': -60, 'reduce_xd_num': 1},
    # 'Equator': {'lat_north': 5, 'lat_south': -5, 'lon_east': -10, 'lon_west': -35, 'reduce_xd_num': 1},
    # 'Interior': {'lat_north': 32, 'lat_south': 22, 'lon_east': -30, 'lon_west': -42, 'reduce_xd_num': 1},
    # 'GulfStream': {'southwest': (-80, 25), 'southeast': (-35, 25), 'northwest': (-80, 39), 'northeast': (-35, 49), 'reduce_xd_num': 1}
}



# Configure Dask memory settings
dask.config.set({
    "distributed.worker.memory.target": 0.7,
    "distributed.worker.memory.spill": 0.8,
    "distributed.worker.memory.pause": 0.85,
    "distributed.worker.memory.terminate": 0.95,
    "logging.distributed": "error",
})

def create_client():
    """Helper function to create a new Dask client."""
    return Client(
        n_workers=8,
        threads_per_worker=1,
        memory_limit="40GB",
        silence_logs=logging.WARNING,
    )

def reset_client(client):
    """Safely reset Dask client without crashing on teardown timeouts."""
    try:
        client.shutdown()
    except Exception as shutdown_err:
        print(f"Client shutdown warning (safe to ignore): {shutdown_err}")

    try:
        client.close()
    except Exception as close_err:
        print(f"Client close warning (safe to ignore): {close_err}")

    gc.collect()
    new_client = create_client()
    print("Created a fresh Dask client")
    return new_client

class RunTimeoutError(TimeoutError):
    pass

def _timeout_handler(signum, frame):
    raise RunTimeoutError("Run exceeded timeout window without completion")

def matching_outputs(region_output_dir, suffix, region, cycnum):
    
    return [
        f for f in os.listdir(region_output_dir)
        if (
            f.endswith(".nc")
            and suffix in f
            and f"simSWOT_{region}_cycle_{cycnum}" in f
        )
    ]

# macOS supports SIGALRM (this notebook appears to run on macOS)
signal.signal(signal.SIGALRM, _timeout_handler)

# ---------- Tunables ----------
STALL_TIMEOUT_MIN = 20          # hard timeout per region/cycle attempt
STALL_TIMEOUT_SEC = STALL_TIMEOUT_MIN * 60
MAX_RETRIES = 2                 # retries per region/cycle before skip
run_new = False                 # False => skip existing completed outputs
# -----------------------------

USE_TRY_EXCEPT = True

region_skips = [
    # "acc", "already processed",
    # "nwpacific", "already processed",
    # "capebasin", "already processed",
    # "labradorsea", "too large, redo with fewer cycles",
    # "nwaustralia", "already processed",
    # "westatlantic", "too large, redo with fewer cycles",
    # "newcaledonia", "already processed",
]

def simswot_pipeline(SIMSWOT_PHASE, client, output_dir, run_new, ASF, LLL, CG, scalar, Bessels, timemean_removal_method, cleaned, taper_SF, flip_swath, coarsen):

    try:
        print(f"Starting SWOT L3 pipeline for phase: {SIMSWOT_PHASE}")

        parent_dir = f"/Volumes/Promise Disk/data/simulated_swot/{SIMSWOT_PHASE}"
        glob_pattern = f"{parent_dir}/*.nc"

        file_paths = glob.glob(glob_pattern)
        mean_filepath = f"{parent_dir}/*Expert_Pass*_Mean*.nc"

        cycles = sorted(set(
            re.search(r"Expert_(\d+)", os.path.basename(fp)).group(1).zfill(3)
            for fp in file_paths if re.search(r"Expert_(\d+)", os.path.basename(fp))
        ))

        for idx, cycnum in enumerate(cycles):
            print(f"\nProcessing cycle: {cycnum} of {len(cycles)}")


            for region, params in region_dict.items():
                if region in region_skips:
                    print(f"Skipping region: {region}, {region_skips[region_skips.index(region) + 1]}")
                    continue

                reduce_xd_num = params["reduce_xd_num"]
                lat_north = params["lat_north"]
                lat_south = params["lat_south"]
                lon_east = params["lon_east"]
                lon_west = params["lon_west"]

                # convert negative longitudes to 0..360
                if lon_west < 0:
                    lon_west += 360
                if lon_east < 0:
                    lon_east += 360

                print(f"Processing region: {region} with reduce_xd_num: {reduce_xd_num}")

                region_output_dir = os.path.join(output_dir, region)
                os.makedirs(region_output_dir, exist_ok=True)

                suffix = ""
                suffix += f"_{region}_cycle_{cycnum}"
                suffix += f"_coarsen_to_1div{(48 / coarsen):.1f}deg"
                suffix += f"_timemean_removed_{timemean_removal_method}" if timemean_removal_method else ""
                suffix += "_cleaned" if cleaned else ""
                suffix += "_flipped_swath" if flip_swath else ""
                suffix += f"{scalar}" if scalar else ""
                suffix += "_LLL" if LLL else ""
                suffix += "_Bessels" if Bessels else ""
                suffix += "_tapered" if taper_SF else ""
                suffix += "_CG" if CG else ""
                print(f"Output filename suffix: {suffix}")

                # skip completed
                if not run_new:
                    existing_files = matching_outputs(region_output_dir, suffix, region, cycnum)
                    if existing_files:
                        print(f"Skipping cycle {cycnum} for region {region} as output file already exists: {existing_files[0]}")
                        continue
                print(f"Client dashboard link: {client.dashboard_link}")
                xsfuncs.compute_and_save_simswot(
                            filepath=glob_pattern,
                            mean_filepath=mean_filepath,
                            prepadded_mfdataset=None,
                            expand_cyclemean_num=None,
                            outpath=output_dir,
                            region_name=region,
                            cycle_num=cycnum,
                            lat_north=lat_north,
                            lat_south=lat_south,
                            lon_east=lon_east,
                            lon_west=lon_west,
                            ASF=ASF,
                            LLL=LLL,
                            CG=CG,
                            cleaned=cleaned,
                            reduce_xd_num=reduce_xd_num,
                            scalar=scalar,
                            Bessels=Bessels,
                            timemean_removal_method=timemean_removal_method,
                            taper_SF=taper_SF,
                            flip_swath=flip_swath,
                            coarsen=coarsen,
                        )
    except IndexError as e:
        print(f"Skipping region {region} cycle {cycnum} due to IndexError: {e}.")

            
if __name__ == "__main__":
    client = create_client()

    ASF = True
    LLL = True
    CG = True
    scalar = "q"
    Bessels = True
    timemean_removal_method = "cycle_mean"  # options: None, "cycle_mean", "linear_fit"
    cleaned = False
    taper_SF = True
    flip_swath = False
    coarsen = 5 # coarsen from 1/48 deg to 1/9.6 deg resolution

    SIMSWOT_PHASE = "science_phase" # options: "fast_phase", "science_phase"
    output_dir = f"data/simSWOT/{SIMSWOT_PHASE}/2026-05-07"
    run_new = False

    simswot_pipeline(SIMSWOT_PHASE, client, output_dir, run_new, ASF, LLL, CG, scalar, Bessels, timemean_removal_method, cleaned, taper_SF, flip_swath, coarsen)