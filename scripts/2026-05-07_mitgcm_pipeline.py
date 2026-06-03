import xarray_sf_funcs as xsfuncs
from dask.distributed import Client
import dask
import os
import re
import warnings
import logging
import gc
import signal

warnings.filterwarnings("ignore")

# Configure Dask memory settings
dask.config.set({
    'distributed.worker.memory.target': 0.7,
    'distributed.worker.memory.spill': 0.8,
    'distributed.worker.memory.pause': 0.85,
    'distributed.worker.memory.terminate': 0.95,
    'logging.distributed': 'error'
})

def create_client():
    """Helper function to create a new Dask client"""
    return Client(
        n_workers=8,
        threads_per_worker=1, 
        memory_limit='48GB',
        silence_logs=logging.WARNING
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

# macOS supports SIGALRM (this notebook appears to run on macOS)
signal.signal(signal.SIGALRM, _timeout_handler)


def mitgcm_pipeline(client, region_list, output_dir, run_new, ASF, LLL, CG, scalar, Bessels, timemean_removal_method, taper_SF, coarsen, max_dates):

    output_dir_tmp = output_dir
    dates = [f"{str(year).zfill(4)}{str(month).zfill(2)}{str(day).zfill(2)}" for year in [2011, 2012] for month in range(1, 13) for day in range(1, 32)]
    dates = [date for date in dates if (date <= "20121115") and (date >= "20110913")]

    # Remove invalid dates
    dates.remove("20110931")
    dates.remove("20111131")
    dates.remove("20120431")
    dates.remove("20120631")

    # Remove dates after cutoff, in case already processed
    # dates = [date for date in dates if date >= "20120206"]

    for idx, date in enumerate(dates):
        if max_dates and int(date) > int(max_dates):
            print(f"Reached max_dates limit of {max_dates}. Stopping further processing.")
            break
        print(f"Processing date: {date}")

        for region in region_list:

            output_dir = f"{output_dir_tmp}/{region}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            suffix = ""
            suffix += f"_{region}_{date}"
            suffix += f"_coarsen_to_1div{(48 / coarsen):.1f}deg" if coarsen else ""
            suffix += f"_timemean_removed_{timemean_removal_method}" if timemean_removal_method else ""
            suffix += f"{scalar}" if scalar else ""
            suffix += "_LLL" if LLL else ""
            suffix += "_Bessels" if Bessels else ""
            suffix += "_tapered" if taper_SF else ""
            suffix += "_CG" if CG else ""
            print(f"Output filename suffix: {suffix}")

            # Check for existing files if run_new is False
            if not run_new:
                existing_files = [
                    f for f in os.listdir(output_dir)
                    if suffix in f
                    and re.search(rf"{date}(?:\b|[^a-zA-Z0-9])", f)
                ]
                if existing_files:
                    print(f"Skipping date {date} for region {region} as output file already exists: {existing_files[0]}")
                    continue

            reduce_xd_num = 1

            child_dir = f"{region}_all"

            # print link to dask dashboard
            print(f"Dask dashboard link: {client.dashboard_link}")
            try:
                xsfuncs.compute_and_save_llc4320(
                    region_name=region,
                    date=date,
                    outpath=output_dir,
                    child_dir=child_dir,
                    ASF=ASF,
                    CG=CG,
                    reduce_xd_num=reduce_xd_num,
                    scalar=scalar,
                    Bessels=Bessels,
                    LLL=LLL,
                    timemean_removal_method=timemean_removal_method,
                    taper_SF=taper_SF,
                    coarsen=coarsen
                )
                # client.run(gc.collect)

            except Exception as e:
                print(f"Failed for date {date} and region {region}: {e}")
                print("Skipping this date and region  retry failure. See logs for details.")
                with open("processing_errors.log", "a") as log_file:
                    log_file.write(f"Failed to process date {date} for region {region}. Error details: {e}\n")

        # Clean up at the end
        # client.close()
        client = reset_client(client)

if __name__ == "__main__":
    client = create_client()

    ASF = True
    LLL = True
    CG = True
    scalar = None
    Bessels = True
    timemean_removal_method = "snapshot_mean"  # options: None, "snapshot_mean"
    taper_SF = True
    coarsen = None # coarsen from 1/48 deg to 1/9.6 deg resolution

    region_list = [
        'acc', 
        'nwpacific', 
        'capebasin', 
        'labradorsea', 
        'newcaledonia', 
        'nwaustralia', 
        'westatlantic',
        ]

    output_dir = "data/MITgcm/2026-06-01"
    run_new = False
    max_dates = False # set to a string representing an integer to limit number of dates processed for testing

    mitgcm_pipeline(client, region_list, output_dir, run_new, ASF, LLL, CG, scalar, Bessels, timemean_removal_method, taper_SF, coarsen, max_dates)