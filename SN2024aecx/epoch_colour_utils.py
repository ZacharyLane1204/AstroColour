"""
Helper functions for building per-epoch RGB colour cubes from JWST/HST
exposures listed in an images CSV (a dataframe with columns:
'instrument', 'filter', 'filename').

Shared by colour_image.ipynb (NIRCam) and colour_miri_image.ipynb (MIRI).
Keep this file in the same folder as those notebooks.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


def get_date_obs(file_path):
    """Read DATE-OBS (+ TIME-OBS if present) from the primary header of a FITS file."""
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        date_str = header.get('DATE-OBS')
        time_str = header.get('TIME-OBS')
    if date_str is None:
        raise KeyError(f"No DATE-OBS in primary header of {file_path}")
    if time_str:
        return pd.Timestamp(f"{date_str}T{time_str}")
    return pd.Timestamp(date_str)


def assign_epochs(df, instrument, gap_days=3, verbose=True):
    """
    Return a copy of the rows for `instrument`, with 'date_obs' and 'epoch'
    columns added.

    Epochs are found by sorting exposures by DATE-OBS and starting a new
    epoch whenever the gap to the previous exposure exceeds `gap_days`.
    Rows whose DATE-OBS can't be read are dropped (and reported).
    """
    sub = df[df['instrument'] == instrument].copy()

    dates = []
    for fp in sub['filename']:
        try:
            dates.append(get_date_obs(fp))
        except Exception as e:
            if verbose:
                print(f"  Skipping (no DATE-OBS): {fp} ({e})")
            dates.append(pd.NaT)
    sub['date_obs'] = dates

    n_before = len(sub)
    sub = sub.dropna(subset=['date_obs']).sort_values('date_obs').reset_index(drop=True)
    if verbose and len(sub) < n_before:
        print(f"  Dropped {n_before - len(sub)} row(s) with unreadable DATE-OBS")

    epoch_ids = []
    epoch_id = 0
    prev_date = None
    for d in sub['date_obs']:
        if prev_date is not None and (d - prev_date).days > gap_days:
            epoch_id += 1
        epoch_ids.append(epoch_id)
        prev_date = d
    sub['epoch'] = epoch_ids

    if verbose:
        summary = sub.groupby('epoch')['date_obs'].agg(['min', 'max', 'count'])
        print(f"Found {summary.shape[0]} epoch(s) for {instrument} (gap_days={gap_days}):")
        print(summary)

    return sub


def build_epoch_images(epoch_sub, filter_list, exclude_substrings=None,
                        skip_i2d=True, plot=False, fig_width=6):
    """
    For one epoch's exposures (a dataframe already filtered to one
    instrument + one epoch), median-combine the exposures for each filter
    in `filter_list`, reprojecting everything onto the WCS of the very
    first usable exposure so all filters share pixel registration.

    - skip_i2d: skip any file whose path contains '_i2d.fits' (as the
      original scripts did).
    - exclude_substrings: list of path substrings to skip, for excluding
      specific bad exposures in a given epoch.

    Returns (med_images, master_wcs):
      med_images  -- dict {filter_name: 2D median-combined array}
      master_wcs  -- the WCS all filters were reprojected onto
    """
    import matplotlib.pyplot as plt

    exclude_substrings = exclude_substrings or []
    master_wcs = None
    master_shape = None
    med_images = {}

    for filt in filter_list:
        temp_df = epoch_sub[epoch_sub['filter'] == filt]
        data_list = []

        for _, row in temp_df.iterrows():
            file_path = row['filename']

            if skip_i2d and '_i2d.fits' in file_path:
                continue
            if any(bad in file_path for bad in exclude_substrings):
                continue

            with fits.open(file_path) as hdul:
                data = hdul[1].data.astype(float)
                wcs = WCS(hdul[1].header)

            if master_wcs is None:
                master_shape = data.shape
                reprojected_data = data.copy()
                master_wcs = deepcopy(wcs)
            else:
                reprojected_data, footprint = reproject_interp(
                    (data, wcs), master_wcs, shape_out=master_shape)
                reprojected_data[footprint == 0] = np.nan

            data_list.append(reprojected_data)

        if not data_list:
            print(f"  WARNING: no usable exposures for filter {filt} in this epoch — skipped")
            continue

        med_data = np.nanmedian(data_list, axis=0)
        med_images[filt] = med_data

        if plot:
            plt.figure(figsize=(fig_width, fig_width))
            plt.imshow(med_data, cmap='gray', origin='lower',
                       vmin=np.nanpercentile(med_data, 5),
                       vmax=np.nanpercentile(med_data, 95))
            plt.colorbar()
            plt.title(f"Median {filt}")
            plt.show()

    return med_images, master_wcs


def make_colour_cube(med_images, channel_filters):
    """
    Combine per-filter median images into a [red, green, blue] list for
    AstroColour's RGB.master_plot, using
        channel_filters = {'red': [...filters...], 'green': [...], 'blue': [...]}

    Filters missing from med_images (e.g. skipped for lack of data in this
    epoch) are dropped with a note; multiple filters assigned to the same
    channel are combined with nanmean.
    """
    channels = []
    for ch in ('red', 'green', 'blue'):
        filts = channel_filters[ch]
        arrs = [med_images[f] for f in filts if f in med_images]
        missing = [f for f in filts if f not in med_images]
        if missing:
            print(f"  NOTE: channel '{ch}' missing {missing} for this epoch")
        if not arrs:
            raise ValueError(f"No images available for channel '{ch}' (filters {filts})")
        channels.append(arrs[0] if len(arrs) == 1 else np.nanmean(arrs, axis=0))
    return channels