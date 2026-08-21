"""!
Helper functions for saving and loading the tabulated accepted-photon /
muon data produced by ``extract_accepted_photons.py``.

The data is stored in a single (row-oriented) parquet file.  Muon-level
quantities are repeated on every row belonging to the same event, so an event
can be reconstructed by grouping on ``(run_id, event_id)``.

There are two output modes, each with its own schema.

**Photon mode** (default) -- one row per accepted photon::

    run_id                int64     run id from the I3EventHeader
    event_id              int64     event id from the I3EventHeader
    frame_index           int64     0-based index of the DAQ frame in the file
    time                  float64   arrival time of the accepted photon [ns]
    string                int64     string id of the hit module
    om                    int64     om id of the hit module
    pmt                   int64     hit PMT number (1-16)
    muon_x                float64   muon interaction-vertex position x [m]
    muon_y                float64   muon interaction-vertex position y [m]
    muon_z                float64   muon interaction-vertex position z [m]
    muon_energy           float64   muon (CC daughter) energy [GeV]
    neutrino_energy       float64   primary neutrino energy [GeV]
    zenith                float64   primary neutrino zenith [rad]
    azimuth               float64   primary neutrino azimuth [rad]

**Light-yield mode** -- one row per hit PMT per event, no per-photon times.
The ``time`` column is replaced by an integer ``count`` giving the number of
accepted photons that reached that (string, om, pmt) in the event::

    run_id, event_id, frame_index, string, om, pmt,
    count                 int64     number of accepted photons at this PMT
    muon_x ... azimuth              (as above)

**List mode** -- one row per hit PMT per event, like light-yield mode, but
keeping every photon's arrival time instead of just the count. The ``time``
column is replaced by a ``times`` list column::

    run_id, event_id, frame_index, string, om, pmt,
    times                 list[float64]  arrival times of all accepted
                                         photons at this PMT in the event
    muon_x ... azimuth              (as above)

Use ``DataFrame.explode("times")`` to recover one row per photon.

Both schemas may optionally also carry the optical-module position relative to
the detector centre (``om_x/y/z``) and the hit-PMT direction relative to the
module (``pmt_dir_x/y/z``); these are included whenever the producing module
emits them.
"""

import glob
import os

import pandas as pd
# import pyarrow.parquet as pq


# Preferred column ordering.  Any keys present in the rows are ordered by this
# list (unknown keys are appended in insertion order).  Not every column is
# required -- both output modes are subsets of this ordering.
COLUMN_ORDER = [
    "run_id",
    "event_id",
    "frame_index",
    # photon mode
    "time",
    # light-yield mode
    "count",
    # list mode
    "times",
    "string",
    "om",
    "pmt",
    "om_x",
    "om_y",
    "om_z",
    "pmt_dir_x",
    "pmt_dir_y",
    "pmt_dir_z",
    "muon_x",
    "muon_y",
    "muon_z",
    "muon_energy",
    "neutrino_energy",
    "zenith",
    "azimuth",
]


def rows_to_dataframe(rows):
    """Build a DataFrame from a list of row dicts, ordering the columns.

    Columns present in the rows are ordered according to :data:`COLUMN_ORDER`;
    any keys not in that list are appended afterwards in first-seen order.
    This lets the photon and light-yield modes share one writer despite having
    slightly different schemas.

    Args:
        rows (list[dict]): One dict per output row.

    Returns:
        pandas.DataFrame: DataFrame with a stable, sensible column order.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ordered = [c for c in COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in COLUMN_ORDER]
    return df[ordered + extras]


def save_parquet(rows_or_df, path, compression="snappy"):
    """Save accepted-photon rows to a parquet file.

    Args:
        rows_or_df: Either a list of row dicts or a ready-made DataFrame.
        path (str): Output ``.parquet`` path.
        compression (str): Parquet compression codec. Defaults to ``"snappy"``.
    """
    if isinstance(rows_or_df, pd.DataFrame):
        df = rows_or_df
    else:
        df = rows_to_dataframe(rows_or_df)
    df.to_parquet(path, engine="pyarrow", compression=compression, index=False)


def load_parquet(path):
    """Load an accepted-photon parquet file into a DataFrame.

    Args:
        path (str): Path to the ``.parquet`` file.

    Returns:
        pandas.DataFrame: The tabulated accepted-photon data.
    """
    return pd.read_parquet(path, engine="pyarrow")


# def collect_parquets(folder, output_path, pattern="*.parquet"):
#     """Concatenate every parquet file in a folder into one output parquet file.

#     Files are streamed row-group by row-group through a single
#     ``pyarrow.parquet.ParquetWriter`` rather than being loaded as full
#     DataFrames and concatenated, so peak memory use stays roughly constant
#     regardless of how many input files there are. All matched files must
#     share the same schema (e.g. all produced by the same
#     ``extract_accepted_photons.py`` mode).

#     Args:
#         folder (str): Directory to search for input parquet files.
#         output_path (str): Path of the combined parquet file to write.
#         pattern (str): Glob pattern (relative to ``folder``) selecting which
#             files to collect. Defaults to ``"*.parquet"``.

#     Returns:
#         str: ``output_path``, for convenience.
#     """
#     paths = sorted(glob.glob(os.path.join(folder, pattern)))
#     paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(output_path)]
#     if not paths:
#         raise FileNotFoundError(
#             "No parquet files matching %r found in %s" % (pattern, folder)
#         )

#     writer = None
#     total_rows = 0
#     try:
#         for path in paths:
#             pf = pq.ParquetFile(path)
#             for batch in pf.iter_batches():
#                 if writer is None:
#                     writer = pq.ParquetWriter(output_path, batch.schema)
#                 writer.write_batch(batch)
#                 total_rows += batch.num_rows
#     finally:
#         if writer is not None:
#             writer.close()

#     print(
#         "Collected %d row(s) from %d file(s) into %s"
#         % (total_rows, len(paths), output_path)
#     )
#     return output_path


def iter_events(df):
    """Iterate over the DataFrame grouped by event.

    Args:
        df (pandas.DataFrame): DataFrame as returned by :func:`load_parquet`.

    Yields:
        tuple[(int, int), pandas.DataFrame]: ``((run_id, event_id), sub_df)``
        where ``sub_df`` holds all accepted-photon rows for that event.
    """
    for key, sub_df in df.groupby(["run_id", "event_id"], sort=True):
        yield key, sub_df
