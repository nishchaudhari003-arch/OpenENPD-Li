from pathlib import Path
import pandas as pd


def load_foo2023_lmc_ph7(csv_path=None):
    """
    Load the Foo et al. 2023 LM-C pH ~7 benchmark dataset.

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Path to the benchmark CSV file. If None, the default repository
        path data/foo2023_lmc_ph7_full.csv is used.

    Returns
    -------
    pandas.DataFrame
        Experimental benchmark rows containing pressure, water flux,
        pH, Li rejection, and Mg rejection.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "foo2023_lmc_ph7_full.csv"

    df = pd.read_csv(csv_path)

    experimental = df[
        ["pressure_bar", "Jw_LMH", "Jw_um_s", "pH", "R_Li", "R_Mg"]
    ].dropna()

    return experimental
