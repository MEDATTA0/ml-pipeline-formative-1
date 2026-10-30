"""Scoring helpers shared by the notebooks."""

import numpy as np
import polars as pl


def day_clustered_gain(y, pred, ref, days, mask):
    """Mean squared-error improvement of `pred` over `ref`, with a 95% CI clustered by day.

    Tickers move together on the same day, so samples are averaged per day before the
    standard error is taken.
    """
    gain = (y[mask] - ref[mask]) ** 2 - (y[mask] - pred[mask]) ** 2
    per_day = (
        pl.DataFrame({"day": days[mask], "g": gain})
        .group_by("day")
        .agg(pl.col("g").mean())["g"]
        .to_numpy()
    )
    return float(per_day.mean()), float(
        1.96 * per_day.std(ddof=1) / np.sqrt(len(per_day))
    )
