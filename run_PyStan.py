import os
import platform
import re
from time import time

import pystan
import pandas as pd


def get_timing(fit):
    """Extract timing.

    PyStan source code hacked.
    """
    timings = []
    for i, timing in enumerate(fit.get_adaptation_info()):
        chain_timing = dict(
            zip(
                ("warmup", "sampling", "total"),
                map(float, re.findall(r"\s*(\d+.\d*)\sseconds\s", timing)),
            )
        )
        chain_timing["chain"] = i
        timings.append(chain_timing)
    return pd.DataFrame(timings)


def t(func, *args, timing_name=None, **kwargs):
    """Time function."""
    start_time = time()
    res = func(*args, **kwargs)
    duration = time() - start_time
    duration_unit = "seconds"
    if duration > (60 * 60):
        duration /= 60 * 60
        duration_unit = "hours"
    elif duration > (60 * 3):
        duration /= 60
        duration_unit = "minutes"
    print(
        f"{timing_name + ': ' if timing_name is not None else ''}Duration",
        "{duration:.1f}",
        duration_unit,
        flush=True,
    )
    return res


if __name__ == "__main__":
    stan_file = "./Stan_models/F1_Base.stan"
    stan_data = pystan.read_rdump("./Stan_models/F1_Base.data.R")

    model = t(pystan.StanModel, timing_name="pystan.StanModel", file=stan_file)
    print("model, done", flush=True)

    fit = t(
        model.sampling,
        timing_name="model.sampling",
        data=stan_data,
        chains=4,
        n_jobs=2 if platform.system() in ("Linus", "Windows") else 1,
        seed=1111,
        warmup=1000,
        iter=2000,
    )

    print("fit, done", flush=True)

    timing_df = t(get_timing, fit, timing_name="get_timing")
    import arviz as az

    summary_df = t(az.summary, fit, timing_name="az.summary")

    savepath_timing = "./results/PyStan_timing_model_1_{}.csv".format(platform.system())
    savepath_summary = "./results/PyStan_summary_model_1_{}.csv".format(
        platform.system()
    )

    os.makedirs("results", exist_ok=True)

    t(timing_df.to_csv, savepath_timing, timing_name="timing_df.to_csv")
    t(summary_df.to_csv, savepath_summary, timing_name="summary_df.to_csv")

    print("Model 1", flush=True)
    print("Timing", flush=True)
    print(timing_df, flush=True)
    print("Summary", flush=True)
    print(summary_df, flush=True)
