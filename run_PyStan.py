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
    from glob import glob

    stan_files = glob("./Stan_models/*.stan")
    stan_datas = [re.sub(r'.stan$', ".data.R", path) for path in stan_files]

    for stan_model, stan_data in zip(stan_files, stan_datas):
        model_name = os.path.basename(stan_model)
        print(f"\n\n{model_name}\n\n")

        model = t(pystan.StanModel, timing_name=f"pystan.StanModel {model_name}", file=stan_model)
        print(f"model: {model_name}, done", flush=True)

        fit = t(
            model.sampling,
            timing_name=f"{model_name}.sampling",
            data=pystan.read_rdump(stan_data),
            chains=4,
            n_jobs=2 if platform.system() in ("Linus", "Windows") else 1,
            seed=1111,
            warmup=1000,
            iter=2000,
        )

        print(f"fit: {model_name}, done", flush=True)

        timing_df = t(get_timing, fit, timing_name=f"{model_name}: get_timing")
        import arviz as az

        summary_df = t(az.summary, fit, timing_name=f"{model_name}: az.summary")

        savepath_timing = f"./results/PyStan_{model_name}_timing_{platform.system()}.csv"
        savepath_summary = f"./results/PyStan_{model_name}_summary_{platform.system()}.csv"

        os.makedirs("results", exist_ok=True)

        t(timing_df.to_csv, savepath_timing, timing_name=f"{model_name}: timing_df.to_csv")
        t(summary_df.to_csv, savepath_summary, timing_name=f"{model_name}: summary_df.to_csv")

        print(model_name, flush=True)
        print(f"Timing: {model_name}", flush=True)
        print(timing_df, flush=True)
        print(f"Summary: {model_name}", flush=True)
        print(summary_df, flush=True)
    print("\n\nFinished", flush=True)
