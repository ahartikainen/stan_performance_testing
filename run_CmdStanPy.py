import os
import platform
import re
from time import time

import cmdstanpy
from cmdstanpy import CmdStanModel, cmdstan_path
import pandas as pd


def get_timing(fit):
    """Extract timing."""
    timings = []
    for i, path in enumerate(fit.runset.stdout_files):
        with open(path) as f:
            timing = ""
            add_timing = False
            for line in f:
                if "Elapsed Time" in line:
                    add_timing = True
                if add_timing:
                    timing += "\n" + line.strip()
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

    # DEFAULTS

    for stan_model, stan_data in zip(stan_files, stan_datas):
        model_name = os.path.basename(stan_model)
        print(f"\n\n{model_name}\n\n")

        model = t(
            CmdStanModel,
            model_name=model_name,
            stan_file=stan_model,
            timing_name=f"CmdStanModel {model_name}",
        )

        print(f"model: {model_name}, done", flush=True)

        fit = t(
            model.sample,
            timing_name=f"{model_name}.sample",
            data=stan_data,
            chains=4,
            cores=2,
            seed=1111,
            iter_warmup=1000,
            iter_sampling=1000,
            metric="diag_e",
            show_progress=True,
        )

        print(f"fit: {model_name}, done", flush=True)

        timing_df = t(get_timing, fit, timing_name=f"{model_name}: get_timing")
        import arviz as az

        summary_df = t(az.summary, fit, timing_name=f"{model_name}: az.summary")

        if platform.system() == "Windows":
            import sys

            rtools = sys.argv[1]
            savepath_timing = f"./results/CmdStanPy_{model_name}_timing_{platform.system()}_RTools_{rtools}.csv"
            savepath_summary = f"./results/CmdStanPy_{model_name}_summary_{platform.system()}_RTools_{rtools}.csv"
        else:
            savepath_timing = f"./results/CmdStanPy_{model_name}_timing_{platform.system()}.csv"
            savepath_summary = f"./results/CmdStanPy_{model_name}_summary_{platform.system()}.csv"

        os.makedirs("results", exist_ok=True)

        t(timing_df.to_csv, savepath_timing, timing_name=f"{model_name}: timing_df.to_csv")
        t(summary_df.to_csv, savepath_summary, timing_name=f"{model_name}: summary_df.to_csv")

        print(model_name, flush=True)
        print(f"Timing: {model_name}", flush=True)
        print(timing_df, flush=True)
        print(f"Summary: {model_name}", flush=True)
        print(summary_df, flush=True)
    print("\n\nFinished", flush=True)
