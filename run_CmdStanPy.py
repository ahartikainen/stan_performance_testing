import os
import platform
import re

import cmdstanpy
from cmdstanpy import CmdStanModel, cmdstan_path
from cmdstanpy.utils import rload
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


stan_file = "./Stan_models/F1_Base.stan"
stan_data = "./Stan_models/F1_Base.data.R"

# DEFAULTS


model = CmdStanModel(model_name="my_model", stan_file=stan_file,)
print("model, good")

fit = model.sample(
    data=stan_data,
    chains=1,
    cores=1,
    seed=1234,
    iter_warmup=100,
    iter_sampling=100,
    metric="diag_e",
    show_progress=True,
)

print("fit, done", flush=True)

timing_df = get_timing(fit)
summary_df = fit.summary()

if platform.system() == "Windows":
    import sys

    rtools = sys.argv[1]
    savepath_timing = "./results/CmdStanPy_timing_model_1_{}_RTools_{}.csv".format(
        platform.system(), rtools
    )
    savepath_summary = "./results/CmdStanPy_summary_model_1_{}_RTools_{}.csv".format(
        platform.system(), rtools
    )
else:
    savepath_timing = "./results/CmdStanPy_timing_model_1_{}.csv".format(
        platform.system()
    )
    savepath_summary = "./results/CmdStanPy_summary_model_1_{}.csv".format(
        platform.system()
    )

os.makedirs("results", exist_ok=True)

timing_df.to_csv(savepath_timing)
summary_df.to_csv(savepath_summary)

print("Model 1", flush=True)
print("Timing", flush=True)
print(timing_df, flush=True)
print("Summary", flush=True)
print(summary_df, flush=True)
