import os
import platform
import re

import arviz as az
import pystan
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
stan_data = pystan.read_rdump("./Stan_models/F1_Base.data.R")

# DEFAULTS


model = pystan.StanModel(file=stan_file)
print("model, good")

fit = model.sampling(
    data=stan_data,
    chains=4,
    n_jobs=2,
    seed=1111,
    warmup=100,
    iter=200,
)

print("fit, done", flush=True)

#timing_df = get_timing(fit)
summary_df = az.summary(fit)

savepath_timing = "./results/PyStan_timing_model_1_{}.csv".format(
    platform.system()
)
savepath_summary = "./results/PyStan_summary_model_1_{}.csv".format(
    platform.system()
)

os.makedirs("results", exist_ok=True)

#timing_df.to_csv(savepath_timing)
summary_df.to_csv(savepath_summary)

print("Model 1", flush=True)
#print("Timing", flush=True)
#print(timing_df, flush=True)
print("Summary", flush=True)
print(summary_df, flush=True)
