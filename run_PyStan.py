import os
import platform
import re

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


if __name__ == "__main__":

    stan_file = "./Stan_models/F1_Base.stan"
    stan_data = pystan.read_rdump("./Stan_models/F1_Base.data.R")

    # DEFAULTS

    model = pystan.StanModel(file=stan_file)
    print("model, good")

    fit = model.sampling(
        data=stan_data, chains=4, n_jobs=2, seed=1111, warmup=100, iter=200,
    )

    print("fit, done", flush=True)

    timing_df = get_timing(fit)
    import arviz as az
    summary_df = az.summary(fit)

    savepath_timing = "./results/PyStan_timing_model_1_{}.csv".format(platform.system())
    savepath_summary = "./results/PyStan_summary_model_1_{}.csv".format(
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
