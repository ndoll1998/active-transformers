import os
import wandb
import numpy as np
import pandas as pd
from itertools import groupby
from typing import Dict

def wandb_build_metric_series(
    project:str,
    group:str,
    strategy:str,
    metric:str,
    api:wandb.Api
) -> pd.DataFrame:
    # load runs
    runs = api.runs(project, filters={
        'group': group,
        'state': 'finished',
        'config.strategy': strategy
    })

    # group by query_size
    key = lambda r: r.config['query_size']
    grouped_runs = groupby(sorted(runs, key=key), key=key)

    # build pandas series
    return pd.Series({
        group: np.mean([r.summary[metric] for r in runs if metric in r.summary])
        for group, runs in grouped_runs
    })

def wandb_build_metric_table(
    project:str,
    group:str,
    strategies:Dict[str, str],
    metric:str,
    api:wandb.Api
) -> pd.DataFrame:
    return pd.DataFrame({
        name: wandb_build_metric_series(
            project=project,
            group=group,
            strategy=strategy,
            metric=metric,
            api=api
        ) for name, strategy in strategies.items()
    })

if __name__ == '__main__':

    group = "swedish-ner-query-analysis-v2"
    strategies={
        "Entropy-Over-Max": "entropy-over-max",
        "Avg-Entropy": "prediction-entropy",
        "EGL": "egl-sampling",
        "ALPS": "alps"
    } 

    api = wandb.Api()
    # load dataframe from wandb
    df = wandb_build_metric_table(
        project="ndoll/active-transformers-all-data",
        group=group,
        strategies=strategies,
        metric="test/Area(F)",
        api=api
    )

    print(df)

    # plot
    ax = df.plot.line(grid=True, title="Conll2003", figsize=(8, 4))
    # save to disk 
    os.makedirs("plots/query_size", exist_ok=True)
    ax.figure.savefig("plots/query_size/%s.png" % group)
