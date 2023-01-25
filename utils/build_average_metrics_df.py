""" Script to extract the final active-learning metrics from a wandb project.
    The resulting dataframe contains the respective values averaged over the
    runs to a specific dataset and strategy pair.
"""

import wandb
import numpy as np
import pandas as pd
from itertools import groupby
from collections import defaultdict

def build_metrics_df(runs, metric_name):
    # group runs by dataset and strategy
    # i.e. wandb group and job-id
    key = lambda run: (run.group, run.job_type)
    grouped_runs = groupby(sorted(runs, key=key), key=key)

    # build metrics dictionary
    metrics_dict = defaultdict(dict)
    for (d, s), runs in grouped_runs:
        # collect metric value from each run
        runs = tuple(runs)
        ms = [r.summary[metric_name] for r in runs if metric_name in r.summary]
        # check if metric values were collected correctly
        if len(ms) < len(runs):
            print("Metric not found in runs of dataset '%s' and strategy '%s'" % (d, s))
            continue
        # write average metric value to dict
        metrics_dict[d][s] = np.mean(ms)

    # build dataframe
    return pd.DataFrame.from_dict(
        data=metrics_dict,
        orient='index'
    )

if __name__ == '__main__':

    # get runs from which to generate metrics dataframe
    runs = wandb.Api().runs(
        path="active-final-q25-b1000",
        filters={'state': 'finished'}
    )

    metric_name = 'test/entity/weighted avg/Area(F)'
    #metric_name = 'test/token/Area(F)'
    #metric_name = 'test/wss'

    # build all metrics dataframes
    metrics_df = build_metrics_df(runs, metric_name=metric_name)
    print(metrics_df)
    print()

    # convert to latex
    style = metrics_df.style
    style = style.format(precision=4)
    style = style.highlight_max(axis=1, props='textbf:--rwrap;')
    print(style.to_latex(hrules=True))
