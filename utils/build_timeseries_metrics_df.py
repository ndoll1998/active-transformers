""" Script to gather the timeseries data of runs to a specific
    dataset and strategy pair from a wandb project. Data is stored
    in a dataframe
"""

import os
import wandb
import pandas as pd
from itertools import groupby

def gather_timeseries_df(runs, metric_name):

    df = pd.DataFrame()
    for run in runs:
        try:
            # get history data of metric        
            h = run.history(keys=[metric_name], x_axis='_step')
            h.set_index("_step", inplace=True)
            # add to dataframe
            df[run.config['seed']] = h[metric_name]
        except KeyError:
            print("KeyError in run %s/%s/%s" % (run.group, run.job_type, run.name))

    return df

if __name__ == '__main__':

    project_name = "active-final-q25-b1000"
    metric_name = "test/entity/weighted avg/F"
    # get all finished runs of a specific project
    project = wandb.Api().runs(
        path="active-final-q25-b1000",
        filters={'state': 'finished'}
    )

    # group runs by dataset (i.e. group) and strategy (i.e. job-type)
    key = lambda run: (run.group, run.job_type)
    grouped_runs = groupby(sorted(project, key=key), key=key)

    for (d, s), runs in grouped_runs:
        # create dump directory and preprocess metric name
        os.makedirs("al_results/%s/%s" % (project_name, d), exist_ok=True)
        m = metric_name.replace("/", ".").replace(" ", "_")
        # gather timeseries dataframe and save it to disk
        df = gather_timeseries_df(runs, metric_name=metric_name)
        if len(df.index) > 0:
            df.to_csv("al_results/%s/%s/%s-%s.csv" % (project_name, d, s, m))
