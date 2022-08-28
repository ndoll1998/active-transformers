import wandb
import numpy as np
import pandas as pd
from itertools import groupby
from typing import Dict

def wandb_build_metric_series(
    project:str,
    group:str,
    metric:str,
    api:wandb.Api
) -> pd.DataFrame:
    # load runs
    runs = api.runs(project, filters={
        'group': group,
        'state': 'finished'
    })

    # group by strategy
    key = lambda r: r.config['strategy']
    grouped_runs = groupby(sorted(runs, key=key), key=key)

    # build pandas series
    return pd.Series({
        group: np.mean([r.summary[metric] for r in runs if metric in r.summary])
        for group, runs in grouped_runs
    })

def wandb_build_metric_table(
    project:str,
    groups:Dict[str, str],
    metric:str,
    api:wandb.Api
) -> pd.DataFrame:
    # build table comparing metric between strategies on different groups/datasets    
    return pd.DataFrame({
        name: wandb_build_metric_series(
            project=project,
            group=group,
            metric=metric,
            api=api
        ) for name, group in groups.items()
    })

if __name__ == '__main__': 

    api = wandb.Api()
    # build metrics dataframe
    df = wandb_build_metric_table(
        project="ndoll/active-transformers-all-data",
        groups={
            "conll2003": "conll2003-fixed-preprocessing",
            "swedish-ner": "swedish-ner-fixed-preprocessing",
            "ncbi": "ncbi-fixed-preprocessing",
        },
        metric="test/Area(F)",
        # metric="test/wss",
        api=api
    )

    # get dataframe style member
    # and highlight maximal values per row
    style = df.style
    style = style.highlight_max(axis=0, props='textbf:--rwrap;')
    # render to latex
    latex = style.to_latex(hrules=True)
    
    print()
    print(latex)

