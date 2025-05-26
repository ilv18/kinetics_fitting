import pandas as pd

def build_raw_data_lookup(combined_df: pd.DataFrame) -> dict:
    lookup = {}
    for name, group in combined_df.groupby("source_file"):
        lookup[name] = {
            "Time (s)_540":  group["Time (s)_540"].values,
            "Intensity (a.u.)_540": group["Intensity (a.u.)_540"].values
        }
    return lookup


# Define visual graph setup as a function to implement;
def annotate_ax(ax,title):
    ax.legend(bbox_to_anchor=(1.01,1), loc='upper left')
    ax.set_title(title)