import numpy as np
import pandas as pd

# Normalization (Min-Max)
def normalize_column(series):
    s_min = series.min()
    s_max = series.max()

    if s_max - s_min == 0:
        return series * 0

    return (series - s_min) / (s_max - s_min)
