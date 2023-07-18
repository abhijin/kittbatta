DESC = '''
Helper functions for data preparation.

By AA
'''

import numpy as np
import pandas as pd
from pdb import set_trace

# Fill zeros: Staple for time series plots. 
### df: A time series dataframe consists of multiple time-series specified by
### different columns but a single value column.
### x_col: like time.
### y_cols: y-axis columns (like states and cascades).
### value_col: The column containing values.
def fill_zeros(df=None, x_col=None, x_min=0, x_max=None, y_cols=None,
        value_col=None, out_y_cols=None):

    if x_max == None:
        x_max = df[x_col].max()
    xvals = pd.Series(index=np.arange(x_min,x_max+1), data=True, name='xmap')

    if type(y_cols) != list:
        y_cols = [y_cols]

    # Make each series a column with x_col as index.
    if y_cols != None:
        df = df[[x_col]+y_cols+[value_col]].pivot_table(index=x_col, 
                columns=y_cols, fill_value=0)
        df_columns = df.columns

    # Fill zeros
    filled_df = df.merge(xvals.to_frame(), how='right', left_index=True, 
            right_index=True).fillna(0).drop('xmap', axis=1)

    if y_cols != None:
        if out_y_cols is None:
            out_y_cols = y_cols
        filled_df.columns = df_columns
        filled_df = filled_df.stack(level=out_y_cols).reset_index().rename(
                columns={'level_0': x_col})

    return filled_df

