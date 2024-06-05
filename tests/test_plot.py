DESC = '''
Tests for aaviz.plot function

By: AA
'''

from aaviz import plot
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
import pandas as pd
from pdb import set_trace
import seaborn as sns

def main():
    iris = sns.load_dataset('iris')
    geyser = sns.load_dataset('geyser')
    
    num_plots = 6
    num_cols = ceil(sqrt(num_plots))
    num_rows = ceil(sqrt(num_plots))
    fig, gs = plot.initiate_figure(x=5*num_cols, y=4*num_rows, 
                                   gs_nrows=num_rows, gs_ncols=num_cols,
                                   gs_wspace=.3, gs_hspace=.4)

    ####
    id = 1; x = (id-1)//num_rows; y = (id-1) % num_cols
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.boxplot', data=iris, 
                      pf_x='species', pf_y='sepal_length',
                      la_title='Boxplot vertical',
                      xt_rotation=25, la_xlabel='')

    ####
    id = 2; x = (id-1)//num_cols; y = (id-1) % num_rows
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.boxplot', data=iris, 
                      pf_orient='h', pf_x='sepal_length', pf_y='species',
                      la_title='Boxplot horizontal',
                      yt_rotation=-65, la_ylabel='')
    dfs = pd.melt(iris,id_vars=['species'], var_name='characteristic', value_name='value')

    ####
    id = 3; x = (id-1)//num_cols; y = (id-1) % num_rows
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.boxplot', data=dfs, 
                      pf_orient='h', 
                      pf_hue='characteristic', 
                      pf_y='species', pf_x='value',
                      ag_xmin=0,
                      hatch=True,
                      la_title='\\parbox{8cm}{\\center Boxplot horizontal with hue and hatchet}',
                      yt_pad=5, la_ylabel='')

    ####
    id = 4; x = (id-1)//num_cols; y = (id-1) % num_rows
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.barplot', data=iris, 
                      pf_x='species', pf_y='sepal_length',
                      la_title='Barplot vertical',
                      xt_rotation=25, la_xlabel='')

    ####
    id = 5; x = (id-1)//num_cols; y = (id-1) % num_rows
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.barplot', data=iris, 
                      pf_orient='h', pf_x='sepal_length', pf_y='species',
                      pf_color=plot.get_style('color',1),
                      la_title='Barplot horizontal with custom color',
                      yt_labelrotation=-65, la_ylabel='')
    dfs = pd.melt(iris,id_vars=['species'], var_name='characteristic', value_name='value')

    ####
    id = 6; x = (id-1)//num_cols; y = (id-1) % num_rows
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.barplot', data=dfs, 
                      pf_orient='h', 
                      pf_hue='characteristic', 
                      pf_y='species', pf_x='value',
                      ag_xmin=0,
                      hatch=True,
                      la_title='\\parbox{8cm}{\\center Barplot horizontal with hue and hatchet}',
                      yt_pad=5, la_ylabel='')
    plot.savefig('test.pdf')
    plot.savefig('test.svg')

if __name__ == '__main__':
    main()
