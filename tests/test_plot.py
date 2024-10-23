DESC = '''
Tests for aaviz.plot function

By: AA
'''

from aadata import loader
from aaviz import plot
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import seaborn as sns

class NewPlot:
    def __init__(self, num_plots):
        self.pid = 0
        self.num_cols = ceil(sqrt(num_plots))
        self.num_rows = ceil(sqrt(num_plots))
        self.num_plots = num_plots

    def rows_cols(self):
        return self.num_rows, self.num_cols

    def new(self):
        self.pid += 1
        if self.pid >= self.num_plots:
            raise ValueError('Number of plots exceeds specified number.')
        return (self.pid-1)//self.num_rows, (self.pid-1) % self.num_cols

def main():
    iris = sns.load_dataset('iris')
    geyser = sns.load_dataset('geyser')
    
    newplot = NewPlot(16)
    num_rows, num_cols = newplot.rows_cols()
    fig, gs = plot.initiate_figure(x=5*num_cols, y=4*num_rows, 
                                   gs_nrows=num_rows, gs_ncols=num_cols,
                                   gs_wspace=.3, gs_hspace=.4,
                                   color='tableau10')

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.boxplot', data=iris, 
                      pf_x='species', pf_y='sepal_length',
                      la_title='Boxplot vertical',
                      xt_rotation=25, la_xlabel='')

    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.boxplot', data=iris, 
                      sp_sharey=0,
                      pf_x='species', pf_y='sepal_length',
                      pf_color=plot.get_style('color',1),
                      la_title='sharey',
                      xt_rotation=25, la_xlabel='')
    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.boxplot', data=iris, 
                      pf_orient='h', pf_x='sepal_length', pf_y='species',
                      la_title='Boxplot horizontal',
                      yt_rotation=-65, la_ylabel='')
    dfs = pd.melt(iris,id_vars=['species'], var_name='characteristic', value_name='value')

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.boxplot', data=dfs, 
                      pf_orient='h', 
                      pf_hue='characteristic', 
                      pf_y='species', pf_x='value',
                      ag_xmin=0,
                      hatch=True,
                      lg_title='char',
                      la_title='\\parbox{8cm}{\\center Boxplot horizontal with hue and hatchet; legend title}',
                      yt_pad=5, la_ylabel='')

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.barplot', data=iris, 
                      pf_x='species', pf_y='sepal_length',
                      la_title='\\parbox{8cm}{\\center Barplot vertical with logscale and text on top of the bar}',
                      sp_yscale='log',
                      xt_rotation=25, la_xlabel='')
    max_sepal = iris[['sepal_length', 'species']
                     ].groupby('species').max().reset_index()
    max_sepal['text'] = max_sepal.sepal_length.astype(str)
    max_sepal['x'] = np.arange(max_sepal.shape[0])
    plot.text(ax=ax, data=max_sepal, x='species', y='sepal_length', 
              textcol='sepal_length')

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.barplot', data=iris, 
                      pf_orient='h', pf_x='sepal_length', pf_y='species',
                      pf_color=plot.get_style('color',1),
                      la_title='Barplot horizontal with custom color',
                      yt_labelrotation=-65, la_ylabel='')
    dfs = pd.melt(iris,id_vars=['species'], var_name='characteristic', value_name='value')

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.barplot', data=dfs, 
                      pf_orient='h', 
                      pf_hue='characteristic', 
                      pf_y='species', pf_x='value',
                      ag_xmin=0,
                      hatch=True,
                      la_title='\\parbox{8cm}{\\center Barplot horizontal with hue and hatchet}',
                      yt_pad=5, la_ylabel='')

    ####
    x,y = newplot.new()
    counties = loader.load('usa_county_shapes')
    counties = counties[counties.statefp=='51']
    counties['intensity'] = list(range(counties.shape[0]))
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='gpd.plot', data=counties, 
                      pf_column='intensity',
                      pf_legend=True, pf_legend_kwds={'shrink': 0.28}
                      )

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='gpd.boundary.plot', 
                      pf_facecolor='white', pf_edgecolor='black',
                      data=counties, 
                      )

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.scatterplot', 
                      pf_x='sepal_length', pf_y='sepal_width',
                      pf_hue='species', data=iris, 
                      )

    ## ####
    ## x,y = newplot.new()
    ## X = np.linspace(0, 10, 100)
    ## Y = np.linspace(0, 10, 100)
    ## X, Y = np.meshgrid(X, Y)
    ## Z = np.sin(X) * np.cos(Y)
    ## ax = plot.subplot(fig=fig, grid=gs[x,y], func='contour', 
    ##                   pf_x=X, pf_y=Y, pf_z=Z, pf_mode='meshgrid',
    ##                   pf_cmap='cividis'
    ##                   )

    ####
    x,y = newplot.new()
    data = pd.DataFrame({
        'x': [1,2,3,4],
        'y': [1,2,3,6],
        'z': [1,3,1,3]
        })
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='contour', 
                      data=data,
                      pf_mode='lines',
                      pf_x='x', pf_y='y', pf_z='z',
                      pf_cmap='cividis',
                      cb_drawedges=True,
                      cb_spacing='proportional',
                      cb_labelsize='footnotesize'
                      )

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.countplot', data=iris, 
                      pf_x='species',
                      pf_order=['setosa', 'versicolor', 'virginica', 'test'],
                      la_title='countplot vertical with zero counts',
                      xt_rotation=25, la_xlabel='')

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.histplot', data=iris, 
                      pf_x='species',
                      la_title='histplot vertical',
                      xt_rotation=25, la_xlabel='')

    ####
    x,y = newplot.new()
    ax = plot.subplot(fig=fig, grid=gs[x,y], func='sns.histplot', data=iris, 
                      pf_x='species',
                      pf_stat='percent',
                      la_title='histplot vertical stat=percent',
                      xt_rotation=25, la_xlabel='')

    plot.savefig('test.pdf')
    plot.savefig('test.svg')

if __name__ == '__main__':
    main()
