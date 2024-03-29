DESC = '''
Tests for aaviz.plot function

By: AA
'''

from aaviz import plot
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt
from pdb import set_trace
import seaborn as sns

def main():
    data = sns.load_dataset('iris')
    
    num_plots = 2
    num_cols = ceil(sqrt(num_plots))
    num_rows = floor(sqrt(num_plots))
    fig, gs = plot.initiate_figure(x=5*num_cols, y=4*num_rows, 
                                   gs_nrows=num_rows, gs_ncols=num_cols)
    ax = plot.subplot(fig=fig, grid=gs[0,0], func='sns.boxplot', data=data, 
                      pf_x='species', pf_y='sepal_length',
                      xt_rotation=25, xt_ha='center', la_xlabel='')
    ax = plot.subplot(fig=fig, grid=gs[0,1], func='sns.boxplot', data=data, 
                      pf_orient='h', pf_x='sepal_length', pf_y='species',
                      yt_rotation=-65, yt_ha='right', la_ylabel='')
    plt.savefig('test.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
