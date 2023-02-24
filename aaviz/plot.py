DESC='''plot functions
By AA
'''

from cycler import cycler
import geopandas as gpd
from itertools import product
import logging
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub, search, match
import seaborn as sns
from shapely.geometry import Point

# contextily EPSG 3857

COLORS = {
        'mathematica': ['#5e82b5','#e09c24','#8fb030','#eb634f','#8778b3','#c46e1a','#5c9ec7','#fdbf6f'],
        'grand_budapest': ['#5b1a18','#fd6467','#f1bb7b','#d67236'],
        'red_blue': ['#0060ad', '#dd181f']
        }
SNS_AXIS_PLOTS = ['sns.lineplot', 'sns.barplot', 'sns.histplot', 'sns.ecdfplot',
        'sns.boxplot']
AXIS_NORMAL = ['sns.lineplot']
AXIS_HIST = ['sns.boxplot', 'sns.barplot']

NON_FUNC_PARAMS = ['fig', 'subplot', 'title', 'xlabel', 'ylabel', 'data']

RC_PARAMS = {
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        'legend.frameon': False,
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'text.usetex': True,
        'axes.prop_cycle': cycler(color=COLORS['mathematica'])
        }

# latex fontsizes
FONT_TABLE=pd.DataFrame({9: {'miniscule': 4, 'tiny': 5, 'scriptsize': 6, 'footnotesize': 7, 'small': 8, 'normalsize': 9, 'large': 10, 'Large': 11, 'LARGE': 12, 'huge': 14, 'Huge': 17, 'HUGE': 20}, 10: {'miniscule': 5, 'tiny': 6, 'scriptsize': 7, 'footnotesize': 8, 'small': 9, 'normalsize': 10, 'large': 11, 'Large': 12, 'LARGE': 14, 'huge': 17, 'Huge': 20, 'HUGE': 25}, 11: {'miniscule': 6, 'tiny': 7, 'scriptsize': 8, 'footnotesize': 9, 'small': 10, 'normalsize': 11, 'large': 12, 'Large': 14, 'LARGE': 17, 'huge': 20, 'Huge': 25, 'HUGE': 30}, 12: {'miniscule': 7, 'tiny': 8, 'scriptsize': 9, 'footnotesize': 10, 'small': 11, 'normalsize': 12, 'large': 14, 'Large': 17, 'LARGE': 20, 'huge': 25, 'Huge': 30, 'HUGE': 36}, 14: {'miniscule': 8, 'tiny': 9, 'scriptsize': 10, 'footnotesize': 11, 'small': 12, 'normalsize': 14, 'large': 17, 'Large': 20, 'LARGE': 25, 'huge': 30, 'Huge': 36, 'HUGE': 48}, 17: {'miniscule': 9, 'tiny': 10, 'scriptsize': 11, 'footnotesize': 12, 'small': 14, 'normalsize': 17, 'large': 20, 'Large': 25, 'LARGE': 30, 'huge': 36, 'Huge': 48, 'HUGE': 60}, 20: {'miniscule': 10, 'tiny': 11, 'scriptsize': 12, 'footnotesize': 14, 'small': 17, 'normalsize': 20, 'large': 25, 'Large': 30, 'LARGE': 36, 'huge': 48, 'Huge': 60, 'HUGE': 72}, 25: {'miniscule': 11, 'tiny': 12, 'scriptsize': 14, 'footnotesize': 17, 'small': 20, 'normalsize': 25, 'large': 30, 'Large': 36, 'LARGE': 48, 'huge': 60, 'Huge': 72, 'HUGE': 84}, 30: {'miniscule': 12, 'tiny': 14, 'scriptsize': 17, 'footnotesize': 20, 'small': 25, 'normalsize': 30, 'large': 36, 'Large': 48, 'LARGE': 60, 'huge': 72, 'Huge': 84, 'HUGE': 96}, 36: {'miniscule': 14, 'tiny': 17, 'scriptsize': 20, 'footnotesize': 25, 'small': 30, 'normalsize': 36, 'large': 48, 'Large': 60, 'LARGE': 72, 'huge': 84, 'Huge': 96, 'HUGE': 108}, 48: {'miniscule': 17, 'tiny': 20, 'scriptsize': 25, 'footnotesize': 30, 'small': 36, 'normalsize': 48, 'large': 60, 'Large': 72, 'LARGE': 84, 'huge': 96, 'Huge': 108, 'HUGE': 120}, 60: {'miniscule': 20, 'tiny': 25, 'scriptsize': 30, 'footnotesize': 36, 'small': 48, 'normalsize': 60, 'large': 72, 'Large': 84, 'LARGE': 96, 'huge': 108, 'Huge': 120, 'HUGE': 132}})

AXES_COLOR='#888888'

FIGSIZE=(8,6)
LINESTYLES=['solid','dashed']
LINEWIDTH=[1]
MARKERS=['.','o','v','^','s']
# LINE_PROPS=pd.DataFrame(product(['solid','dashed'],MARKERS,MATHEMATICA[0:7],LINEWIDTH),columns=['style','marker','color','width'])

## PLOT_FUNCTIONS = [
##         'gpd.plot'
##         }

pd.options.display.float_format = '{:.10g}'.format

# The helper functions are arranged in the order in which they should be called
def initiate_figure(**kwargs):
    argval = {
            'mode': 'figure',
            'subplot_mode': 'subplots',
            'x': FIGSIZE[0],
            'y': FIGSIZE[1],
            'gs_wspace': 0.005,
            'gs_hspace': 0.005,
            'sp_nrows': 1,
            'sp_ncols': 1,
            }

    for k in kwargs.keys():
        argval[k] = kwargs[k] 

    # Setting font families and latex stuff
    for k,v in RC_PARAMS.items():
        rcParams[k] = v

    # Figure
    if argval['mode'] == 'figure':
        if argval['subplot_mode'] == 'gridspec':
            fig = plt.figure(figsize=[argval['x'],argval['y']])
            subplot_args = {k[3:]: v for k,v in argval.items() if k[0:3] == 'gs_'}
            gs = fig.add_gridspec(subplot_args['nrows'], subplot_args['ncols'])
            gs.update(wspace=subplot_args['wspace'], hspace=subplot_args['hspace']) # set the spacing between axes.
            return fig, gs
        elif argval['subplot_mode'] == 'subplots':
            subplot_args = {k[3:]: v for k,v in argval.items() if k[0:3] == 'sp_'}
            fig, ax = plt.subplots(**subplot_args, figsize=[argval['x'],argval['y']])
            if type(ax) != np.ndarray:
                return fig, [ax]
            else:
                return fig, ax
    elif mode == 'facetgrid':
        return  # currently, nothing to return

    # ax = fig.add_subplot(layout)

# All non-plot-function variables will be prefixed by "n_"
def subplot(**kwargs):

    ax = kwargs['ax']
    func = kwargs['func']

    # Collect all arguments that pertain to func
    for k in kwargs.keys():
        if k in ['fig', 'ax', 'func', 'data']:
            continue
        pat = search('[^_]*_', k)[0]
        if pat not in ['pf_', 'ag_', 'la_', 'fs_', 'xt_', 'yt_', 'lg_']:
            raise ValueError(f'Unsupported input class "{k}".')
    plot_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'pf_'}
    axes_grid_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'ag_'}
    label_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'la_'}
    fontsize_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'fs_'}
    xtick_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'xt_'}
    ytick_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'yt_'}
    legend_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'lg_'}

    # Decide function
    plot_args['data'] = kwargs['data']
    funcobj = subplot_func(plot_args=plot_args, **kwargs)

    funcobj(**plot_args)

    subplot_axes_grid(func=func, ax=ax, plot_args=plot_args, **axes_grid_args)
    subplot_labels(ax=ax, **label_args)
    subplot_fonts(ax=ax, func=func, **fontsize_args)
    
    if xtick_args:
        ax.set_xticklabels(ax.get_xticklabels(), **xtick_args)
    if ytick_args:
        ax.set_yticklabels(ax.get_yticklabels(), **ytick_args)
    # Only if a legend is present and there are some modifications to be made
    if len(ax.get_legend_handles_labels()[0]) and legend_args:  
        if 'visible' in legend_args.keys():
            if not legend_args['visible']:
                ax.legend().set_visible(False)
            else:
                legend_args.pop('visible')
                ax.legend(**legend_args)

    return rcParams

def subplot_func(**kwargs):
    kwargs['plot_args']['data'] = kwargs['data']
    if kwargs['func'] in ['gpd.plot', 'gpd.boundary.plot']:
        kwargs['plot_args'].pop('data')
        return eval(f'kwargs["data"].plot')
    elif kwargs['func'] in SNS_AXIS_PLOTS:
        return eval(kwargs["func"])
    else:
        raise ValueError(f'Unsupported function type {kwargs["func"]}.')

def subplot_axes_grid(**kwargs):
    if 'type' in kwargs.keys():
        axis_type = kwargs['axis_type']
    elif kwargs['func'] in ['gpd.plot', 'gpd.boundary.plot']:
        axis_type = 'none'
    elif kwargs['func'] in AXIS_NORMAL:
        axis_type = 'normal'
    elif kwargs['func'] in AXIS_HIST:
        if 'orient' in kwargs['plot_args']:
            if kwargs['plot_args']['orient'] == 'h':
                axis_type = 'histx'
        else:
            axis_type = 'histy'
    else:
        raise KeyError('Unable to assign axis_type. Check if plot is supported.')

    ax = kwargs['ax']
    if axis_type == 'normal':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_color(AXES_COLOR)
        ax.spines['left'].set_color(AXES_COLOR)
        ax.grid(color='#cccccc',which='major',linewidth=1)
        ax.grid(True,color='#dddddd',which='minor',linewidth=.5)
        set_minor_tics(ax)
    elif axis_type == 'histy':
        ax.grid(axis='y')
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.spines['bottom'].set_color(AXES_COLOR)
        ax.tick_params(axis='y', length=0)
        ax.set_xticks(ax.get_yticks(), minor=False)
    elif axis_type == 'histx':
        ax.grid(axis='x')
        ax.spines[['bottom', 'right', 'top']].set_visible(False)
        ax.spines['left'].set_color(AXES_COLOR)
        ax.tick_params(axis='x', length=0)
        ax.set_xticks(ax.get_xticks(), minor=False)
    elif axis_type == 'none':
        ax.spines[['bottom', 'right', 'top', 'left']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        raise ValueError(f'Unsupported grid type "{axis_type}".')
    ax.set_axisbelow(True)

    # Scale
    if 'xscale' in kwargs.keys():
        ax.set_xscale(kwargs['xscale'])
    if 'yscale' in kwargs.keys():
        ax.set_xscale(kwargs['yscale'])

    # Axes limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if 'xmin' in kwargs.keys():
        xmin = kwargs['xmin']
    if 'xmax' in kwargs.keys():
        xmax = kwargs['xmax']
    if 'ymin' in kwargs.keys():
        ymin = kwargs['ymin']
    if 'ymax' in kwargs.keys():
        ymax = kwargs['ymax']
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return

def subplot_labels(**kwargs):
    ax = kwargs['ax']
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'])
    return

def subplot_fonts(**kwargs):
    ax = kwargs['ax']

    argvals = {
            'fontsize': 17,
            'title': 'large',
            'xlabel': 'normalsize',
            'ylabel': 'normalsize',
            'xtick': 'small',
            'ytick': 'small',
            'legend': 'normalsize',
            'colorbar': 'normalsize'
            }

    for k,v in kwargs.items():
        argvals[k] = v

    font_set = FONT_TABLE[argvals['fontsize']]

    ax.set_title(ax.get_title(), fontsize=font_set[argvals['title']])
    ax.set_xlabel(ax.get_xlabel(), 
            fontsize=font_set[argvals['xlabel']])
    ax.set_ylabel(ax.get_ylabel(), 
            fontsize=font_set[argvals['ylabel']])
    ax.tick_params(axis='x', which='major', 
            labelsize=font_set[argvals['xtick']])
    ax.tick_params(axis='y', which='major', 
            labelsize=font_set[argvals['ytick']])

    if len(ax.get_legend_handles_labels()[0]):  # only if a legend is present
        ax.legend(fontsize=font_set[argvals['legend']])

    # color bar
    if kwargs['func'] in ['gpd.plot']:
        try:
            # Note that axes is arranged the following way: 
            # (ax1, cbar1, ax2, cbar2, ...)
            cbar = ax.figure.axes[-1]
            cbar.tick_params(labelsize=font_set[argvals['colorbar']])
        except Exception as err:
            logging.warning(f'Some error related to colorbar: {err}')
            pass
    return









def set_axes_grid(figobj, axis_type='normal'):
    if type(figobj) == sns.axisgrid.FacetGrid:
        if axis_type == 'normal':
            pass
        elif axis_type == 'histy':
            figobj.despine(left=True)
        else:
            raise ValueError(f'Unsupported grid type "{axis_type}".')
    return

def set_labels(ax=None, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return

def set_fonts(ax=None, global_font_size=None, **kwargs):
    font_set = FONT_TABLE[global_font_size]

    if 'title' in kwargs.keys():
        ax.set_title(ax.get_title(), fontsize=font_set[kwargs['title']])
    else:
        ax.set_title(ax.get_title(), fontsize=font_set['large'])
    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_set[kwargs['xlabel']])
    else:
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_set['normalsize'])
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_set[kwargs['ylabel']])
    else:
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_set['normalsize'])
    if 'xtick' in kwargs.keys():
        ax.tick_params(axis='x', which='major', 
                labelsize=font_set[kwargs['xtick']])
    else:
        ax.tick_params(axis='x', which='major', labelsize=font_set['small'])
    if 'ytick' in kwargs.keys():
        ax.tick_params(axis='y', which='major', 
                labelsize=font_set[kwargs['ytick']])
    else:
        ax.tick_params(axis='y', which='major', labelsize=font_set['small'])
    return

def texify(string):
    string=sub('_','\_',string)
    string=sub('%','\%',string)
    return string

def set_plot_at_zero(axis):
    #axis.set_facecolor('#eeeeee')
    axis.spines['bottom'].set_position('zero')
    return

def set_minor_tics(axis):
    axis.minorticks_on()
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    return

def square_grid_cells_by_x(axis, num_cells_x, num_cells_y, 
        labels_step_x, labels_step_y, type_x = None, type_y = None):

    # set xticks
    xmin, xmax = axis.get_xlim()
    xticks = np.linspace(xmin, xmax, num=num_cells_x+1)
    xtick_labels = [None] * len(xticks)

    # check if all 
    for i in range(0,len(xticks),labels_step_x):
        if type_x:
            xtick_labels[i] = type_x(xticks[i])
        else:
            xtick_labels[i] = xticks[i]
    axis.set_xticks(xticks, labels=xtick_labels)

    # set yticks
    ymin, ymax = axis.get_ylim()
    yticks = np.linspace(ymin, ymax, num=num_cells_y+1)
    ytick_labels = [None] * len(yticks)
    for i in range(0,len(yticks),labels_step_y):
        if type_y:
            ytick_labels[i] = type_y(yticks[i])
        else:
            ytick_labels[i] = yticks[i]
    axis.set_yticks(yticks, labels=ytick_labels)

    # set aspect
    axis.set_aspect((xmax-xmin)/(ymax-ymin)*num_cells_y/num_cells_x)

    # grid needs to be redrawn
    axis.grid()

    return
    
def set_scientific(axis, xy):
    scientific_formatter = FuncFormatter(_scientific)
    if xy == 'y':
        axis.yaxis.set_major_formatter(scientific_formatter)
    else:
        axis.xaxis.set_major_formatter(scientific_formatter)
    return

def _scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return '%.1E' % x

def coords_to_geom(lat, lon, crs=None):
    gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(lon, lat)])
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(**crs)
    return gdf

def main():
    # parser
    parser=argparse.ArgumentParser(description=DESC, 
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--properties_files', required=True,
            nargs='*', help='Cascade property files in format specified by "--input_format".')
    parser.add_argument('-f', '--input_format', default='parquet',
            choices=['json', 'parquet'],
            help='Input format.')
    parser.add_argument('-p','--plot_request_file',
            help=f'Plots to visualize will be specified through a JSON file with some mandatory and some optional fields. The structure is still evolving. Some template(s) or example(s) will be provided in the future.')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    # set logger
    if args.debug:
       logging.basicConfig(level=logging.DEBUG,format=FORMAT)
    elif args.quiet:
       logging.basicConfig(level=logging.WARNING,format=FORMAT)
    else:
       logging.basicConfig(level=logging.INFO,format=FORMAT)

    start = time()

    # Reading plot request file
    logging.info(f'Reading plot list file: {args.plot_request_file} ...')
    with open(args.plot_request_file) as f:
        plot_request = load(f)
    data_request = plot_request['data']
    del plot_request['data']

    # Preparing data
    logging.info(f'Reading {len(args.properties_files)} property file names ..')
    cascade_props = data.CascadeProperties(args.properties_files,
            data_request=data_request, input_format=args.input_format, 
            argparse=vars(args))

    # Start plotting
    logging.info(f'Start plotting ...')
    for k, item in plot_request.items():
        logging.info(f'Plotting {k} ...')
        plotter.plot(cascade_props, k, item)
        
    logging.info(f'Time taken: {(time()-start)//60}.')
    logging.info('Done.')

if __name__=='__main__':
    main()
