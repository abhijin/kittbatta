DESC='''plot functions
By AA
'''

from cycler import cycler
from itertools import product
import logging
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc, patches
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import pandas as pd
from pdb import set_trace
from re import sub, search, match
import seaborn as sns
try:
    import geopandas as gpd
    from shapely.geometry import Point
except:
    print('Warning: GIS modules not loaded.')

# contextily EPSG 3857

COLORS = {
        'mathematica': ['#5e82b5','#e09c24','#8fb030','#eb634f','#8778b3','#c46e1a','#5c9ec7','#fdbf6f'],
        'grand_budapest': ['#5b1a18','#fd6467','#f1bb7b','#d67236'],
        'red_blue': ['#0060ad', '#dd181f'],
        'datanovia': ["#FFDB6D", "#C4961A", "#F4EDCA", "#D16103", "#C3D7A4", 
            "#52854C", "#4E84C4", "#293352"]
        }
SNS_AXIS_PLOTS = ['sns.lineplot', 'sns.barplot', 'sns.histplot', 'sns.ecdfplot',
        'sns.boxplot', 'sns.violinplot', 'sns.heatmap']
AXIS_NORMAL = ['sns.lineplot']
AXIS_HEAT = ['sns.heatmap']
AXIS_HIST = ['sns.barplot', 'sns.histplot']
AXIS_BOX = ['sns.boxplot', 'sns.violinplot']
AXIS_NONE = ['gpd.plot', 'gpd.boundary.plot']

NON_FUNC_PARAMS = ['fig', 'subplot', 'title', 'xlabel', 'ylabel', 'data']
HATCH = ['+', 'x', '\\', '.', 'o', '|', '*']

RC_PARAMS = {
        'font.family': 'serif',
        'font.serif': 'Computer Modern Roman',
        'legend.frameon': False,
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'text.usetex': True,
        'text.latex.preamble': '',
        'axes.prop_cycle': cycler(color=COLORS['mathematica']) 
        }

DEFAULT_FONTS = {
        'fontsize': 17,
        'title': 'large',
        'xlabel': 'normalsize',
        'ylabel': 'normalsize',
        'xtick': 'small',
        'ytick': 'small',
        'legend': 'normalsize',
        'legend_title': 'normalsize',
        'colorbar': 'normalsize',
        }


# latex fontsizes
FONT_TABLE=pd.DataFrame({9: {'miniscule': 4, 'tiny': 5, 'scriptsize': 6, 'footnotesize': 7, 'small': 8, 'normalsize': 9, 'large': 10, 'Large': 11, 'LARGE': 12, 'huge': 14, 'Huge': 17, 'HUGE': 20}, 10: {'miniscule': 5, 'tiny': 6, 'scriptsize': 7, 'footnotesize': 8, 'small': 9, 'normalsize': 10, 'large': 11, 'Large': 12, 'LARGE': 14, 'huge': 17, 'Huge': 20, 'HUGE': 25}, 11: {'miniscule': 6, 'tiny': 7, 'scriptsize': 8, 'footnotesize': 9, 'small': 10, 'normalsize': 11, 'large': 12, 'Large': 14, 'LARGE': 17, 'huge': 20, 'Huge': 25, 'HUGE': 30}, 12: {'miniscule': 7, 'tiny': 8, 'scriptsize': 9, 'footnotesize': 10, 'small': 11, 'normalsize': 12, 'large': 14, 'Large': 17, 'LARGE': 20, 'huge': 25, 'Huge': 30, 'HUGE': 36}, 14: {'miniscule': 8, 'tiny': 9, 'scriptsize': 10, 'footnotesize': 11, 'small': 12, 'normalsize': 14, 'large': 17, 'Large': 20, 'LARGE': 25, 'huge': 30, 'Huge': 36, 'HUGE': 48}, 17: {'miniscule': 9, 'tiny': 10, 'scriptsize': 11, 'footnotesize': 12, 'small': 14, 'normalsize': 17, 'large': 20, 'Large': 25, 'LARGE': 30, 'huge': 36, 'Huge': 48, 'HUGE': 60}, 20: {'miniscule': 10, 'tiny': 11, 'scriptsize': 12, 'footnotesize': 14, 'small': 17, 'normalsize': 20, 'large': 25, 'Large': 30, 'LARGE': 36, 'huge': 48, 'Huge': 60, 'HUGE': 72}, 25: {'miniscule': 11, 'tiny': 12, 'scriptsize': 14, 'footnotesize': 17, 'small': 20, 'normalsize': 25, 'large': 30, 'Large': 36, 'LARGE': 48, 'huge': 60, 'Huge': 72, 'HUGE': 84}, 30: {'miniscule': 12, 'tiny': 14, 'scriptsize': 17, 'footnotesize': 20, 'small': 25, 'normalsize': 30, 'large': 36, 'Large': 48, 'LARGE': 60, 'huge': 72, 'Huge': 84, 'HUGE': 96}, 36: {'miniscule': 14, 'tiny': 17, 'scriptsize': 20, 'footnotesize': 25, 'small': 30, 'normalsize': 36, 'large': 48, 'Large': 60, 'LARGE': 72, 'huge': 84, 'Huge': 96, 'HUGE': 108}, 48: {'miniscule': 17, 'tiny': 20, 'scriptsize': 25, 'footnotesize': 30, 'small': 36, 'normalsize': 48, 'large': 60, 'Large': 72, 'LARGE': 84, 'huge': 96, 'Huge': 108, 'HUGE': 120}, 60: {'miniscule': 20, 'tiny': 25, 'scriptsize': 30, 'footnotesize': 36, 'small': 48, 'normalsize': 60, 'large': 72, 'Large': 84, 'LARGE': 96, 'huge': 108, 'Huge': 120, 'HUGE': 132}})

AXES_COLOR='#999999'
GRID_COLOR='#cccccc'
TICKS_COLOR='#222222'

MAJOR_TICK_LINEWIDTH = .75
MINOR_TICK_LINEWIDTH = .25

FIGSIZE=(8,6)
LINESTYLES=['solid','dashed']
LINEWIDTH=[1]
MARKERS=['.','o','v','^','s']
# LINE_PROPS=pd.DataFrame(product(['solid','dashed'],MARKERS,MATHEMATICA[0:7],LINEWIDTH),columns=['style','marker','color','width'])

pd.options.display.float_format = '{:.10g}'.format

# The helper functions are arranged in the order in which they should be called
def initiate_figure(**kwargs):
    argvals = {
            'mode': 'figure',
            'subplot_mode': 'gridspec',
            'x': FIGSIZE[0],
            'y': FIGSIZE[1],
            'gs_wspace': 0.2,
            'gs_hspace': 0.2,
            'gs_nrows': 1,
            'gs_ncols': 1,
            'st_y': 0.9,
            'st_fontsize': 'large'
            }

    for k in kwargs.keys():
        argvals[k] = kwargs[k] 

    # Setting font families and latex stuff
    # AA: need to be able to modify this too at some point
    for k,v in RC_PARAMS.items():
        rcParams[k] = v

    # Setting default font sizes
    fs_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'fs_'}
    for k,v in fs_args.items():
        DEFAULT_FONTS[k] = v

    # Figure
    if argvals['mode'] == 'figure':
        fig_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'fg_'}
        if argvals['subplot_mode'] == 'gridspec':
            fig = plt.figure(figsize=[argvals['x'],argvals['y']], **fig_args)
            subplot_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'gs_'}
            gs = fig.add_gridspec(subplot_args['nrows'], subplot_args['ncols'])
            gs.update(wspace=subplot_args['wspace'], 
                    hspace=subplot_args['hspace']) # set the spacing between axes.

            if 'suptitle' in kwargs.keys():
                suptitle_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'st_'}
                suptitle_args['fontsize'] = FONT_TABLE[DEFAULT_FONTS[
                    'fontsize']][suptitle_args['fontsize']]
                fig.suptitle(kwargs['suptitle'], **suptitle_args)

            return fig, gs
        # AA: not used currently
        elif argvals['subplot_mode'] == 'subplots':
            raise('Not supported. Some issues.')
            subplot_args = {k[3:]: v for k,v in argvals.items() if k[0:3] == 'sp_'}
            fig, ax = plt.subplots(**subplot_args, figsize=[argvals['x'],argvals['y']])
            if type(ax) != np.ndarray:
                return fig, [ax]
            else:
                return fig, ax
    elif mode == 'facetgrid':
        return  # currently, nothing to return

def subplot(**kwargs):

    try:
        logging.info(kwargs['la_title'])
    except KeyError:
        logging.info('Empty title')

    # Collect all arguments that pertain to func
    for k in kwargs.keys():
        if k in ['fig', 'ax', 'func', 'data', 'text', 'hatch', 'sharey', 'sharex', 'grid']:
            continue
        pat = search('[^_]*_', k)[0]
        if pat not in ['sp_', 'pf_', 'ag_', 'la_', 'fs_', 'xt_', 'yt_', 'lg_', 'tx_']:
            raise ValueError(f'Unsupported input class "{k}".')
    subplot_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'sp_'}
    plot_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'pf_'}
    axes_grid_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'ag_'}
    label_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'la_'}
    fontsize_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'fs_'}
    xtick_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'xt_'}
    ytick_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'yt_'}
    legend_args = {k[3:]: v for k,v in kwargs.items() if k[0:3] == 'lg_'}

    fig = kwargs['fig']
    func = kwargs['func']

    # Create subplot if not exists
    if 'ax' in kwargs.keys():   # axis given
        ax = kwargs['ax']
    else:
        axis_list = fig.axes
        if 'sharex' in subplot_args.keys():
            subplot_args['sharex'] = axis_list[subplot_args['sharex']]
        if 'sharey' in subplot_args.keys():
            subplot_args['sharey'] = axis_list[subplot_args['sharey']]
        ax = fig.add_subplot(kwargs['grid'], **subplot_args)
        kwargs['ax'] = ax

    # Decide function
    try:
        plot_args['data'] = kwargs['data']
    except KeyError:
        print('WARNING: "data" field absent.')
    funcobj = subplot_func(plot_args=plot_args, **kwargs)

    funcobj(ax=ax, **plot_args)

    # Hatching
    try:
        if kwargs['hatch']:
            ax = subplot_hatch(ax=ax, func=func)
    except KeyError:
        pass

    subplot_axes_grid(func=func, ax=ax, plot_args=plot_args, 
            subplot_args=subplot_args, **axes_grid_args)
    subplot_labels(ax=ax, **label_args)
    fontsizes = subplot_fonts(ax=ax, func=func, **fontsize_args)
    
    if xtick_args:
        ax.set_xticklabels(ax.get_xticklabels(), **xtick_args)
    if ytick_args:
        ax.set_yticklabels(ax.get_yticklabels(), **ytick_args)

    set_legend(ax, legend_args, FONT_TABLE[fontsizes['fontsize']], 
            fontsize_args)

    return ax

def set_legend(ax, legend_args, fonts_table, fontsize_args):
    # Only if a legend is present and there are some modifications to be made
    try:
        legend_visible = legend_args['visible']
        legend_args.pop('visible')
    except KeyError:
        legend_visible = True

    # set fonts
    legend_fonts = {}
    for type in ['legend', 'legend_title']:
        if type in fontsize_args.keys():
            legend_fonts[type] = fontsize_args[type]
        else:
            legend_fonts[type] = DEFAULT_FONTS[type]

    legend_handles = ax.get_legend_handles_labels()[0]
    if len(legend_handles):
        if legend_visible:
            ax.legend(**legend_args, 
                    fontsize=fonts_table[legend_fonts['legend']], 
                    title_fontsize=fonts_table[legend_fonts['legend_title']])
        else:
            ax.legend().set_visible(False)

def set_legend_invisible(ax):
    ax.saved_legend_handles = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    return

def separate_legend(fig=None, axes=None, labels=None, **kwargs):
    handles = []
    derived_labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.saved_legend_handles
        handles = handles + ax_handles
        derived_labels = derived_labels + ax_labels

    if labels == None:
        labels = derived_labels

    if 'fontsize' in kwargs.keys():
        kwargs['fontsize'] = FONT_TABLE[DEFAULT_FONTS['fontsize']][
                kwargs['fontsize']]

    fig.legend(handles, labels, **kwargs)

def subplot_func(**kwargs):
    funcname = kwargs['func']
    plot_args = kwargs['plot_args']
    ax = kwargs['ax']
    try:
        plot_args['data'] = kwargs['data']
    except KeyError:
        pass
    if funcname in ['gpd.plot', 'gpd.boundary.plot']:
        plot_args.pop('data')
        func = eval(f'kwargs["data"].plot')
    elif funcname in SNS_AXIS_PLOTS:
        if funcname in ['sns.barplot', 'sns.histplot']:
            argvals = {
                    'alpha': 1,
                    'edgecolor': 'white'
                    }
        elif funcname in AXIS_BOX:
            argvals = {
                    'boxprops': {'edgecolor': 'white'},
                    'whiskerprops': {'color': TICKS_COLOR},
                    }
        elif funcname in AXIS_NORMAL:
            argvals = {}
        elif funcname in AXIS_HEAT:
            argvals = {}
        elif funcname in AXIS_NONE:
            argvals = {}
        else:
            raise(f'The plot "{func_name}" is not supported.')

        for k,v in argvals.items():
            if k not in plot_args.keys():
                plot_args[k] = v

        func = eval(funcname)
    else:
        raise ValueError(f'Unsupported function type {kwargs["func"]}.')
    return func

def subplot_hatch(**kwargs):
    ax = kwargs['ax']
    func = kwargs['func']
    if func in AXIS_HIST:
        for bars, h in zip(ax.containers, cycler(hatch=HATCH)()):
            for bar in bars:
                bar.set_hatch(h['hatch'])
    # For box plots, this is tricky
    if func in AXIS_BOX:
        pp = [patch for patch in ax.patches if 
                type(patch) == patches.PathPatch]
        huelist = []
        for patch in pp:
            huelist.append(patch.get_facecolor())
        hues = pd.DataFrame.from_records(huelist).drop_duplicates()

        for patch in pp:
            col = patch.get_facecolor()
            hue = hues[hues == col].dropna().index[0]
            patch.set_hatch(HATCH[hue])
            edgecol = patch.get_edgecolor() # assumes same edgecolor for all

        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        for patch in handles:
            col = patch.get_facecolor()
            hue = hues[hues == col].dropna().index[0]
            patch.set_edgecolor(edgecol)
            patch.set_hatch(HATCH[hue])

        #ax.legend(handles=for_legend)

    return ax

def subplot_axes_grid(**kwargs):
    if 'axis_type' in kwargs.keys():
        axis_type = kwargs['axis_type']
    elif kwargs['func'] in AXIS_NONE:
        axis_type = 'none'
    elif kwargs['func'] in AXIS_HEAT:
        axis_type = 'heat'
    elif kwargs['func'] in AXIS_NORMAL:
        axis_type = 'normal'
    elif kwargs['func'] in AXIS_HIST:
        if 'orient' in kwargs['plot_args']:
            if kwargs['plot_args']['orient'] == 'h':
                axis_type = 'histx'
        else:
            axis_type = 'histy'
    elif kwargs['func'] in AXIS_BOX:
        if 'orient' in kwargs['plot_args']:
            if kwargs['plot_args']['orient'] == 'h':
                axis_type = 'boxx'
            else:
                axis_type = 'boxy'
        else:
            axis_type = 'boxy'
    else:
        raise KeyError('Unable to assign axis_type. Check if plot is supported.')
    ax = kwargs['ax']
    if axis_type == 'normal':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_color(AXES_COLOR)
        ax.spines['left'].set_color(AXES_COLOR)
        ax.grid(color=GRID_COLOR, which='major', linewidth=1)
        ax.grid(True,color='#dddddd',which='minor',
                linewidth=MINOR_TICK_LINEWIDTH)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    elif axis_type == 'heat':
        ax.spines[['bottom', 'right', 'top', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0)
    elif axis_type == 'boxy':
        ax.grid(axis='y', color=GRID_COLOR, which='major', 
                linewidth=MAJOR_TICK_LINEWIDTH)
        ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        ax.spines['bottom'].set_color(AXES_COLOR)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0)
        ax.set_yticks(ax.get_yticks(), minor=False, colors=AXES_COLOR)
    elif axis_type == 'boxx':
        ax.grid(axis='x', color=GRID_COLOR, which='major', 
                linewidth=MAJOR_TICK_LINEWIDTH)
        ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        ax.spines['left'].set_color(AXES_COLOR)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', length=0)
        ax.set_xticks(ax.get_xticks(), minor=False, colors=AXES_COLOR)
    elif axis_type == 'histy':
        ax.grid(axis='y', color=GRID_COLOR, which='major', 
                linewidth=MAJOR_TICK_LINEWIDTH)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.spines['bottom'].set_color(AXES_COLOR)
        ax.tick_params(axis='y', length=0)
        ax.set_yticks(ax.get_yticks(), minor=False, colors=AXES_COLOR)
    elif axis_type == 'histx':
        ax.grid(axis='x', color=GRID_COLOR, which='major', 
                linewidth=MAJOR_TICK_LINEWIDTH)
        ax.spines[['bottom', 'right', 'top']].set_visible(False)
        ax.spines['left'].set_color(AXES_COLOR)
        ax.tick_params(axis='x', length=0)
        ax.set_xticks(ax.get_xticks(), minor=False, colors=AXES_COLOR)
    elif axis_type == 'none':
        ax.spines[['bottom', 'right', 'top', 'left']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        raise ValueError(f'Unsupported grid type "{axis_type}".')
    ax.set_axisbelow(True)
    ax.tick_params(color=AXES_COLOR, labelcolor=TICKS_COLOR, which='both')

    # Scale
    # logscale: sp
    if 'xscale' in kwargs.keys():
        ax.set_xscale(kwargs['xscale'])
        if kwargs['xscale'] == 'log':
            ax.minorticks_off()
    if 'yscale' in kwargs.keys():
        ax.set_yscale(kwargs['yscale'])
        if kwargs['yscale'] == 'log':
            ax.minorticks_off()

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

    subplot_args = kwargs['subplot_args'] 
    if 'sharex' in subplot_args.keys():
        plt.setp(ax.get_xticklabels(), visible=False)
    if 'sharey' in subplot_args.keys():
        plt.setp(ax.get_yticklabels(), visible=False)

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

def savefig(filename, pad_inches=0.05, **kwargs):
    # use pad_inches=0 to completely remove white space
    plt.savefig(filename, bbox_inches='tight', **kwargs)
    return

def subplot_fonts(**kwargs):
    ax = kwargs['ax']

    argvals = {k: v for k,v in DEFAULT_FONTS.items()}
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
    return argvals

def update_rc_params(field=None, mode='replace', value=None):
    if mode == 'append':
        rcParams[field] = rcParams[field] + value
    elif mode == 'replace':
        rcParams[field] = value
    return

# Get actual coordinates of points in pixels w.r.t. (0,0).
# Helps in downstream tikz tasks.
# Tip: Use pad_inches=0 during plot.savefig()
def get_real_coordinates(fig, ax, units='cm', x=None, y=None):
    # Shift coordinates by making them relative to (0,0)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xc = x - xmin
    yc = y - ymin

    # Set units
    if units == 'px':
        units_ratio = fig.dpi
    elif units == 'cm':
        units_ratio = 2.54 # 1 inch = 2.54cm
    elif units == 'in':
        units_ratio = 1

    # Get ratio to scale
    coord_width = xmax - xmin
    coord_height = ymax - ymin
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    real_width = bbox.width * units_ratio
    real_height = bbox.height * units_ratio
    x_ratio = real_width / coord_width
    y_ratio = real_height / coord_height

    return xc*x_ratio, yc*y_ratio, real_width, real_height

# For fancy square grids
def square_grid_cells(axis=None, num_cells_x=None, num_cells_y=None, 
        labels_step_x=1, labels_step_y=1, type_x=None, type_y=None):

    # Set xticks
    # We are setting minor ticks due to an offset issue. Hence, reducing the 
    # number of cells by half.
    xmin, xmax = axis.get_xlim()
    xticks = np.linspace(xmin, xmax, num=int(np.floor(num_cells_x/2))+1)
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
    yticks = np.linspace(ymin, ymax, num=int(np.floor(num_cells_y/2))+1)
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
    axis.tick_params(axis='both', which='minor', labelleft=False, 
            labelbottom=False)
    axis.grid(color=GRID_COLOR, which='both', linewidth=.5)

    return

# This will return the ith color for now
def get_style(k, i):
    j = 0
    for d in rcParams['axes.prop_cycle']():
        if i == j:
            return d[k]
        j += 1

# Mandatory dictionary fields: x, y, text for each list element
def text(ax=None, text=None, **kwargs):
    if type(text) == pd.DataFrame:
        text = text.to_dict('records')

    for t in text:
        ax.text(t['x'], t['y'], t['text'], **kwargs)

def vlines(ax=None, lines=None, **kwargs):
    if type(lines) == pd.DataFrame:
        lines = lines.to_dict('records')

    for l in lines:
        ax.vlines(l['x'], l['ymin'], l['ymax'], **kwargs)

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
