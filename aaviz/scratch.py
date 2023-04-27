    # Identifying communities
    def community(self, column=None, label=False):
        self.communities = self.nodes.groupby(column).apply(
                lambda x: x['name'].tolist()).reset_index().rename(columns={
                    column: 'name',
                    0: 'community'})
        if label:
            self.communities['label'] = self.communities.name
        else:
            self.communities['label'] = ''
        self.communities['style'] = DEFAULT_COMMUNITY_STYLE



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

SNS_PARAMS = {
        'style': 'whitegrid',
        'palette': COLORS['mathematica']
        }

def create_subplot_grids(fig, rows, cols):
    return fig.add_gridspec(rows, cols)
    # Use: ax=fig.add_subplot(gs[0,0])

class single:   # Seaborn single plot
    def __init__(self, **kwargs):
        self.fig = initiate_plot()
        ax = self.fig.add_subplot()
        self.out_file_prefix = kwargs['out_file_prefix']
        self.data = kwargs['data']
        self.plot_function = eval(kwargs['plot_function'])

        set_optional_attributes(self, kwargs)

        # Plot
        self.plot_function(ax=ax, data=self.data, **self.plot_func_args)

        # Title, etc.
        ax.set_title(self.title)
        set_labels(ax=ax, xlabel=self.xlabel, ylabel=self.ylabel)
        
        # Grid
        set_grid(ax=ax, type=self.grid_type) 
        
        # Font sizes for now
        set_fonts(ax=ax, global_font_size=self.global_font_size, 
                font_size=self.font_size)

        # Any other custom plot modification before saving
        if self.plot_modifier is not None:
            _locals = locals()
            exec(self.plot_modifier, globals(), _locals)
        plt.savefig(self.out_file_prefix+'.pdf', bbox_inches='tight')

def set_hist(axis, mode='horizontal'):
    axis.spines['left'].set_visible(False)

    if mode=='horizontal':
        axis.grid(False,axis='x')
        axis.grid(True,axis='y')
    elif mode=='vertical':
        axis.grid(True,axis='x')
        axis.grid(False,axis='y')
    else:
        raise ValueError(f'Wrong mode {mode}.')

    axis.tick_params(colors=AXES_COLOR,labelcolor='black',direction='inout')
    axis.set_axisbelow(True)
    return



class relplot:
    def __init__(self, **kwargs):
        plt.clf()
        self.out_file_prefix = kwargs['out_file_prefix']
        self.data = kwargs['data']
        self.title = None
        return

    def default(self, plot_args):
        fig = initiate_plot()
        sns.set_theme(style='whitegrid')
        grid = sns.relplot(data=self.data, **plot_args)
        if self.title is not None:
            grid.fig.suptitle(self.title, y=1)

        plt.savefig(self.out_file_prefix+'.pdf', bbox_inches='tight')
        
class boxplot:
    def __init__(self, **kwargs):
        plt.clf()
        self.out_file = out_file
        self.data = data
        return

    def default(self, **kwargs):
        sns.set_theme(style='whitegrid')
        sns.boxplot(data=self.data, **kwargs)
        plt.savefig(self.out_file, bbox_inches='tight')
        
# This function is the frontend for individual plots.
# AA: Probably, we will stick to this and not have multiple files being produced.
def viz(**kwargs):
    # level 0 fields
    data = kwargs['data']
    out_file = kwargs['out_file']
    if 'active' in kwargs.keys():
        if not kwargs['active']:
            logging.info('Inactive: skipping.')
            return
    logging.info('Setting figure mode ...')
    if plot_request['fig_mode'] not in FIG_MODES:
        raise ValueError(f'Unsupported figure mode: {plot_request["fig_mode"]}; Allowed modes are {FIG_MODES}.')
    else:
        fig_mode = eval(plot_request['fig_mode'])

    logging.info(f'Preparing data ...')
    data_list = []
    # AA: df should be the only one exposed externally
    if 'data_modifier' in plot_request:
        if plot_request['data_modifier']:
            _locals = locals()
            exec(plot_request['data_modifier'], globals(), _locals)
            rd = _locals['rd']
            df = rd['df']

    logging.info(f'Preparing plotting function ...')
    plt.clf()
    parse_func = plot_request['func'].split('.')
    plot_class = eval(parse_func[0])
    plot_obj = plot_class(out_file_prefix=plot_identifier,
            data=df, **plot_request)
    plot_func = getattr(plot_obj, parse_func[1])

    logging.info(f'Generating plot ...')
    plot_func()

def set_optional_attributes(obj, in_att_dict):
    att_dict = {
            'plot_modifier': None,
            'plot_func_args': None,
            'title': '',
            'xlabel': None,
            'ylabel': None,
            'grid_type': 'normal',
            'global_font_size': 15,
            'font_size': {}
            }
    for key in att_dict.keys():
        if key in in_att_dict.keys():
            setattr(obj, key, in_att_dict[key])
        else:
            setattr(obj, key, att_dict[key])
    return

