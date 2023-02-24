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

