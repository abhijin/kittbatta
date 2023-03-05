DESC='''Given node and edge attributes, create tikz visualization. The
main script is just an example that demos many functionalities. To fully
utilize the features, import this file and use it in a script.

By: AA

Pending:
    - labels outside
    - edge labels
    - pajek style edges
    - multiple networks
    - edge weights helper functions
    - spring layout with edge weights
'''

import argparse
import logging
from math import atan2
import networkx as nx
import numpy as np
from os import devnull
import pandas as pd
from pdb import set_trace
from random import seed
from re import sub
from subprocess import check_call

FORMAT = "[%(filename)s] [%(levelname)s]: %(message)s"
PREAMBLE = r'''
\documentclass[tikz,border=2]{standalone}
\usetikzlibrary{shadows,arrows,shapes,positioning,calc,backgrounds,fit}
\usepackage{amssymb}
\usepackage{array}
\usepackage{colortbl}
\pdfpageattr {/Group << /S /Transparency /I true /CS /DeviceRGB>>}
\newcommand{\lens}[7]{ % ux,uy,vx,vy,in,out,color
\path[fill=#7,out=#5,in=#6] (#1,#2) -- (#3,#4);
}
\newcommand{\lensarrow}[7]{ % ux,uy,vx,vy,in,out,arrowwidth
\draw[-{Latex[width=#7},out=#5,in=#6] (#1,#2) -- (#3,#4);
}
'''

BEGIN_STATEMENT = r'''
\begin{document}
\begin{tikzpicture}
'''

END_STATEMENT = r'''
\end{tikzpicture}
\end{document}
'''

DEFAULT_NODE_STYLE = 'draw'
DEFAULT_EDGE_STYLE = ''
DEFAULT_EDGE_DRAW = ''
DEFAULT_NODE_LABEL = ''
DEFAULT_EDGE_LABEL = ''
DEFAULT_EDGE_LABEL_STYLE = ''

LENS_THICKNESS_FACTOR = 20
LENS_INNER_FACTOR = 10
LENS_NODE_DISP = .2

# The GraphToDraw class contains properties and methods to
# - generate coordinates for each node
# - set those individual node and edge styles which are difficult to set directly.
# - set global styles
class GraphToDraw:
    def __init__(self, nodes, edges, **kwargs):
        self.nodes = nodes
        if 'seed' in kwargs.keys():
            seed(kwargs['seed'])
            np.random.seed(kwargs['seed'])

        self.edges = edges
        if 'directed' in kwargs.keys():
            self.directed = kwargs['directed']
        else:
            self.directed = False

        if 'style' not in self.nodes.columns:
            self.nodes['style'] = DEFAULT_NODE_STYLE
        if 'style' not in self.edges.columns:
            self.edges['style'] = DEFAULT_EDGE_STYLE
        if 'draw' not in self.edges.columns:
            self.edges['draw'] = DEFAULT_EDGE_DRAW
        if 'label' not in self.nodes.columns:
            self.nodes['label'] = DEFAULT_NODE_LABEL
        if 'label' not in self.edges.columns:
            self.edges['label'] = DEFAULT_EDGE_LABEL
        if 'label_style' not in self.edges.columns:
            self.edges['label_style'] = DEFAULT_EDGE_LABEL_STYLE
        if 'lens' not in self.edges.columns:
            self.edges['lens'] = False
        if 'weight' not in self.edges.columns:
            self.edges['weight'] = 1
        self.global_style = ''
        self.colors = ''
        self.appendix = ''
        self.Gnx = None

    def __convert_to_networkx(self):
        if self.Gnx is None:
            logging.info('Generating networkx graph ...')
            if self.directed:
                constructor = nx.DiGraph
            else:
                constructor = nx.Graph
            self.Gnx = nx.from_pandas_edgelist(self.edges, source='source', 
                    target='target', create_using=constructor)
            # isolated nodes
            isolated_nodes = list(set(self.nodes.name.tolist()).difference(
                    set(self.Gnx.nodes)))
            if isolated_nodes:
                self.Gnx.add_nodes_from(isolated_nodes)
        return

    def layout(self, layout_type, **kwargs):
        logging.info(f'Layout type: {layout_type}')
        if not 'seed' in kwargs:    # for layouts that require random
            seed = None
        else:
            seed = kwargs['seed']
        if layout_type=='given':
            # expecting x,y in nodes
            if 'x' not in self.nodes.columns or 'y' not in self.nodes.columns:
                logging.error("Expected columns 'x' and 'y'.")
                raise
            # check for overlaps
            coords = pd.Series(list(zip(self.nodes['x'],self.nodes['y'])))
            lc = coords.shape[0]
            if lc > coords.drop_duplicates().shape[0]:
                logging.warning('Found overlapping nodes.')
        elif layout_type=='spring':
            self.__convert_to_networkx()
            pos = pd.DataFrame(nx.spring_layout(self.Gnx,
                seed=seed)).T
            pos = pos.rename(columns={0: 'x', 1: 'y'})
            try:
                logging.info('Removing any existing coordinates ...')
                self.nodes = self.nodes.drop(['x','y'], axis=1)
            except KeyError:
                logging.info('Found none.')
            self.nodes = self.nodes.merge(pos, left_on='name', 
                    right_index=True)
        else:
            logging.error('Unsupported layout type')
            raise

    def layout_scale_round(self, minx=None, maxx=None, 
            miny=None, maxy=None, round=2, padding=0):
        ominx = self.nodes.x.min()
        omaxx = self.nodes.x.max()
        ominy = self.nodes.y.min()
        omaxy = self.nodes.y.max()

        xratio = (maxx-minx-2*padding)/(omaxx-ominx)
        yratio = (maxy-miny-2*padding)/(omaxy-ominy)
        self.nodes.x = ((self.nodes.x - ominx) * xratio + minx + padding).round(round)
        self.nodes.y = ((self.nodes.y - ominy) * yratio + miny + padding).round(round)

    def compute_edge_angles(self):
        # this is for looseness (atan2)
        edges = self.edges[['source', 'target']]
        nodes = self.nodes[['name', 'x', 'y']]
        edges = edges.merge(nodes, left_on='source', right_on='name')
        edges = edges.rename(columns={'x': 'source_x', 'y': 'source_y'})
        edges = edges.reset_index().merge(nodes, left_on='target', right_on='name'
                ).set_index('index').sort_index()
        edges = edges.rename(columns={'x': 'target_x', 'y': 'target_y'})

        coords = edges[['source_x', 'source_y', 'target_x', 'target_y']
                ].to_numpy()
        x1 = coords[:,0]
        y1 = coords[:,1]
        x2 = coords[:,2]
        y2 = coords[:,3]

        self.edges['target_angle'] = np.round(np.arctan2((y1-y2),(x1-x2))*180/np.pi,2)
        self.edges['source_angle'] = np.round(np.arctan2((y2-y1),(x2-x1))*180/np.pi,2)
        
        return

    def displace_angles(self,displacement,mode='random'):
        # displacement for curved lines
        if mode == 'random':
            disp = np.random.choice([-1,1],self.edges.shape[0]) * displacement
        elif mode == 'fixed':
            disp = displacement
        else:
            raise ValueError(f'Invalid mode "{mode}".')
        self.edges['out_angle'] = self.edges.source_angle - disp
        self.edges['in_angle'] = self.edges.target_angle + disp

    def set_global_style(self, style_string):
        logging.info('Setting global style ...')
        self.global_style = style_string
        return

    def append_node_attribute(self, prefix, values):
        self.nodes['style'] = self.nodes['style'] + ',' + prefix + values
        return

    def append_edge_attribute(self, prefix, values, mode='edge'):
        if mode == 'edge':
            # Note that "style" is overloaded as it is also used in pandas
            self.edges['style'] = self.edges['style'] + ',' + prefix + values
        elif mode == 'draw':
            self.edges.draw = self.edges.draw + ',' + prefix + values
        else:
            raise ValueError(f'Wrong mode {mode}.')
        return

    def append_edge_label_attribute(self, prefix, values):
        self.edges['label_style'] = self.edges['label_style'] + ',' + prefix + \
                values

    def add_appendix(self, string):
        self.appendix = string
        return

    def define_html_colors(self, names, codes):
        color_string = '\n% custom colors\n'
        for name, code in zip(names,codes):
            color_string += f'\\definecolor{{{name}}}{{HTML}}{{{code}}}\n'
        self.colors = color_string

def generate_lens_edge(edge, directed):
    outer_out_angle = np.round(edge.out_angle - 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    inner_out_angle = np.round(edge.out_angle + 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    outer_in_angle = np.round(edge.in_angle + 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    inner_in_angle = np.round(edge.in_angle - 
            LENS_THICKNESS_FACTOR * (1-np.exp(-edge.weight)), 2)
    # remove any out=,in=
    estyle = sub('out=[^,]*', '', edge.style)
    estyle = sub('in=[^,]*', '', estyle)
    estyle = sub(',,*', ',', estyle)

    # source and target locations
    sxd = np.round(LENS_NODE_DISP*np.cos(edge.source_angle),2)
    syd = np.round(LENS_NODE_DISP*np.sin(edge.source_angle),2)
    txd = np.round(LENS_NODE_DISP*np.cos(edge.target_angle),2)
    tyd = np.round(LENS_NODE_DISP*np.sin(edge.target_angle),2)
    source = f'($({edge.source}.center) + ({sxd},{syd})$)'
    target = f'($({edge.target}.center) + ({txd},{tyd})$)'

    # add edges
    estring = f'\\path {source} edge[fill={edge.color},{estyle},out={outer_out_angle},in={outer_in_angle}] {target};\n'
    estring += f'\\path {source} edge[fill={edge.color},{estyle},out={inner_out_angle},in={inner_in_angle}] {target};\n'
    return estring


def draw(G, mode='segment', outfile='out.tex'):

    # set global styles
    if G.global_style == '':
        out = ''
    else:
        out = '[' + G.global_style + ']\n\n'

    out += '% Nodes\n'

    for id,node in G.nodes.iterrows():
        out += f"\\node[{node.style}] ({node['name']}) at ({node.x},{node.y}) {{{node.label}}};\n"

    out += '\n% Edges\n'

    directed = ''
    if G.directed:
        directed = ',->'

    for id,edge in G.edges.iterrows():
        try:
            if edge.directed == True:   # for mixed edges
                directed = ',->'
            else:
                directed = ''
        except:
            pass

        if edge.lens:
            out += generate_lens_edge(edge, directed)
        elif edge.label != DEFAULT_EDGE_LABEL:
            edge.label_style = 'inner sep=.5pt' + edge.label_style
            out += f'\\draw[{edge.draw}] ({edge.source}) edge[{edge.style}{directed}] node[{edge.label_style}] {{{edge.label}}} ({edge.target});\n'
        else:
            out += f'\\draw[{edge.draw}] ({edge.source}) edge[{edge.style}{directed}] ({edge.target});\n'


    if mode == 'segment':
        return G.colors + out
    elif mode == 'standalone':
        return PREAMBLE + G.colors + BEGIN_STATEMENT + out + G.appendix + \
                END_STATEMENT
    elif mode == 'file':
        with open(outfile,'w') as f:
            f.write(PREAMBLE + G.colors + BEGIN_STATEMENT + out + G.appendix + \
                    END_STATEMENT)
    else:
        logging.error(f'Wrong mode {mode}')
        raise

def compile(texfile):
    with open(devnull,'w') as fnull:
        check_call(
                ['pdflatex','-interaction','nonstopmode','-halt-on-error',texfile],
                stdout=fnull)

def main():
    # Parser
    parser=argparse.ArgumentParser(description=DESC,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s", "--seed", type=int, help="Random seed for layouts.")
    parser.add_argument("--directed", action='store_true',
            help="Directed or undirected.")
    
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    
    # set logger
    if args.debug:
       logging.basicConfig(level=logging.DEBUG,format=FORMAT)
    elif args.quiet:
       logging.basicConfig(level=logging.WARNING,format=FORMAT)
    else:
       logging.basicConfig(level=logging.INFO,format=FORMAT)

    # read node and edge files
    nodes = pd.DataFrame({'name': [0,1,2,3]})
    edges = pd.DataFrame({'source': [0,0,1,2,3], 'target': [1,2,2,3,0]})

    # initiate graph for drawing
    G = GraphToDraw(nodes,edges,directed=args.directed)

    # layout
    G.layout('spring')

    # scale
    G.layout_scale_round(0,5,0,5)

    # find angles for curved edges (after scaling)
    G.compute_edge_angles()
    # G.displace_angles(10)
    G.append_edge_attribute('out=',G.edges.out_angle.astype(str))
    G.append_edge_attribute('in=',G.edges.in_angle.astype(str))

    # lens edges
    G.edges.loc[1, 'lens'] = True
    G.edges['color'] = 'black!20'

    # color
    colors = ['blue', 'green']
    html = ['CBD5E8', 'B3E2CD']
    G.define_html_colors(colors,html)
    node_color = pd.Series(['blue'] * G.nodes.shape[0])
    node_color[0] = 'purple!50'
    node_color[2] = 'green'
    node_color[3] = 'red!60'
    G.append_node_attribute('fill=',node_color)

    # shape
    node_shape = pd.Series(['circle'] * G.nodes.shape[0])
    node_shape[1] = 'square'
    G.append_node_attribute('',node_shape)

    # set styles
    G.set_global_style('''
every node/.style={circle},
square/.style={rectangle, minimum width=4mm, minimum height=4mm},
every edge/.style={draw,dashed,looseness=1,>=latex}''')

    # label
    G.nodes['label'] = G.nodes.name

    # draw
    out = draw(G, mode='standalone')

    # output graph
    with open('out.tex','w') as f:
        f.write(out)
    logging.info('Written to out.tex.')

    compile('out.tex')

if __name__ == "__main__":
    main()
