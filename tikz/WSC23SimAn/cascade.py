DESC='''
Cascade example

By: AA
'''

import numpy as np
import pandas as pd
from pdb import set_trace

import aaviz.graph_to_tikz as gtt

nodes = pd.DataFrame({'name': np.arange(1,23),
    'cascade': [1]*12 + [0]*10})
nodes.loc[nodes.cascade == 1, 'type'] = 'infected'
nodes.loc[nodes.cascade == 0, 'type'] = 'boundary'
nodes.loc[nodes.name.isin([1,7]), 'type'] = 'source'
# nodes['label'] = nodes.name

edges = pd.DataFrame({
    'source': [1,1,2,3,3,7,8,8,9,9],
    'target': [2,3,4,5,6,8,9,10,11,12]})
##     'label': ['E','M','E','N','M','E','M','E','M','N']})
## edges.label = '\\texttt{' + edges.label + '}'
edges['directed'] = True
edges['dashed'] = ''

boundary = pd.DataFrame({
    'source': [13,13,13,14,14,15,15,16,16,17,17,18,18,19,20,21,21,22],
    'target': [2,5,8,4,3,9,12,7,11,1,13,5,10,2,2,8,10,7]})
boundary['directed'] = False
## boundary['label'] = gtt.DEFAULT_EDGE_LABEL
boundary['dashed'] = 'dashed'

edges = pd.concat([edges, boundary], ignore_index=True)

G = gtt.GraphToDraw(nodes, edges, seed=5)

opacity=.85
G.set_global_style('''
arr/.style={>=latex, shorten >=1pt, shorten <=1pt},
source/.style={circle,minimum width=3.5mm,fill=myRed,draw=myRed!20,line width=1mm},
infected/.style={circle,minimum width=2.5mm,fill=myRed,draw=white,opacity=%f},
boundary/.style={circle,minimum width=2.5mm,fill=myBlue,draw=white,opacity=%f},
''' %(opacity, opacity))

G.layout('spring', seed=4)
G.layout_scale_round(minx=0, maxx=6, miny=0, maxy=5)

G.define_html_colors(['myBlue', 'myRed'], ['0060AD','DD181F'])


G.append_node_attribute('', nodes.type)

G.append_edge_attribute('', 'black!80')
G.append_edge_attribute('', 'arr')
G.append_edge_attribute('', edges.dashed)

## G.append_edge_label_attribute('font=', '\small')
## G.append_edge_label_attribute('fill=', 'white')

G.compute_edge_angles()
G.displace_angles(10)
# Finer adjustments of angles
# AA: This might become a function
for x,y in [(13,2), (20,2), (19,2)]:
    d = G.edges[(G.edges.source==x) & (G.edges.target==y)].displacement
    G.edges.loc[(G.edges.source==x) & (G.edges.target==y), 'displacement'] = -d
G.displace_angles(G.edges.displacement, mode='fixed') # for finer control
G.append_edge_attribute('out=',G.edges.out_angle.astype(str))
G.append_edge_attribute('in=',G.edges.in_angle.astype(str))

app = '''
\\node (src) at (0,1) [source] {};
\\node (inf) at (0,.5) [infected] {};
\\node (bou) at (0,0) [boundary] {};
\\node [right=of src,xshift=-1cm,anchor=west] {Seed node};
\\node [right=of inf,xshift=-1cm,anchor=west] {Infected node};
\\node [right=of bou,xshift=-1cm,anchor=west] {Boundary node};
%% \\draw[arr,<-,black!60] (.7,3) -- +(.25,1) node[above,black] {Edge label};
%% \\node at (1.9,1.1) {$d_{\\textrm{out}}=2$};
%% \\node at (3.2,3) {$d^B_C=3$};
'''
G.add_appendix(app)

gtt.draw(G, mode='file', outfile='cascade_example.tex')
gtt.compile('cascade_example.tex')


