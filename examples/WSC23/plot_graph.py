import copy
import networkx as nx
import numpy as np
import pandas as pd
from pdb import set_trace

import aaviz.graph_to_tikz as gtt

def plot_graph(nodes, edges):
    # Global styles
    global_style = '''
    every node/.style={circle, inner sep=1pt},
    source/.style={circle,minimum width=3.5mm,fill=myRed,draw=myRed!20,line width=1mm},
    '''
    
    # Any column or row modifications
    nodes = nodes.rename(columns={'node': 'name'})
    edges = edges.rename(columns={'u': 'source', 'v': 'target'})

    # Initiate main graph
    G = gtt.GraphToDraw(nodes, edges)

    # Layout
    ### Assign weights for layout
    pid_hid_map = pd.Series(index=G.nodes['name'].tolist(), 
            data=G.nodes.hid.tolist()) 
    G.edges['target_hid'] = G.edges.target.map(pid_hid_map)
    G.edges['source_hid'] = G.edges.source.map(pid_hid_map)
    # max_duration = G.edges.duration.max()
    G.edges['weight'] = 12 #10 * (1 - np.exp(-G.edges.duration / max_duration)) 

    pid_com_map = pd.Series(index=G.nodes['name'].tolist(),
            data=G.nodes.community.tolist())
    G.edges['target_com'] = G.edges.target.map(pid_com_map)
    G.edges['source_com'] = G.edges.source.map(pid_com_map)
    G.edges.loc[G.edges.target_com==G.edges.source_com, 'weight'] = 20

    G.edges.loc[G.edges.target_hid==G.edges.source_hid, 'weight'] = 21

    ## G.nodes['label'] = G.nodes.hid.astype('str') + ',' + G.nodes.name.astype('str')
    #G.nodes['label'] = G.nodes.name.astype(str).str[-2:]
    G.layout('spring', seed=3)
    G.layout_scale_round(minx=0, maxx=6, miny=0, maxy=5)
    
    # Node attributes
    ### Age group
    G.nodes['color'] = 'Blue!120'
    G.nodes.loc[G.nodes.adult==False, 'color'] = 'Red!120'
    G.append_node_attribute('fill=', G.nodes.color)
    G.append_node_attribute('draw=', G.nodes.color)
    
    # Edge attributes
    max_duration = G.edges.duration.max()
    G.edges['duration_weight'] = np.ceil(
            100*(1-np.exp(-2*G.edges.duration/max_duration))).astype(int).astype(str)

    G.append_edge_attribute('', 'black!'+G.edges.duration_weight)
    
    # Household 
    G.community(column='hid', stretch=.08, label=False)
    child_map = G.nodes[['hid', 'community']].drop_duplicates().set_index(
            'hid', drop=True).squeeze()
    G.communities['type'] = G.communities.name.map(child_map)
    G.communities['color'] = G.communities.type.map(
            {'nc': 'Orange', 'c': 'Green'})
    G.append_community_attribute('', G.communities.color + '!100')
    G.append_community_attribute('fill=', G.communities.color + '!20')
    #G.append_community_attribute('opacity=', '.5')

    # Annotations
    G.appendix = '''
\\node[font=\\footnotesize,text width=2cm] (aa) at (1,.8) {Household without children};
\draw[->] ($(aa.center)+(.2,-.2)$) -- +(1,-.5);
\\node[font=\\footnotesize,text width=2cm] (aa) at (3,4) {Household with children};
\draw[->] ($(aa.center)+(0,-.3)$) -- +(0,-.4);
\\node[font=\\footnotesize] (aa) at (4.5,.5) {Adult};
\draw[->] (aa) -- +(-1.05,-.47);
\\node[font=\\footnotesize] (aa) at (.5,2.8) {Child};
\draw[->] (aa) -- +(1.05,-.25);
'''

## 32: 1.97,1.87
## 62: (1.73,2.56)
## 51: 2.36,3.0

    # Cascade graph
    Gc = copy.deepcopy(G)
    Gc.scope_x = 7 

    # Cascade characteristics
    edges = Gc.edges
    Gc.reset_edge_style()
    edges['directed'] = False
    edges.loc[edges.cid!=0, 'directed'] = True
    edges['color'] = 'black!30'
    edges.loc[edges.cid==1, 'color'] = 'Green!150'
    edges.loc[edges.cid==2, 'color'] = 'Orange!150'
    Gc.append_edge_attribute('', edges.color)

    edges['opacity'] = '0.3'
    edges.loc[edges.cid!=0, 'opacity'] = '1'
    Gc.append_edge_attribute('opacity=', edges.opacity)

    edges['thickness'] = '.5pt'
    edges.loc[edges.cid!=0, 'thickness'] = '1pt'
    Gc.append_edge_attribute('line width=', edges.thickness)

    nodes['opacity'] = '0.3'
    nodes.loc[nodes.cascade==True, 'opacity'] = '1'
    Gc.append_node_attribute('opacity=', nodes.opacity)

    Gc.appendix = '''
\\node[font=\\footnotesize,text width=1cm] (aa) at (3,4.5) {Boundary nodes};
\draw[->] (aa.west) -- +(-.9,-.2);
\draw[->] (aa.south east) -- +(.75,-.38);
\\node[font=\\footnotesize,text width=.6cm, anchor=west] (aa) at (0,.3) {\\texttt{NCHH}};
\draw[Orange!150, line width =1pt, ->] (aa.east) -- +(.5,0);
\\node[font=\\footnotesize,text width=.6cm, anchor=west] (aa) at (0,.6) {\\texttt{CHH}};
\draw[Green!150, line width =1pt, ->] (aa.east) -- +(.5,0);
\\node[font=\\footnotesize, anchor=west] (aa) at (0,1) {Cascade};
\draw[black,rounded corners=1pt] ($(368037032)+(-.1,-.1)$) -- ($(368110762)+(-.12,.05)$) -- ($(368020651)+(-.07,.1)$) -- ++(.1,.05) -- ++(.15,-.2) -- ($(368110762)+(+.12,-.05)$) -- ($(368037032)+(.15,-.07)$) -- cycle;
\\node[font=\\footnotesize,text width=1.5cm, anchor=west, xshift=.7cm] (aa) at (368110762) 
{$c\\rightarrow c\\rightarrow c$ motif};
\draw[->] (aa) -- +(-1.3,0);
'''

    # Convert to tikz
    gtt.to_tikz([G, Gc], global_style=global_style, 
            mode='file', outfile='example.tex',
            colors=gtt.load_color('cb_set3'))
    gtt.compile('example.tex')
    return

def process_cascade(infile=None, id=None, edges=None, num_lines=None):

    # read cascade
    cascade = pd.read_csv(infile).rename(columns={'pid': 'v', 
        'contact_pid': 'u'}).head(num_lines)

    # prepare edges for joining
    idx = cascade.u > cascade.v
    df = cascade.loc[idx].copy()
    ordered_cascade = cascade[['u', 'v']].copy()
    ordered_cascade.loc[idx, ['u', 'v']] = \
            ordered_cascade.loc[idx, ['v', 'u']].values
    ordered_cascade['cid'] = id
    ordered_cascade.loc[idx, 'cid'] = -id
    return ordered_cascade

def main():
    nodes = pd.read_csv('../wsc23/data/networks/richmond_kcore_11_4/nodes.csv')
    edges = pd.read_csv('../wsc23/data/networks/richmond_kcore_11_4/edges.csv')

    # make all edges ascending order
    idx = edges.u > edges.v
    edges.loc[idx, ['u', 'v']] = edges.loc[idx, ['v', 'u']].values
    edges = edges.drop_duplicates(subset=['u', 'v'])

    df = pd.concat([process_cascade(id=1, edges=edges, num_lines=15,
            infile='../wsc23/data/cascades/richmond_kcore_11_4/replicate_1aa/output.csv.gz'),
            process_cascade(id=2, edges=edges, num_lines=15,
            infile='../wsc23/data/cascades/richmond_kcore_11_4/replicate_6aa/output.csv.gz')])

    # merge with edges
    edges = edges.merge(df, left_on=['u', 'v'], right_on=['u', 'v'], 
            how='left').fillna(0)
    idx = edges.cid < 0
    edges.loc[idx, ['u', 'v']] = edges.loc[idx, ['v', 'u']].values
    edges.cid = edges.cid.abs()

    # get nodes in cascades
    df = edges[edges.cid!=0]
    cnodes = pd.concat([df.u, df.v]).drop_duplicates()
    cnodes = pd.Series(index=cnodes, data=True)
    nodes['cascade'] = False
    nodes.cascade = nodes.node.map(cnodes).fillna(False)

    plot_graph(nodes, edges)

if __name__ == '__main__':
    main()
