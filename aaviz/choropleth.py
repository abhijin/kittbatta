DESC='''
Visualizing communities of a spatial network on a map.

AA
'''

import argparse
import geopandas as gpd
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import devnull
import pandas as pd
from pdb import set_trace

import plot
## import tikz_network as tikz

FORMAT="%(levelname)s:%(funcName)s:%(message)s"

def world():
    return gpd.GeoDataFrame.from_file('../data/world/ne_50m_admin_0_countries.shp')

def contiguous_US():
    return gpd.GeoDataFrame.from_file('../data/world/ne_50m_admin_0_countries.shp')

def main():

    # parser
    parser=argparse.ArgumentParser(description=DESC, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--nodes", required=True,
            help="Node attributes containing some mandatory columns.")
    parser.add_argument("--node_column", required=True,
            help="Column to join with shape.")
    parser.add_argument("--community_column", required=True,
            help="Column name for community.")
    parser.add_argument("--nodes_to_exclude", nargs='*', default="",
            help="Nodes that the user would like to exclude from being plotted on the map. This helps plot subregions of a map.")
    parser.add_argument("-s", "--shape", required=True,
            help="Input GADM shape.")
    parser.add_argument("--shape_column", required=True,
            help="Column to join with nodes.")
    parser.add_argument("--text",
            help="Expects 'text' column.")

    # Default is the web mercarator
    parser.add_argument("--vsize", type = int, default = 10, help = "vertex size")
    parser.add_argument("-T", "--title", help = "Title of the plot", default = '')
    parser.add_argument("-o", "--out_file", help = "Output file", default = 'out.pdf')
    parser.add_argument("--num_top_clusters", type = int, 
            help = "Number of clusters to consider by size", default = 8)

    parser.add_argument("-d", "--debug", action = "store_true")
    parser.add_argument("-q", "--quiet", action = "store_true")
    args = parser.parse_args()

    # set logger
    if args.debug:
       logging.basicConfig(level=logging.DEBUG,format=FORMAT)
    elif args.quiet:
       logging.basicConfig(level=logging.WARNING,format=FORMAT)
    else:
       logging.basicConfig(level=logging.INFO,format=FORMAT)

    logging.info('Loading shape file ...')
    if args.shape == 'world':
        #shape = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        shape = world()
    else:
        shape = gpd.GeoDataFrame.from_file(args.shape)

    logging.info('Loading node information ...')
    nodes = pd.read_csv(f'{args.nodes}')

    logging.info('Aligning index data types, and joining nodes and shape ...')
    node_index_type = nodes.dtypes[args.node_column]
    nodes = nodes.set_index(args.node_column)
    shape = shape.astype({args.shape_column: node_index_type})
    shape = shape.set_index(args.shape_column)
    nodes = nodes.join(shape)

    logging.info('Checking data ...')
    unresolved = nodes[nodes.geometry.isnull()].index.tolist()
    logging.warning(f'Unresolved rows with null geometry: {unresolved}')
    nodes = nodes[~nodes.geometry.isnull()]
    unresolved = nodes[nodes[args.community_column].isnull()].index.tolist()
    logging.warning(f'Unresolved rows with null community ID: {unresolved}')
    nodes = nodes[~nodes[args.community_column].isnull()]

    logging.info('Generating figure axis and map ...')
    fig = plt.figure()
    ax = fig.add_subplot()

    logging.info('Selecting clusters to plot ...')
    top_clusters = nodes[args.community_column].value_counts().head(args.num_top_clusters).index.to_numpy()
    nodes = nodes[nodes[args.community_column].isin(top_clusters)]
    cluster_id_remap = pd.Series(range(1,len(top_clusters)+1),index=top_clusters)
    nodes['cluster'] = nodes[args.community_column].map(cluster_id_remap)
    
    logging.info(f'Dropping nodes to exclude: {args.nodes_to_exclude} ...')
    # converting data type
    nodes_to_exclude = np.array(args.nodes_to_exclude).astype(node_index_type)
    try:
        shape = shape.drop(nodes_to_exclude, axis=0)
    except TypeError:
        pass

    logging.info('Plotting clusters ...')

    shape.boundary.plot(ax=ax, edgecolor='black', lw=.3, alpha=.1)
    ## shape = shape.apply(lambda x: ax.annotate(text=x['CFS17_NAME'], 
    ##     xy=x.geometry.centroid.coords[0], ha='center'), axis=1);
    
    # plot map
    gdf = gpd.GeoDataFrame(nodes)
    gdf.plot(ax=ax, column='cluster', legend=True, 
            legend_kwds={'frameon': False},
            categorical=True, 
            cmap='Spectral')
    ax.set_title(args.title)

    plt.axis('off')
    # plt.savefig(args.out_file, bbox_inches='tight')
    plt.savefig(args.out_file, bbox_inches='tight', dpi=1000)

if __name__ == "__main__":
    main()
