DESC='''
Helper function for the digital twin figure.

By: AA
'''
import pandas as pd
from pdb import set_trace

import aaviz.graph_to_tikz as gtt

# L1 graph
nodes = pd.DataFrame({'name': [1,2,3,4,5]})
edges = pd.DataFrame({
    'source': [1,1,1,2,2,3,4],
    'target': [2,3,5,3,4,4,5]})
G = gtt.GraphToDraw(nodes,edges)
G.layout('spring', seed=0)
G.layout_scale_round(1,3,4,7,padding=.5)
G.set_global_style('agent/.style={circle,black!60,inner sep=2pt}')
G.append_node_attribute('', 'agent')
G.append_edge_attribute('', 'black!60')
print(gtt.draw(G, mode='segment'))

# L2 graph
nodes = pd.DataFrame({'name': [1,2,3,4]})
edges = pd.DataFrame({
    'source': [1,1,2,3],
    'target': [2,4,3,4]})
G = gtt.GraphToDraw(nodes,edges)
G.layout('spring', seed=0)
G.layout_scale_round(3,7,5,7,padding=.5)
G.append_node_attribute('', 'agent')
G.append_edge_attribute('', 'black!60')
print(gtt.draw(G, mode='segment'))

# L3 graph
nodes = pd.DataFrame({'name': [1,2,3,4,5,6]})
edges = pd.DataFrame({
    'source': [1,1,1,2,2,2,3,3,3,4,4,5],
    'target': [2,3,4,3,4,5,4,5,6,5,6,6]})
G = gtt.GraphToDraw(nodes,edges)
G.layout('spring', seed=0)
G.layout_scale_round(1,4,1,4,padding=.5)
G.append_node_attribute('', 'agent')
G.append_edge_attribute('', 'black!60')
print(gtt.draw(G, mode='segment'))

# L4 graph
nodes = pd.DataFrame({'name': [1,2,3]})
edges = pd.DataFrame({
    'source': [1,1,2],
    'target': [2,3,3]})
G = gtt.GraphToDraw(nodes,edges)
G.layout('spring', seed=0)
G.layout_scale_round(4,7,3,5,padding=.5)
G.append_node_attribute('', 'agent')
G.append_edge_attribute('', 'black!60')
print(gtt.draw(G, mode='segment'))

# L5 graph
nodes = pd.DataFrame({'name': [1,2,3,4]})
edges = pd.DataFrame({
    'source': [1,1,2,3],
    'target': [2,3,3,4]})
G = gtt.GraphToDraw(nodes,edges)
G.layout('spring', seed=0)
G.layout_scale_round(4,7,1,3,padding=.5)
G.append_node_attribute('', 'agent')
G.append_edge_attribute('', 'black!60')
print(gtt.draw(G, mode='segment'))

