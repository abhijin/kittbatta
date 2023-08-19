import numpy as np
import pandas as pd
from pdb import set_trace
from subprocess import check_call
import sys

# read network
df = pd.read_csv(sys.argv[1])

# reset nodes to 0,...,n
nodes = pd.concat([df.u, df.v]).drop_duplicates().reset_index(drop=True)
node_map = pd.Series(index=nodes, data=np.arange(nodes.shape[0]))
inv_node_map = pd.Series(index=np.arange(nodes.shape[0]), data=nodes)

df.u = df.u.map(node_map)
df.v = df.v.map(node_map)

# write to format for greedy
net_file_for_greedy = f'{sys.argv[1]}.net'
with open(net_file_for_greedy, 'w') as f:
    f.write(f'{node_map.shape[0]}\n')
    df.to_csv(f, index=False, sep=' ', header=False)

# Obtain coloring
color_file = f'{sys.argv[1]}.color'

with open(color_file, 'w') as f:
    check_call(['./greedy', net_file_for_greedy], stdout=f)

# Get colors
colors = pd.read_csv(color_file, names=['node','color'], sep=' ')
colors.node = colors.node.map(inv_node_map)

# Max. independent set
print(colors[colors.color == colors.color.value_counts().argmax()])

