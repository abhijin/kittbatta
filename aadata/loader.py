DESC = '''
Loading datasets.

By: AA
'''

import geopandas as gpd
from pdb import set_trace
import pkg_resources

DATA = ['usa_tract_shapes', 'usa_county_shapes', 'usa_state_shapes']

def list_data():
    print(f'Datasets:\n{"\n\t".join(DATA)}')

def load(data):
    if data == 'usa_tract_shapes':
        stream = pkg_resources.resource_stream(__name__, 
                'datasets/usa/cb_2018_53_tract_500k.shp')
        gdf = gpd.read_file(stream.name)
        gdf.columns = gdf.columns.str.lower()
        gdf['tract'] = gdf.statefp + gdf.countyfp + gdf.tractce
        return gdf
    elif data == 'usa_county_shapes':
        stream = pkg_resources.resource_stream(__name__, 
                'datasets/usa/cb_2018_us_county_20m.shp')
        gdf = gpd.read_file(stream.name)
        gdf.columns = gdf.columns.str.lower()
        gdf.name = gdf.name.str.lower()
        return gdf
    else:
        raise ValueError(f'{data} not in package.')
