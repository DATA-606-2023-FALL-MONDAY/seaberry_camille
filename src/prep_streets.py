import osmnx as ox
import networkx as nx
import pandas as pd  

def download_streets(loc):
  net = ox.graph_from_place(loc)
  return net

def get_simple_streets(net, tolerance = 15):
  net = ox.project_graph(net)
  net = ox.consolidate_intersections(net, tolerance = tolerance, rebuild_graph = True, reconnect_edges = True)
  return net  

def get_coords(net):
  coords = ox.graph_to_gdfs(net, nodes = True, edges = False)
  coords = coords.reset_index()
  coords = coords.loc[:, ['osmid_original', 'lon', 'lat', 'geometry']]
  coords = coords.rename(columns = { 'osmid_original': 'id' })
  return coords

if __name__ == '__main__':
  balt_dl = download_streets('Baltimore, Maryland, USA')
  # save raw download
  ox.io.save_graph_geopackage(balt_dl, 'data/fetch_data/streets_nx.gpkg')
  balt_net = get_simple_streets(balt_dl)
  balt_gdf = get_coords(balt_net)
  balt_gdf.to_file('data/input_data/street_coords.gpkg', driver = 'GPKG')
