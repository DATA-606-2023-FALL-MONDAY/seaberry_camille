# functions to download, clean, and write census data with geometry attached

import pandas as pd
import geopandas as gpd
import dotenv
from urllib.parse import urljoin
import requests
import json
from pathlib import Path
import os 
import re


def get_decennial(table, 
                  year = 2020, 
                  sumfile = 'dhc',
                  geo_type = 'block group', 
                  state_fips = '24', 
                  county_fips = '510', 
                  dir = 'data/fetch_data',
                  key_name = 'CENSUS_API_KEY'):
  """Get decennial data from the Census API, and dump into a json file.

  Args:
      table (str): Table number, e.g. 'P5', 'H1', etc.
      year (int, optional): Year. Defaults to 2020.
      sumfile (str, optional): Summary file. Defaults to 'dhc'.
      geo_type (str, optional): Geography level. Defaults to 'block group'.
      state_fips (str, optional): State FIPS code. Defaults to '24'.
      county_fips (str, optional): County FIPS code. Defaults to '510'.
      dir (str, optional): Directory to write json to. Defaults to 'data/fetch_data'.
      key_name (str, optional): Name of API key in .env file. Defaults to 'CENSUS_API_KEY'.

  Returns:
      str: Path to the file written. 
  """  
  # e.g. https://api.census.gov/data/2020/dec/dhc?get=NAME,group(P5)&for=tract:*&in=state:24&in=county:510
  base_url = f'https://api.census.gov/data/{year}/dec/{sumfile}'
  var_str = f'NAME,group({table})'
  geo_type_str = f'{geo_type}:*'
  geo_lvls = {
    'state': state_fips, 
    'county': county_fips
  }
  geo_lvls_str = [f'{k}:{v}' for k, v in geo_lvls.items()]
  
  # get key from dotenv
  dotenv.load_dotenv()
  auth = requests.auth.HTTPBasicAuth('key', os.getenv(key_name))
  
  params = {
    'get': var_str,
    'for': geo_type_str,
    'in': geo_lvls_str
  }
  r = requests.get(base_url, 
                   params = params,
                   auth = auth)
  
  geo_type = re.sub(r'\s', '_', geo_type)
  fn = f'{table}_{geo_type}_{state_fips}{county_fips}_{sumfile}_{year}.json'
  file = os.path.abspath(Path(dir) / fn)
  with open(file, 'w') as f:
    f.write(json.dumps(r.json()))
  return file  

def get_decennial_vars(table, 
                       year = 2020, 
                       sumfile = 'dhc'):
  """Given a decennial census table, get human-friendly variable labels. 

  Args:
      table (str): Table number, e.g. 'P5', 'H1', etc.
      year (int, optional): Year. Defaults to 2020.
      sumfile (str, optional): Summary file. Defaults to 'dhc'.

  Returns:
      dict: A dictionary, where keys are variable IDs and values are corresponding labels.
  """  
  # https://api.census.gov/data/2020/dec/dhc/groups/P5.json
  url = f'https://api.census.gov/data/{year}/dec/{sumfile}/groups/{table}'
  r = requests.get(url)
  variables = {}
  for key, val in r.json()['variables'].items():
    if re.search(r'N$', key):
      lbl = val['label']
      lbl = lbl.strip()
      lbl = re.sub(r'\!\!Total\:', '', lbl)
      lbl = re.sub(r'^\!\!', '', lbl)
      lbl = re.sub(r'\:', '', lbl)
      lbl = re.sub(r'\!\!', '_', lbl)
      if lbl == '':
        lbl = 'Total'
      variables[key] = lbl
  return variables
  
  
def clean_decennial(file):
  """Basic clean-up of decennial data.

  Args:
      file (str): Path to json file. 

  Returns:
      DataFrame: A pandas DataFrame, with nicer column names, extracted GEOID, and no empty columns. 
  """  
  fn = os.path.basename(file)
  table, sumfile, year = re.match(r'^([A-Z]+[0-9]+)_(?:[a-z_]+)_(?:\d{5})_([a-z0-9]+)_(\d{4})', fn).groups()
  with open(file) as f:
    data = json.load(f)
  df = pd.DataFrame(data[1:], columns = data[0])
  variables = get_decennial_vars(table, year, sumfile)
  df = df.rename(columns = variables)
  df = df.dropna(axis = 1, how = 'all', inplace = False)
  geoid = df['GEO_ID'].str.extract(r'(?<=US)([0-9]+$)')
  df = df.drop(columns = ['NAME', 'GEO_ID', 'state', 'county', 'tract'])
  df = df.astype('int64')
  df.insert(0, 'geoid', geoid)
  return df


def calc_race(df):
  """Calculate counts of race/ethnicity. 

  Args:
      df (DataFrame): A pandas DataFrame as returned from table P5. 

  Returns:
      DataFrame: DataFrame with columns for GEOID, total population, and white, Black, Latino, Asian, and other race populations. 
  """  
  total = df['Total']
  white = df['Not Hispanic or Latino_White alone']
  black = df['Not Hispanic or Latino_Black or African American alone']
  latino = df['Hispanic or Latino']
  asian = df['Not Hispanic or Latino_Asian alone']
  other_race = df.loc[:, ['Not Hispanic or Latino_American Indian and Alaska Native alone',
                'Not Hispanic or Latino_Native Hawaiian and Other Pacific Islander alone',
                'Not Hispanic or Latino_Some Other Race alone',
                'Not Hispanic or Latino_Two or More Races']] \
    .sum(axis = 1)
  return pd.DataFrame({
    'geoid': df['geoid'],
    'total': total,
    'white': white,
    'black': black,
    'latino': latino,
    'asian': asian,
    'other_race': other_race
  })
  

def get_tiger(geo_type = 'block group',
              year = 2020,
              state_fips = '24'):
  """Download Census TIGER shapefile and read into a GeoDataFrame.

  Args:
      geo_type (str, optional): Geographic level. Defaults to 'block group'.
      year (int, optional): Year. Defaults to 2020.
      state_fips (str, optional): State FIPS code. Defaults to '24'.

  Returns:
      GeoDataFrame: A geopandas GeoDataFrame with columns for GEOID and geometry.
  """  
  # https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_24_tract.zip
  # https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_24_bg.zip
  if geo_type == 'block group':
    geo_type = 'bg'
  # not actually setting up to handle 2010
  geo_type_upp = geo_type.upper()
  base_url = f'https://www2.census.gov/geo/tiger/TIGER{year}/{geo_type_upp}'
  fn = f'tl_{year}_{state_fips}_{geo_type}.zip'
  url = base_url + '/' + fn
  gdf = gpd.read_file(url)
  gdf = gdf.loc[gdf['ALAND'] > 0, ['GEOID', 'geometry']]
  gdf = gdf.rename(columns = {'GEOID': 'geoid'})
  return gdf
  



if __name__ == '__main__':
  md_fips = '24'
  balt_fips = '510'

  dec_bg = clean_decennial(
    get_decennial('P5')
  )
  race_bg = calc_race(dec_bg)
  bg_shp = get_tiger('block group')
  race_shp = bg_shp.merge(race_bg, on = 'geoid')
  race_shp.to_file('data/input_data/bg_by_race.gpkg', layer = 'bg', driver = 'GPKG')