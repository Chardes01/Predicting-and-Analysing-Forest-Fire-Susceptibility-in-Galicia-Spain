import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from shapely.geometry import box
from tqdm.notebook import tqdm

'''
In this file the collection of topographical attributes is implemented 
(the data have to be downloaded before and RichDEM was used to create the respective attribute files)

'''

    
def read_topo_data_group(path,slope_path,aspect_path,curvature_path,df):
    try:
        with rasterio.open(path) as dataset, \
            rasterio.open(slope_path) as slope_src, \
            rasterio.open(aspect_path) as aspect_src, \
            rasterio.open(curvature_path) as curvature_src:

            if 'HU29' in path:
                crs = 'EPSG:25829'
            else:
                crs = 'EPSG:25830'
 
            transform = dataset.transform
            transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
            for index, row in df.iterrows():
                longitude,latitude = row['Longitude'],row['Latitude']
                easting, northing = transformer.transform(longitude, latitude)                
                row, col = rowcol(transform, easting, northing)
                window = Window(col, row, 1, 1)

                try:
                    value = dataset.read(1, window=window)[0, 0]
                    slope = slope_src.read(1, window=window)[0, 0]
                    aspect = aspect_src.read(1, window=window)[0, 0]
                    curvature = curvature_src.read(1, window=window)[0, 0]
                    df.loc[index,'altitude'] = value
                    df.loc[index,'slope'] = slope
                    df.loc[index,'aspect'] = aspect
                    df.loc[index,'curvature'] = curvature

                    
                except IndexError:
                    print("Index error")
                    df.loc[index,'altitude'] = False
                    pass
    except rasterio.errors.RasterioIOError:
        print(f"Fehler beim Öffnen von {path}")
    return df
    
        

def prepare_topo_data_grouped(df,df_path_bounding,num_path = 0):   
    df['altitude'] = None
    df['slope'] = None
    df['aspect'] = None
    df['curvature'] = None
   

    for index,point in df.iterrows():
        start_path = 0
        lon,lat = point['Longitude'],point['Latitude']
        
        for _,row in df_path_bounding.iterrows():
            lat_min, lon_min = row['lat_min'],row['lon_min']
            lat_max,lon_max = row['lat_max'],row['lon_max']
            if lon < lon_max and lon > lon_min and lat < lat_max and lat > lat_min:
                if start_path < num_path:
                    start_path += 1
                    continue
                path = row['path']
                df.loc[index,'path'] = path
                break
    for _,row in df_path_bounding.iterrows():
        path = row['path']     
        print(f'Process path: {path}')   
        slope_path,curvature_path,aspect_path = row['slope_path'],row['curvature_path'],row['aspect_path']
        df_relevant = df[df['path'] == path]
        read_topo_data_group(path,slope_path,aspect_path,curvature_path,df_relevant)
        df.update(df_relevant)

    return df
    

def process_group(group):
    path = group['path'].iloc[0]
    slope_path = group['slope_path'].iloc[0]
    aspect_path = group['aspect_path'].iloc[0]
    curvature_path = group['curvature_path'].iloc[0]
    print(f"Process path: {path} and {len(group)} points")
    return read_topo_data_group(path, slope_path, aspect_path, curvature_path, group)

def prepare_topo_data_grouped_gdf(df, df_path_bounding, num_path=0):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    gdf_bounds = gpd.GeoDataFrame(df_path_bounding, geometry=gpd.points_from_xy(df_path_bounding.lon_min, df_path_bounding.lat_min))
    gdf_bounds['geometry'] = gdf_bounds.apply(lambda row: box(row.lon_min, row.lat_min, row.lon_max, row.lat_max), axis=1)

    gdf = gpd.sjoin(gdf, gdf_bounds, how="left", op="within")
    if num_path > 0:
        gdf = gdf[gdf.index_right >= num_path]
    total_groups = len([group for _, group in gdf.groupby('path')])

    with ProcessPoolExecutor() as executor:
        with tqdm(total=total_groups, desc="Process cluster") as pbar:
            futures = []
            for _, group in gdf.groupby('path'):
                future = executor.submit(process_group, group)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)
            
            results = [future.result() for future in futures]

    df_result = pd.concat(results)    
    return df_result





