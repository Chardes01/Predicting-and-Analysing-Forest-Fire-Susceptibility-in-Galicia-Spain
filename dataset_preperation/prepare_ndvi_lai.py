                                                                       
import matplotlib.pyplot as mplot                         
import numpy as np                                        
from terracatalogueclient import Catalogue               
from pyproj import Transformer
import time
from osgeo import gdal
import pickle
from utils import *
import pandas as pd
import ast
import traceback
import copy
from tqdm.notebook import tqdm

'''
In this functions to efficiently gather NDVI data for a grouped dataset are implemented.

'''



def get_collections(catalogue,platform_name):
    return list(catalogue.get_collections(platform=platform_name))

def get_products_sample(lon,lat,date_range,catalogue):
    s2_platform = 'Sentinel-2'
    s2_collections = get_collections(catalogue,s2_platform)
    ndvi_collection = s2_collections[23]
    products = list(catalogue.get_products(ndvi_collection,
                                        start=date_range[0],
                                        end=date_range[1],
                                        bbox= f'{lon},{lat},{lon},{lat}',
                                        )) 
    return products


def get_ndvi_values_cluster_gdal(date_range, df_cluster,signature,cluster,username,password,catalogue):

    min_lon,min_lat,max_lon,max_lat = df_cluster['lon'].min(),df_cluster['lat'].min(),df_cluster['lon'].max(),df_cluster['lat'].max()

    average_lon = (min_lon + max_lon)/2
    average_lat = (min_lat + max_lat)/2
    products = get_products_sample(average_lon, average_lat, date_range,catalogue)
    
    print(f"Original length products: {len(products)}")
    products = [p for p in products if signature in p.data[0].href]
    result_dict = {}
    for p in products:
        s = p.data[0].href
        key = s.lower().replace('v200', 'vXXX').replace('v210', 'vXXX')
        if key not in result_dict or 'v210' in s.lower():
            result_dict[key] = s
    filtered_products = list(result_dict.values())
    if len(filtered_products) != len(products):
        print([p.data[0].href for p in products])


    max_retries = 2
    temp_ndvi = copy.deepcopy(df_cluster['ndvi'].apply(lambda x: x.copy() if isinstance(x, list) else x))
    temp_failed = copy.deepcopy(df_cluster['failed'].apply(lambda x: x.copy() if isinstance(x, list) else x))
        

    with tqdm(total=len(filtered_products), desc="Process", unit="products",position=1) as pbar:
        pbar.set_description(f'Process {len(filtered_products)} products for cluster {cluster}, sinature: {signature}, size: {len(df_cluster)}:')
        for i, link in enumerate(filtered_products):            
            attempt = 0
            
            while attempt < max_retries:
               
                try:
                    login_string = f'{username}:{password}'
                    gdal.SetConfigOption("GDAL_HTTP_AUTH", "BASIC")
                    gdal.SetConfigOption("GDAL_HTTP_USERPWD", login_string)
                    gdal.SetConfigOption("GDAL_ENABLE_VSIL_CURL", "YES")

                    dataset = gdal.Open(f"/vsicurl/{link}")
                    if dataset is None:
                        raise RuntimeError(f"Fehler beim Öffnen der Datei: {link}")
                    else:
                        transform = dataset.GetGeoTransform()
                        crs_wkt = dataset.GetProjection()
                        band = dataset.GetRasterBand(1)
                        transformer = Transformer.from_crs("epsg:4326", crs_wkt, always_xy=True)

                        for index, point in df_cluster.iterrows():
                            lon,lat = point['lon'],point['lat']
                            easting, northing = transformer.transform(lon, lat)                       
                            col = int((easting - transform[0]) / transform[1])
                            row = int((northing - transform[3]) / transform[5])

                            try_again = 0
                            while(try_again < 2):    
                                try:
                                    value = band.ReadAsArray(col, row, 1, 1)[0, 0]
                                    if value != 255:
                                            temp_ndvi[index].append(value)
                                    break
                                except IndexError:
                                    break
                                except RuntimeError as e:
                                    if try_again == 1:
                                        print("Wait 15 seconds")
                                        time.sleep(15)
                                    if try_again == 2:
                                        print(f"\033[GdalError aufgetreten: {str(e)} for point: {index}\033[0m")
                              
                                try_again += 1 
                        pbar.update(1)
                        break
                    
                except RuntimeError as e:
                    if attempt == 0:
                        print(f"Runtime error aufgetreten: {str(e)}")
                        print("wait 15 seconds")
                        time.sleep(15)

                    attempt += 1
          

            if attempt == max_retries:
                print(f"\033[91mFailed for {i}: {link}\033[0m")          
                temp_failed = temp_failed.apply(lambda x: x + [link])
                pbar.update(1)

    return temp_ndvi,temp_failed


def get_ndvi_by_link_gdal(df_cluster,link,username,password):

    max_retries = 2
    temp_ndvi = copy.deepcopy(df_cluster['ndvi'].apply(lambda x: x.copy() if isinstance(x, list) else x))
    temp_failed = copy.deepcopy(df_cluster['failed'].apply(lambda x: x.copy() if isinstance(x, list) else x))

    attempt = 0
        
    while attempt < max_retries:
        try:

            login_string = f'{username}:{password}'
            gdal.SetConfigOption("GDAL_HTTP_AUTH", "BASIC")
            gdal.SetConfigOption("GDAL_HTTP_USERPWD", login_string)
            gdal.SetConfigOption("GDAL_ENABLE_VSIL_CURL", "YES")

            dataset = gdal.Open(f"/vsicurl/{link}")
            if dataset is None:
                raise RuntimeError(f"Fehler beim Öffnen der Datei: {link}")
            else:
                transform = dataset.GetGeoTransform()
                crs_wkt = dataset.GetProjection()
                band = dataset.GetRasterBand(1)

                transformer = Transformer.from_crs("epsg:4326", crs_wkt, always_xy=True)

                for index, point in df_cluster.iterrows():
                    lon,lat = point['lon'],point['lat']
                    easting, northing = transformer.transform(lon, lat)                       

                    col = int((easting - transform[0]) / transform[1])
                    row = int((northing - transform[3]) / transform[5])

                    try_again = 0
                    while(try_again < 2):    
                        try:
                            value = band.ReadAsArray(col, row, 1, 1)[0, 0]
                            if value != 255:
                                    temp_ndvi[index].append(value)
                            break
                        except IndexError:
                            break
                        except RuntimeError as e:
                            if try_again == 1:
                                print("Wait 15 seconds")
                                time.sleep(15)
                            if try_again == 2:
                                print(f"\033[GdalError aufgetreten: {str(e)} for point: {index}\033[0m")
                        
                        try_again += 1 
                             
                temp_failed = temp_failed.apply(lambda x: [item for item in x if item != link])         
                break
            
        except RuntimeError as e:
            if attempt == 0:
                print(f"Runtime error aufgetreten: {str(e)}")
                print("wait 15 seconds")
                time.sleep(15)

            attempt += 1
    

    if attempt == max_retries:
        print(f"\033[91mFailed again for{link}\033[0m")       

    return temp_ndvi,temp_failed





    
def load_state_cluster(saving_path,id):
    try:
        df = pd.read_csv(saving_path,converters={'cluster': ast.literal_eval,'ndvi': ast.literal_eval,'failed': ast.literal_eval})
        df_mapping = pd.read_csv(f'Daten/Dataset/mapping_{id}.csv')
        with open(f'Daten/Dataset/ndvi_cluster_state_{id}.pkl', 'rb') as f:
            state = pickle.load(f)
        
        print(f"Wird fortgesetzt ab: {state[0]}. Bereits {len(state[1])} clusters abgeschlossen. Abgeschlossene clusters: {state[1]}")
        return df,df_mapping,state[0],state[1]
    except FileNotFoundError:
        return None



def apply_ndvi_per_cluster(saving_path,id,username,password,df=None,df_mapping=None,give_index=None,load_new=None,gdal=True,cluster=None):
    catalogue = Catalogue().authenticate_non_interactive(username, password)
    if not load_new:
        df,df_mapping,start_index,old_cluster = load_state_cluster(saving_path,id)
    else:
        start_index = 0
        df['ndvi'] = [[] for _ in range(len(df))]
        df['failed'] = [[] for _ in range(len(df))]
        df_mapping.to_csv(f'Daten/Dataset/mapping_{id}.csv')
        old_cluster = []

    if give_index:
        start_index = give_index

    df_mapping['month'] = pd.to_datetime(df_mapping['month'])
    if cluster:
        df_mapping = df_mapping[df_mapping['cluster'] == cluster]
        start_index = 0

    try:
        with tqdm(total=len(df_mapping), initial=len(old_cluster),desc="Process", unit="clusters",position=0) as pbar:
            for index, row in df_mapping.loc[start_index:].iterrows():
                update_successful = False
                try:
                    signature = row['signature']
                    cluster = row['cluster']
                    date_range = get_month_range(row['month'].year,row['month'].month)
                    df_cluster = copy.deepcopy(df[df['cluster'].apply(lambda x: cluster in x)])
                    pbar.set_description(f'Process cluster of {len(df_cluster)}, cluster: {cluster}, signature: {signature}')      
                    temp_ndvi,temp_failed = get_ndvi_values_cluster_gdal(date_range,df_cluster,signature,cluster,username,password,catalogue)               
                    df_cluster['ndvi'] = temp_ndvi
                    df_cluster['failed'] = temp_failed
    
                    try:
                        df.update(df_cluster)
                        old_cluster.append(cluster)
                        update_successful = True
                        pbar.update(1)
                    except Exception as update_error:
                        print(f"Error updating df: {update_error}")
                    
                        
                except Exception as e:
                    print(f"Error at index {index}: {e}")
                    print(f"type: {type(e).__name__}")
                    print(f"message: {str(e)}")
                    print("Traceback:")
                    traceback.print_exc()
                finally:
                    if update_successful:
                        df.to_csv(saving_path, index=False)
                        with open(f'Daten/Dataset/ndvi_cluster_state_{id}.pkl', 'wb') as f:
                            pickle.dump((index + 1,old_cluster), f)
            
    except KeyboardInterrupt:
        df.to_csv(saving_path, index=False)
        print(old_cluster)
        with open(f'Daten/Dataset/ndvi_cluster_state_{id}.pkl', 'wb') as f:
            pickle.dump((index,old_cluster), f)

        print(f'\nProgramm unterbrochen bei Index {index} Zustand gespeichert.')
        return df

    print(f'NDVI abgeschlossen.')
    return df



def failed_gdal(df_path,username,password,unique_failures=None):
    catalogue = Catalogue().authenticate_non_interactive(username, password)
    df = pd.read_csv(df_path, converters={'ndvi': ast.literal_eval,'failed': ast.literal_eval})
    if not unique_failures:
        unique_failures = set([item for sublist in df['failed'] for item in sublist])
    print(f'Collect data from {len(unique_failures)} links...')
    complete = False

    for failure in unique_failures:
        print(failure)
        df_failed = copy.deepcopy(df[df['failed'].apply(lambda x: failure in x)])
        print("Process: ",len(df_failed))
        temp_ndvi,temp_failed = get_ndvi_by_link_gdal(df_failed,failure,username,password,catalogue)
    
        df_failed['ndvi'] = temp_ndvi
        df_failed['failed'] = temp_failed
        df.update(df_failed)
    unique_failures = set([item for sublist in df['failed'] for item in sublist])
    print(f'After: {len(unique_failures)}')    
    if len(unique_failures) == 0:
        complete = True
    df.to_csv(df_path,index=False)
    return complete


def finalize(df_path):
    df = pd.read_csv(df_path, converters={'ndvi': ast.literal_eval})    
    print(f"len before: {len(df)}")
    df = df[df['ndvi'].apply(lambda x: len(x) > 0)]
    df = df.drop(columns='failed')
    print(f"len after: {len(df)}")
    df.to_csv(df_path,index=False)
    










