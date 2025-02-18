from utils import *
import pandas as pd
from prepare_topo_data import prepare_topo_data_grouped_gdf
from prepare_weather_data import apply_add_weather
from prepare_ndvi_lai import finalize,failed_gdal,apply_ndvi_per_cluster
import numpy as np

'''
In this file functions to create a dataset out of the different gathering steps are implemented

'''

def weather_and_save(data,label,save_path):
    data = apply_add_weather(data,label)
    data.to_csv(save_path,index=False)
    return data

def topo_and_save(data,save_path):
    df_path_bounding = "Daten/Topografie/Galicia/list_bounding_box.csv"
    data = prepare_topo_data_grouped_gdf(data,df_path_bounding)
    data = data[~data['altitude'].apply(lambda x: pd.isna(x))]
    data.to_csv(save_path,index=False)
    return data

def load_data_pos(year=None,path=1):
    if path == 1:
        pos_samples = pd.read_csv("Daten/Thermal anomalies VIIRS FIRMS/forest_fires_2020_23_gal.csv")
    else:  
        pos_samples = pd.read_csv("Daten/Modis Nasa/fire_data/forest_fires_2020_24_gal.csv")
    
    df = pd.DataFrame()
    df.loc[:, 'id'] = range(len(pos_samples))
    df[['lon','lat','date','confidence']] = pos_samples[['longitude','latitude','acq_date','confidence']] 
    df['date'] = pd.to_datetime(df['date'])
    if year:
        df = df[df['date'].dt.year == year]
    else:
        
        df = df[df['date'].dt.year < 2024]
    df['forest_type'] = None
    get_month(df)
    
    return df
    

def prep_dataset_neg(data_num,season = True,pos_samples_path = None, year=None,num_samples=None):
    saving_path = f'Daten/Dataset/neg_samples/all_{data_num}.csv'
    if season:
        saving_path = f'C:/Users/clara/OneDrive/Dokumente/Studium/Bachelorarbeit/Coding/Daten/Dataset/neg_samples/all_Apr_Sep{data_num}.csv'
    if not num_samples:
        pos_samples = pd.read_csv(pos_samples_path)

        if year:
            num_samples = len(pos_samples[pd.to_datetime(pos_samples['date']).dt.year == year])
            saving_path = f'Daten/Dataset/neg_samples/{year}_{data_num}.csv'
        else: 
            num_samples = len(pos_samples)
            
    df = pd.DataFrame()
    df['id'] = None
    for i in range(num_samples):
        df.loc[i, 'id'] = i
    if year:
        get_samples(year,year,df) 
    else:
        get_samples(2020,2022,df,season) 

    print(f'{num_samples} samples wurden generiert. Topografiedaten werden geladen...')
    df.to_csv(saving_path,index=False)
    return df,saving_path
    
    
def prep_dataset(save_path,label,load_new=None,data_num=None,ndvi=None):
    if load_new:
        if label == 1:
            data = load_data_pos()
    else:
        data = pd.read_csv(save_path)
    print(f'Es werden {len(data)} Daten bearbeitet...')
    df = topo_and_save(data,save_path)
    print(f'Topografiedaten wurden berechnet und gespeichert. Wetterdaten werden geladen...')
    df = weather_and_save(df,label,save_path)
    len_before = len(df)
    df = delete_dublicates(df)
    print(f"Keep {len(df)} out of {len_before}")
    print(f'Wetterdaten wurden berechnet und gespeichert.')
    if ndvi:
        df,df_mapping  = map_tiles(df)
        print("Ndvi Werte werden gesammelt...")
        username = input("Provide the Terrascope username")
        password = input("Provide the Terrascope password")
        apply_ndvi_per_cluster(save_path,data_num,username,password,df,df_mapping,load_new=True)
        if failed_gdal(save_path):
            finalize(save_path)




