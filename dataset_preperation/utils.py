import datetime as dt                                      
import rasterio                                          
import numpy as np                                       
import random
import calendar
import pandas as pd


def get_month_range(year, month):    
        first_day = dt.date(year, month, 1)
        last_day = dt.date(year, month, calendar.monthrange(year, month)[1])
        return (first_day,last_day)

def get_samples(start_year,end_year,df,season=None):
    df['lon'] = None
    df['lat'] = None
    df['forest_type'] = None
    df['month'] = None
    
    land_cover_path = "Daten/Corine Land Cover/Results/U2018_CLC2018_V2020_20u1/U2018_CLC2018_V2020_20u1/U2018_CLC2018_V2020_20u1.tif"
    with rasterio.open(land_cover_path) as src:
        data = src.read(1) 
        wald_maske = np.isin(data, [20,21,22,23,24,25,26,27,28,29]) 
        wald_daten = np.where(wald_maske, data, 0) 
        forest_areas = wald_daten != 0
        valid_data_points = np.argwhere(forest_areas)
        for index,row in df.iterrows():
            random_coordinate = random.choice(valid_data_points)
            row, col = random_coordinate
            x, y = src.xy(row, col)
            random_value = wald_daten[row, col]
            df.loc[index,'lon'] = x
            df.loc[index,'lat'] = y
            df.loc[index,'forest_type'] = random_value
            year = random.randint(start_year, end_year)
            if not season:
                month = random.randint(1, 12) 
            else:
                month = random.randint(4, 9)
            date = dt.date(year, month, 1)
            df.loc[index,'month'] = date

def get_month(df):
    for index,row in df.iterrows():
        month = dt.date(row['date'].year, row['date'].month, 1)
        df.loc[index,'month'] = month


def truncate(num, precision=4):
    return float(int(num * (10 ** precision))) / (10 ** precision)

def compare_lat_lon(lon,lat,df):
    lon_df = df['lon']
    lat_df = df['lat']
    return (truncate(lon) in [truncate(v) for v in lon_df.values] and truncate(lat) in [truncate(v) for v in lat_df.values])

def delete_dublicates(df):
    df = df.drop_duplicates(subset=['lon', 'lat', 'date'], keep='first')
    return df

def assign_signature_list(row, df_boundaries):
    boundaries = []
    for _, boundary in df_boundaries.iterrows():
        if (boundary['lon_min'] <= row['lon'] <= boundary['lon_max'] and
            boundary['lat_min'] <= row['lat'] <= boundary['lat_max']):
            boundaries.append(boundary['signature'])
    

    if len(boundaries) == 0:
        print(f"Keine passenden tiles für {row['id']}.")
        return None
    else:
        return boundaries

def assign_signature(row, df_boundaries):
    for _, boundary in df_boundaries.iterrows():
        if (boundary['lon_min'] <= row['lon'] <= boundary['lon_max'] and
            boundary['lat_min'] <= row['lat'] <= boundary['lat_max']):
            return boundary['signature']
    return None  

def create_cluster(df,month=None,round=1):
    
    df['lat_rounded'] = df['lat'].round(round)
    df['lon_rounded'] = df['lon'].round(round)

    if month:   
        grouped = df.groupby(['month', 'lat_rounded', 'lon_rounded'])
    else:
        grouped = df.groupby(['date', 'lat_rounded', 'lon_rounded'])

    df['cluster'] = grouped.ngroup()
    df = df.drop(columns=['lat_rounded','lon_rounded'])
    return df



def map_tiles(df):
    df_boundaries = pd.read_csv('Daten/Vegetation/tile_boundings.csv')
    df['signature'] = None
    for index,row in df.iterrows():
        df.at[index,'signature'] = assign_signature_list(row,df_boundaries)
    df_exploded = df.explode('signature')
    grouped = df_exploded.groupby(['signature', 'month'])
    df_exploded['cluster'] = grouped.ngroup()
    df_merged = df_exploded.groupby(['id']).agg({'cluster':list})
    df_result = df.merge(df_merged,on='id')    
    df_mapping = df_exploded[['signature', 'month', 'cluster']].drop_duplicates()
    df_mapping = df_mapping.reset_index()
    return df_result,df_mapping

def average_ndvi(ndvi_values): 
    if type(ndvi_values) == int:
        print(ndvi_values)
        return False
    return sum(ndvi_values)/len(ndvi_values)



