
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from math import radians

'''
File with functions to collect weather data from the nearest weather stations
(data was downloaded before)

'''

def get_haversine_distance(point,weather_station):
      point_radians = [radians(_) for _ in point]
      weather_radians = [radians(_) for _ in weather_station]
      result = haversine_distances([point_radians, weather_radians])
      result = result * 6371000/1000  # multiply by Earth radius to get kilometers
      return result[0][1]

def altitude_difference(altitude,altitude_station):
      return abs(altitude - altitude_station)
def weighted_distance_altitude(distance,altitude_diff,w_d,w_a,d_max, a_max):
      return w_d*distance/d_max + w_a*altitude_diff/a_max

def add_distance_altitude(df, lat, lon, altitude,w_d=0.7,w_a=0.3,d_max=30,a_max=35): 
    df['distance'] = df.apply(lambda row: get_haversine_distance([lat, lon], [row['Latitude'], row['Longitude']]), axis=1)
    df['altitude_difference'] = df.apply(lambda row: altitude_difference(altitude,row['altitude']), axis=1)
    df['weighted_d_a'] = df.apply(lambda row: weighted_distance_altitude(row['distance'],row['altitude_difference'],w_d,w_a,d_max,a_max),axis=1)
    return df 

def find_best_station(df,wind=False):
      if wind:
            wind_df = df[df['complete'] == True]
            nearest_index = wind_df['weighted_d_a'].idxmin()  
      else: 
            nearest_index = df['weighted_d_a'].idxmin()   
      return df.loc[nearest_index, 'Indicator']


def merge_dataframes(df_1, df_2, columns_to_merge=['dir', 'velmedia','racha']):
    result_df = df_1.copy()
    if 'fecha' in result_df.columns:
        result_df.set_index('fecha', inplace=True)
    if 'fecha' in df_2.columns:
        df_2.set_index('fecha', inplace=True)
    combined_index = result_df.index.union(df_2.index)
    result_df_extended = result_df.reindex(combined_index)  
    df2_extended = df_2.reindex(combined_index)    
    result_df_extended = result_df_extended.fillna(df2_extended)
    for col in columns_to_merge:
        if col not in result_df_extended.columns:
            result_df_extended[col] = df2_extended[col]
    result_df_extended.sort_index(inplace=True)
    result_df_extended = result_df_extended.reset_index()
    
    return result_df_extended


def get_weather_data(indicator):
    file_path = f'Daten/Wetterdaten/Galicien/{indicator}/all.csv'
    weather_df = pd.read_csv(file_path)
    weather_df['fecha'] = pd.to_datetime(weather_df['fecha'])
    return weather_df

def drop_nan_rows(df,list_parameters):
    df = df.dropna(subset=list_parameters, how='all')
    return df

        
def add_weather_data(weather_stations,lat,lon,altitude,date=None,day=None):
        # tmed (mittlere Temperatur)
        # prec (Niederschlag)
        # tmin (Minimumtemperatur)
        # tmax (Maximumtemperatur)
        # dir (Windrichtung)
        # velmedia (mittlere Windgeschwindigkeit)
        # racha (Windböen)
        # sol (Sonnenscheindauer)
        # hrMedia (mittlere relative Luftfeuchtigkeit)
        # hrMin (minimale relative Luftfeuchtigkeit)   
   
    list_parameters = ['tmed','tmin','tmax','prec','dir','velmedia','racha','hrMedia','hrMin']
    
    date = pd.to_datetime(date)
    if day:
        date = date.date()
    weather_stations_filtered = weather_stations.copy()
    if date.year == 2021:
        weather_stations_filtered = weather_stations[~weather_stations['missing_years'].apply(lambda x: '2021' in x)]
    if date.year == 2020:
        weather_stations_filtered = weather_stations[~weather_stations['missing_years'].apply(lambda x: '2020' in x)]
    weather_stations_filtered = add_distance_altitude(weather_stations_filtered,lat,lon,altitude)    
    weather_data = {}
    date_given = day
    
    while(len(list_parameters) != 0):
        station_i = find_best_station(weather_stations_filtered)
        weather_df = get_weather_data(station_i)              
        if 'racha' not in weather_df:
            weather_stations_filtered = weather_stations_filtered[weather_stations_filtered['Indicator'] != station_i]
            station_i = find_best_station(weather_stations_filtered, wind=True)
            weather_df_2 = get_weather_data(station_i)
            weather_df = merge_dataframes(weather_df,weather_df_2)
            
        
        weather_df = drop_nan_rows(weather_df,list_parameters=list_parameters)

        
        if day:
            weather_df = weather_df[weather_df['fecha'].dt.date == date]
            
        else: 
            weather_df = weather_df[weather_df['fecha'].dt.year == date.year]
            weather_df = weather_df[weather_df['fecha'].dt.month == date.month]
            if not weather_df.empty:
                weather_df = weather_df.sample(1)
                date = weather_df['fecha'].dt.date.values[0]
                day = True
            else:
                weather_stations_filtered = weather_stations_filtered[weather_stations_filtered['Indicator'] != station_i]
                continue

        weather_stations_filtered = weather_stations_filtered[weather_stations_filtered['Indicator'] != station_i]


        if not weather_df.empty:
            l = list_parameters.copy()
            for parameter in l:
                value = weather_df[parameter].values[0]
                if not pd.isnull(value):
                    if parameter == 'prec' and value == 'lp':
                        print("value: lp")
                        value = 0
                    else:
                        try:
                            value = value.replace(',', '.') 
                            value = float(value)
                        except AttributeError:
                            pass
                        except ValueError:
                            continue
                    weather_data[parameter] = value
                    list_parameters.remove(parameter)

    if not date_given:
        weather_data['date'] = date
    return weather_data



def apply_add_weather(df,label):
    weather_stations = pd.read_csv("Daten/Wetterdaten/Galicien/weather_stations_coordinates_missing_data.csv")
    if label:
        df = df.join(df.apply(lambda row: pd.Series(add_weather_data(weather_stations,row['lat'],row['lon'],row['altitude'],pd.to_datetime(row['date']),label)),axis=1))
    else:
        df = df.join(df.apply(lambda row: pd.Series(add_weather_data(weather_stations,row['lat'],row['lon'],row['altitude'],row['month'],label)),axis=1))     
    return df
