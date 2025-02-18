import pandas as pd
import requests
import os
import time

'''
Crawler to collect weather data from @AEMET when having a file with weather station Indicators and coordinates
Also an api key has to be inquired

'''

df = pd.read_csv("Daten/Wetterdaten/Galicien/weather_stations_coordinates_galicia.csv")
base_url = 'https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{}/fechafin/{}/estacion/{}'

api_key = 'loremipsum' #### api key has to be inquired at @AEMET

months = ['04', '05', '06', '07', '08', '09']
years = [2020,2021,2022]

results = []
start_index = 0
for indicator in df['Indicator'].iloc[start_index:]:
    print(indicator)
    for year in years:
        for months in range(2):
            if months == 0:
                start_date = f"{year}-01-01T00:00:00UTC"
                end_date = f"{year}-06-30T23:59:59UTC"
            else: 
                start_date = f"{year}-07-01T00:00:00UTC"
                end_date = f"{year}-12-31T23:59:59UTC"
            url = base_url.format(start_date, end_date, indicator)
            go_on = True
            while(go_on):
                response = requests.get(url, headers={'api_key': api_key})
                
                if response.status_code == 200:
                    go_on = False
                    data = response.json()
                    print(data)
                    if 'datos' in data:
                        print(True)
                        
                        link = data['datos']
                        
                    else:
                        description = data['descripcion']
                        if description == 'No hay datos que satisfagan esos criterios':
                            print(f'No data for {indicator} and year half {year}/{months}')
                            break
                        break
                    print("link appended")
                    response2 = requests.get(link, headers={'api_key': api_key})
                    if response2.status_code == 200:
                        data = response2.json()
                        df = pd.DataFrame(data)
                        filename = f'Daten/Wetterdaten/Galicien/{indicator}/{year}_{months}.csv'
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        df.to_csv(filename, index=False)
                        print(f'Daten für {indicator} im Jahr {year} gespeichert in {filename}')
                    else:
                        print(f"Fehler beim Abrufen der Daten von {link}: Status Code {response2.status_code}")
                    
                else:
                    print(f"Fehler beim Abrufen der Daten für {indicator} im Jahr {year}: Status Code {response.json()['descripcion']} Try again in 30 seconds")
                    go_on = True
                    time.sleep(30)
                   

                    

