# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:18:06 2022

@author: ArdiMirzaei
"""
import json
import pandas as pd 
import requests


#%%
headers = {"X-Api-key": "_BLOCKED_"}

cities = ['Sydney', 'Melbourne', 'Brisbane', 'Adelaide', 'Canberra']

# id_url = 'https://api.domain.com.au/v1/salesResults/Sydney/' 
# sales_data = requests.get(id_url, headers = headers)
# sales_data = json.loads(sales_data.text)
# sales_data = pd.DataFrame.from_dict(sales_data.items()).T
# sales_data.columns = sales_data.loc[0]
# sales_data = sales_data.loc[1]
# sales_data.to_csv( 'Sales_Data_'+str((now.strftime("%Y_%m_%d"))+'.csv'))

#%%
for city in cities:
    id_url = f'https://api.domain.com.au/v1/salesResults/{city}/' 
    sales_data = requests.get(id_url, headers = headers)
    sales_data = json.loads(sales_data.text)
    sales_data = pd.DataFrame.from_dict(sales_data.items()).T
    sales_data.columns = sales_data.loc[0]
    sales_data = sales_data.loc[1]
    sales_data['City'] = city
    results = pd.DataFrame(sales_data.values, index = sales_data.keys()).T
    old_data = pd.read_csv(f'../data/Domain_Sales_Data_{city}.csv')
    results = pd.concat((old_data, results))
    results.to_csv(f'../data/Domain_Sales_Data_{city}.csv', index = False)
