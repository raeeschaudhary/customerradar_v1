import dash
import dash_auth
from flask import request
#import dash_html_components as html
import dash_bootstrap_components as dbc
#import dash_core_components as dcc
import jupyter_dash as jd
from jupyter_dash import JupyterDash

from dash.dependencies import Output, Input, State
from dash import html, dcc, dash_table, ctx
import plotly.graph_objects as go
from urllib.parse import unquote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import plotly.express as px
import random

from datetime import datetime
from scipy import stats

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def load_csv_data(data_path, file_name):
    import os
    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path)



def make_clusters(dataset, K):
    CLV_M3 = pd.DataFrame()
    CLV_M3["recency"] = stats.boxcox(dataset['recency']+0.00001)[0]  #Box-Cox worked best for recency
    CLV_M3["frequency"] = stats.boxcox(dataset['frequency']+0.00001)[0] #Box-Cox worked best for frequency
    CLV_M3["monetary"] = pd.Series(np.cbrt(dataset['monetary'])).values #We are left with only this for monetary
    scaler = StandardScaler()
    scaler.fit(CLV_M3)
    CLV_M3_normalized = scaler.transform(CLV_M3)
    CLV_M3_normalized = pd.DataFrame(CLV_M3_normalized)
    
    model = KMeans(n_clusters=K, random_state=42)
    model.fit(CLV_M3_normalized)
    
   
    return model

CLV_M_1 = load_csv_data('', 'CLV_M1.csv')
CLV_M2 = CLV_M_1.copy()
model = make_clusters(CLV_M2, 5)
CLV_M2["Cluster"] = model.labels_

airline_data = load_csv_data('', 'airline_data.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server

#VALID_USERNAME_PASSWORD_PAIRS = {
#    'user1': 'pass1',
#    'user2': 'pass2'
#}

#auth = dash_auth.BasicAuth(
#    app,
#    VALID_USERNAME_PASSWORD_PAIRS
#)

def make_empty_fig():
    fig = go.Figure()
    fig.layout.paper_bgcolor = '#E5ECF6'
    fig.layout.plot_bgcolor = '#E5ECF6'
    return fig

def load_csv_data(data_path, file_name):
    import os
    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path)

incoming_orders = load_csv_data('', 'incoming_orders.csv')
CLV_incoming = load_csv_data('', 'clv_incoming.csv')
processed_data = load_csv_data('', 'processed_data.csv')

max_date = '20220930'
max_date = datetime.strptime(max_date, '%Y%m%d')
CLV_M_1 = load_csv_data('', 'CLV_M1.csv')

#CLV_M5.to_csv('CLV_M5.csv')

CLV_M5 = load_csv_data('', 'CLV_M5.csv')

def make_RFM_frame(dataset, max_date):
    dataset['createdon'] = pd.to_datetime(dataset['createdon'])
    #extract customers
    customers = dataset.customer.unique().tolist()
    #make data frame for RFM
    col_names =  ['customer', 'recency', 'frequency', 'monetary']
    rfm  = pd.DataFrame(columns = col_names)
    #reset index to run in loop
    rfm.reset_index(drop=True, inplace=True)
    #extract data for each customer 
    for i in range(0, len(customers)):
        each_customer = dataset[dataset['customer'].isin([customers[i]])]
        #reset index to run in loop
        each_customer.reset_index(drop=True, inplace=True)
        #combine orders created on the same date
        each_agg = each_customer.groupby(['createdon']).agg({'zssc1': lambda x: sum(x)}).reset_index()
        each_agg['customer'] = str(customers[i])
        
        #perform RFM on each customer
        customer, recency, frequency, monetary = calculate_rfm(each_agg, max_date)

        #add into datafame.
        rfm.loc[len(rfm)] = [customer, recency, frequency, monetary]

    return rfm

def remove_outliers(rfm_fig):
    #Take the quartile 5-95 (interval) for recency
    y1 = rfm_fig['recency']
    size = rfm_fig.shape[0]
    outliers = y1.between(y1.quantile(.01), y1.quantile(.99))
    #print(str(y1[outliers].size) + "/" + str(size) + " data points.") 
    index_names = rfm_fig[~outliers].index 
    rfm_fig.drop(index_names, inplace=True)

    #Take the quartile 5-95 (interval) for frequency
    y2 = rfm_fig['frequency']
    size = rfm_fig.shape[0]
    outliers = y2.between(y2.quantile(.01), y2.quantile(.99))
    #print(str(y2[outliers].size) + "/" + str(size) + " data points.") 
    index_names = rfm_fig[~outliers].index 
    rfm_fig.drop(index_names, inplace=True)

    #Take the quartile 5-95 (interval) for monetary
    y3 = rfm_fig['monetary']
    size = rfm_fig.shape[0]
    outliers = y3.between(y3.quantile(.01), y3.quantile(.99))
    #print(str(y3[outliers].size) + "/" + str(size) + " data points.") 
    index_names = rfm_fig[~outliers].index 
    rfm_fig.drop(index_names, inplace=True)
    return rfm_fig

def plot_rfm_fig(rfm_fig):
    rfm_fig['Cluster'] = rfm_fig['Cluster'].astype(str)
    #rfm_fig['customer'] = CLV_M_1['customer']
    #just taking the int value * 1000
    rfm_fig['monetary'] = (rfm_fig['monetary'] / 1000).astype(int)

    import plotly.express as px
    fig = px.scatter_3d(rfm_fig, x='recency', y='frequency', z='monetary', hover_name='customer',
                        labels={'monetary': 'monetary x 1000 (EURO)', 'recency': 'recency (days)', 'frequency': 'frequency (days)'},
                  color='Cluster', opacity=0.7,color_discrete_sequence=px.colors.qualitative.G10)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig


app.layout = html.Div([
        dbc.Row([
        html.H1('US Domestic Airline Flights Performance', style={'textAlign': 'center'}),
            dbc.Label('Report Type'),
            dcc.Dropdown(id='incoming_order_dropdown', optionHeight=40,
                         value='Select a report type',
                         options=[{'label': indicator, 'value': indicator}
                                  for indicator in ['Yearly airline performance report', 'Yearly average flight delay statistics']]),
            dbc.Label('Choose Year'),
            dcc.Dropdown(id='incoming_order_dropdown2', optionHeight=40,
                         value='',
                         options=[{'label': indicator, 'value': indicator}
                                  for indicator in range(2005, 2021)]),
    ]),
    dbc.Row([
        html.Div([
                dcc.Graph(id='graph_5',
                      figure=make_empty_fig()),
            ]),
    ]),
    dbc.Row([
        
        dbc.Col(lg=1),
        dbc.Col([
            html.Br(),
            html.Div([
                dcc.Graph(id='graph_3',
                      figure=make_empty_fig()),
            ]),
            html.Br(),
            html.Div([
                dcc.Graph(id='graph_4',
                      figure=make_empty_fig()),
            ]),
        ], md=12, lg=5),

        dbc.Col([
            html.Br(),
            html.Div([
                dcc.Graph(id='graph_1',
                      figure=make_empty_fig()),
            ]),
            html.Br(),
            html.Div([
                dcc.Graph(id='graph_2',
                      figure=make_empty_fig()),
            ]),
            
        ], md=12, lg=5),
    ]),
], style={'backgroundColor': '#E5ECF6'})


#This method plots the incoming orders
@app.callback(Output('graph_1', 'figure'), Output('graph_2', 'figure'), Output('graph_3', 'figure'), Output('graph_4', 'figure'), Output('graph_5', 'figure'),
             Input('incoming_order_dropdown', 'value'),
             Input('incoming_order_dropdown2', 'value'))
def plot_incoming_orders(rep, year):
    df =  airline_data.copy()
    if year:
        try:
            df = airline_data[airline_data['Year'] == int(year)]
        except:
            print('')
    if rep == 'Yearly airline performance report':
        # do all
        delay_type = df.groupby(['Reporting_Airline', 'CancellationCode'])['IATA_CODE_Reporting_Airline'].count().reset_index()
        avg_time = df.groupby(['Month', 'Reporting_Airline'])['CRSElapsedTime'].mean().reset_index() 
        diverted = df.groupby(['Month','Reporting_Airline'])['Diverted'].count().reset_index()
        avg_sec = df.groupby(['DestState', 'Reporting_Airline'])['IATA_CODE_Reporting_Airline'].count().reset_index()
        avg_late = df.groupby(['DestState', 'Reporting_Airline'])['IATA_CODE_Reporting_Airline'].count().reset_index()
                
        # Line plot for carrier delay
        delay_fig = px.bar(delay_type, x='IATA_CODE_Reporting_Airline', y='CancellationCode', color='IATA_CODE_Reporting_Airline', title='Number of flights under different cancellation categories')
        # Line plot for weather delay
        time_fig = px.line(avg_time, x='Month', y='CRSElapsedTime', color='Reporting_Airline', title='Average flight time by reporting airline')
        # Line plot for nas delay
        div_fig = px.pie(diverted, values='Diverted', names='Reporting_Airline', title='Percentage of diverted airport landings per reporting airline')
        # Line plot for security delay
        sec_fig = px.choropleth(
        locations=avg_sec['DestState'], color = avg_sec['Reporting_Airline'], locationmode="USA-states", scope="usa", 
                   color_continuous_scale = 'IATA_CODE_Reporting_Airline', 
                   labels={'IATA_CODE_Reporting_Airline':'Flights'},
        title = 'Number of flights flying from each state'
        )
        # Line plot for late aircraft delay
        late_fig = px.treemap(avg_late,
                    path=['DestState', 'Reporting_Airline'],
                    values='IATA_CODE_Reporting_Airline')
        late_fig.update_traces(root_color="lightgrey")
        late_fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

        return delay_fig, time_fig, div_fig, sec_fig, late_fig
    else:
        #do year wise
        avg_car = df.groupby(['Month','Reporting_Airline'])['CarrierDelay'].mean().reset_index()
        avg_weather = df.groupby(['Month','Reporting_Airline'])['WeatherDelay'].mean().reset_index()
        avg_NAS = df.groupby(['Month','Reporting_Airline'])['NASDelay'].mean().reset_index()
        avg_sec = df.groupby(['Month','Reporting_Airline'])['SecurityDelay'].mean().reset_index()
        avg_late = df.groupby(['Month','Reporting_Airline'])['LateAircraftDelay'].mean().reset_index()

        carrier_fig = px.line(avg_car, x='Month', y='CarrierDelay', color='Reporting_Airline', title='Average carrrier delay time (minutes) by airline')
        # Line plot for weather delay
        weather_fig = px.line(avg_weather, x='Month', y='WeatherDelay', color='Reporting_Airline', title='Average weather delay time (minutes) by airline')
        # Line plot for nas delay
        nas_fig = px.line(avg_NAS, x='Month', y='NASDelay', color='Reporting_Airline', title='Average NAS delay time (minutes) by airline')
        # Line plot for security delay
        sec_fig = px.line(avg_sec, x='Month', y='SecurityDelay', color='Reporting_Airline', title='Average security delay time (minutes) by airline')
        # Line plot for late aircraft delay
        late_fig = px.line(avg_late, x='Month', y='LateAircraftDelay', color='Reporting_Airline', title='Average late aircraft delay time (minutes) by airline')
            
        return carrier_fig, weather_fig, nas_fig, sec_fig, late_fig


if __name__ == '__main__':
    app.run_server(debug=True)
