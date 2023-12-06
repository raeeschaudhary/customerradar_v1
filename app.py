# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:34:54 2022

@author: Muhammad Raees
"""
import dash
import dash_auth
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input

from flask import request

import plotly.graph_objects as go
from urllib.parse import unquote

import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server

VALID_USERNAME_PASSWORD_PAIRS = {
    'user1': 'pass1',
    'user2': 'pass2'
}

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

def make_empty_fig():
    fig = go.Figure()
    fig.layout.paper_bgcolor = '#E5ECF6'
    fig.layout.plot_bgcolor = '#E5ECF6'
    return fig

def plot_rfm_fig(rfm_fig):
    rfm_fig['Cluster'] = rfm_fig['Cluster'].astype(str)
    #rfm_fig['customer'] = CLV_M_1['customer']
    #just taking the int value * 1000
    rfm_fig['monetary'] = (rfm_fig['monetary'] / 1000).astype(int)

    import plotly.express as px
    fig = px.scatter_3d(rfm_fig, x='recency', y='frequency', z='monetary', hover_name='customer',
                        labels={'monetary': 'Monetary x (K-EURO)', 'recency': 'Recency(days)', 'frequency': 'Frequency (counts)'},
                  color='Cluster', opacity=0.7,color_discrete_sequence=px.colors.qualitative.G10)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig

def load_csv_data(data_path, file_name):
    import os
    csv_path = os.path.join(data_path, file_name)
    return pd.read_csv(csv_path)

incoming_orders = load_csv_data('', 'incoming_orders.csv')
CLV_incoming = load_csv_data('', 'clv_incoming.csv')
processed_data = load_csv_data('', 'processed_data.csv')

CLV_M_1 = load_csv_data('', 'CLV_M1.csv')
CLV_M5 = load_csv_data('', 'CLV_M5.csv')


main_layout = html.Div([
    html.Div([
    dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Customers", href="customers")),
        dbc.NavItem(dbc.NavLink("Orders", href="Orders")),
    ],
    brand="Sappi - Customer Radar Prototype",
    brand_href="/",
    color="primary",
    dark=True,
    ),
    dbc.NavbarSimple([
        ]),
    dcc.Location(id='location'),
    html.Div(id='main_content'),
    html.Br(),
]),
    html.Br(),
],  style={'backgroundColor': '#E5ECF6'})

radar_dashboard = html.Div([
        dbc.Row([
        dbc.Col(lg=1),
        dbc.Col([
            html.H2('Incoming Order'),
            dcc.Dropdown(id='incoming_order_dropdown', optionHeight=40,
                         value='cust68',
                         options=[{'label': indicator, 'value': indicator}
                                  for indicator in incoming_orders['customer']]),
            html.Div(id='message'),
            html.H3('Calculations'),
            html.Div(id='message2'),
            dbc.Label('When trying to make a decision for the above order, how do you rate the estimation on a scale of 1 to 5: Where 1-Extremely Useless and 5-Extremely Useful'),
            dcc.Slider(id='estimation_slider',
                    min=1, max=5, step=1, included=False,
                    value=4, marks={n: str(n) for n in range(1, 6)}),            
            html.Br(),
            html.H3('Predictions'),
            html.H6('Radar predicts the probability of this incoming order becoming rejected is: Extremely Low'),
            dbc.Label('How do you rate feedback for this prediction on a scale of 1 to 5'),
            html.P('Where 1-Extremely Useless and 5-Extremely Useful'),
            dcc.Slider(id='estimation_slider2',
                    min=1, max=5, step=1, included=False,
                    value=4, marks={n: str(n) for n in range(1, 6)}),
            html.H3('Recency Comparison with Other Customers'),
            dcc.Graph(id='order_incoming_graph_r',
                      figure=make_empty_fig()),
            ], md=12, lg=5),
        dbc.Col([
            html.H3('Contribution Comparison of Current Open Orders'),
            dcc.Graph(id='order_incoming_graph',
                      figure=make_empty_fig()),
            html.Br(),
            html.H3('Comparison of Customer with Other Customers (having open orders)'),
            dcc.Graph(id='customer_graph',
                      figure=make_empty_fig()),
            dcc.Dropdown(id='change_cust',
                 value='1',
                 options=[{'label': year, 'value': str(year)}
                          for year in range(0, 1)]),
            ], md=12, lg=5),
        ]),
    dbc.Row([
        dbc.Col(lg=1),
        dbc.Col([
            html.H3('Frequency Comparison with Other Customers'),
            dcc.Graph(id='order_incoming_graph_f',
                      figure=make_empty_fig()),
            ], md=12, lg=5),
        dbc.Col([
            html.H3('Monetary Comparison with Other Customers'),
            dcc.Graph(id='order_incoming_graph_m',
                      figure=make_empty_fig()),
            ], md=12, lg=5),
            html.Div(id="saved1"),
    ]),
    
], style={'backgroundColor': '#E5ECF6'})


order_dashboard = html.Div([
    dbc.Row([
        dbc.Col(lg=1),
        dbc.Col([
            html.H1('Clustering of All Customers'),
            dbc.Label('Distribution Channel:'),
            dcc.Dropdown(id='multi_channel_selector',
                         multi=True,
                         placeholder='Select one or more channels',
                         options=[{'label': channel, 'value': channel}
                                  for channel in processed_data['distributionchannel'].drop_duplicates().sort_values()]),
            dbc.Label('Product Group:'),
            dcc.Dropdown(id='multi_product_selector',
                         multi=True,
                         placeholder='Select one or more product groups',
                         options=[{'label': product, 'value': product}
                                  for product in processed_data['productgroup'].drop_duplicates().sort_values()]),
            html.P('The visualization shows the clustering of customers based on the RFM values'),
            dcc.Graph(id='customer_graph1',
                      figure=make_empty_fig()),
            dbc.Label('Change number of clusters'),
            dcc.Slider(id='ncluster_slider',
            min=2, max=7, step=1, included=False,
            value=6, marks={n: str(n) for n in range(2, 8)}),
            dbc.Label('Rename clusters with user defined names, separate names by comma'),
            html.Br(),
            dcc.Input(id="cluster_names", type="text"),
            html.Br(),
            dbc.Label('When trying to make sales decisions this clustering is:'),
            dcc.Slider(id='clustering_slider',
                    min=1, max=5, step=1, included=False,
                    value=4,
                    marks={n: str(n) for n in range(1, 6)}),
            html.P('1: Extremely Useless, 2: Useless, 3: Neither, 4:Useful, 5: Extremely Useful'),
            html.Br(),
            ], md=12, lg=5),
        dbc.Col([
            html.H1('Cluster Insights'),
            html.Div(id="div_tab"),
            html.Br(),
            html.Div([
                dcc.Graph(id='cluster_insight_graph',
                      figure=make_empty_fig()),
            ]),
            html.Br(),
            html.Div(id="info_tab"),
            html.Br(),
            #dbc.Row([
                #dbc.Col([
                #    html.H5('Recency Weight'),
                #    html.Br(),
                #    dcc.Input(id="nrecency", type="number", min="1", max="100", value=33),
                #], md=12, lg=4),
                #dbc.Col([
                #    html.H5('Frequency Weight'),
                #    html.Br(),
                #    dcc.Input(id="nfrequency", type="number", min="1", max="100", value=33),
                #], md=12, lg=4),
                #dbc.Col([
                #    html.H5('Monetary Weight'),
                #    html.Br(),
                #    dcc.Input(id="nmonetary", type="number", min="1", max="100", value=33),
                #], md=12, lg=4),
            #]),
            #dbc.Row([
            #    dbc.Col([
            #        html.Button('Modify Weights', id='weight_rfm', n_clicks=0),
            #        html.Div(id='message'),
            #    ], md=12, lg=4),
            #]),
        ], md=12, lg=5),
    ]),
    dbc.Row([
        dbc.Col(lg=1),
        dbc.Col([
        html.Div(id="samp_tab"),
         html.Div(id="saved"),
         ], md=12, lg=5),
    ]),
])


app.validation_layout = html.Div([
    main_layout,
    radar_dashboard,
    order_dashboard,
])


app.layout = main_layout

    
#this method updates the layout to order and main 
@app.callback(Output('main_content', 'children'),
              Input('location', 'pathname'))
def display_content(pathname):
    if unquote(pathname[1:]) in ['customers']:
        return order_dashboard
    else:
        return radar_dashboard

#This loads and tracks the incoming order.
@app.callback(Output('message', 'children'),
              Output('message2', 'children'),
              Input('incoming_order_dropdown', 'value'))
def info_incoming_order(cust):
    #the user information is here
    #username = request.authorization['username']
    return_str = ''
    return_str2 = ''
    if str(cust) == 'Select an order to view':
        return_str += "Select an order"
        return_str2 += "Select an order"
    else:
        current_order = incoming_orders[incoming_orders['customer'] == str(cust)]
        current_order['createdon'] = current_order['createdon']
        return_str += "Customer  " + str(current_order['customer'].unique()[0]) + ' ordering ' + str(current_order['productgroup'].unique()[0]) 
        #return_str += '-' + str(current_order['brand'].unique()[0]) + '-' + str(current_order['grammage'].unique()[0]) + 'gm'
        return_str += ' with ' + str(current_order['zbweight'].unique()[0]) + ' weight and delivery date ' 
        return_str += str(current_order['billdate'].unique()[0]) + ' to ' + str(current_order['customercountry'].unique()[0]) 
        
        current_CLV = CLV_incoming[CLV_incoming['customer'] == str(cust)]
        return_str2 += "Radar calculates " + str(current_CLV['customer'].unique()[0]) + " CLV at: "  
        return_str2 += ' R = ' + str(current_CLV['recency'].unique()[0]) + ' Days, F = '
        return_str2 += str(current_CLV['frequency'].unique()[0]) + ' Times, M = ' + str(current_CLV['monetary'].unique()[0]) + ' Euro'

    return html.H6(return_str), html.H6(return_str2)

#This method plots the incoming orders
@app.callback(Output('order_incoming_graph', 'figure'),
             Input('incoming_order_dropdown', 'value'))
def plot_incoming_orders(order):
    orders = incoming_orders.copy()
    orders['Contribution'] = orders['zssc1']
    current_order = orders[orders['customer'] == str(order)]
    in_orders = orders[orders['customer'] != str(order)]
    in_orders = in_orders.sort_values(by='zssc1', ascending=False)
    order_frames = [current_order, in_orders]
    result = pd.concat(order_frames)
    fig = px.bar(result, x='customer', y='Contribution', text_auto='.3s',
             hover_data=['customer', 'Contribution', 'billdate'], labels={'customer': 'customer-order', 
             'Contribution': 'Contribution (Euro)', 'billdate': 'Billed Date'}, height=400)
    fig.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="green")
    return fig

#This method plots the clusters
@app.callback(Output('customer_graph', 'figure'),
              Output('order_incoming_graph_r', 'figure'),
              Output('order_incoming_graph_f', 'figure'),
              Output('order_incoming_graph_m', 'figure'),
             Input('incoming_order_dropdown', 'value'))
def plot_incoming_customer_by_rfm(order):
    CLV_incoming = load_csv_data('', 'clv_incoming.csv')
    rfm_fig = CLV_incoming.copy()
    rfm_fig['monetary'] = (rfm_fig['monetary'] / 1000).astype(int)
    current_order = rfm_fig[rfm_fig['customer'] == str(order)]
    current_order['Order'] = 'Selected'
    in_orders = rfm_fig[rfm_fig['customer'] != str(order)]
    in_orders['Order'] = 'Others'
    order_frames = [current_order, in_orders]
    result = pd.concat(order_frames)
    import plotly.express as px
    fig = px.scatter_3d(result, x='recency', y='frequency', z='monetary', hover_name='customer',
                        labels={'monetary': 'Monetary x K(EURO)', 'recency': 'Recency (days)', 'frequency': 'Frequency (Count)'},
                  color='Order', opacity=0.7, color_discrete_sequence=px.colors.qualitative.G10)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    
    fig2 = px.bar(result, x='customer', y='recency', text_auto='.2s',
             hover_data=['customer', 'recency'], labels={'customer': 'customer-order', 'recency': 'Recency (Days)'},height=400)
    fig2.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="green")
    fig3 = px.bar(result, x='customer', y='frequency', labels={'customer': 'customer-order', 'frequency': 'Frequency (Count)'}, 
                text_auto='.2s', hover_data=['customer', 'frequency'], height=400)
    fig3.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="green")
    fig4 = px.bar(result, x='customer', y='monetary', labels={'customer': 'customer-order', 
            'monetary': 'Monetary (K-Euro)'}, text_auto=True,
             hover_data=['customer', 'monetary'], height=400)
    fig4.add_vline(x=0.5, line_width=3, line_dash="dash", line_color="green")
    
    return fig, fig2, fig3, fig4

    
#This method re-plots the clusters with input from user
@app.callback(Output('customer_graph1', 'figure'),
              Output('div_tab', 'children'),
              Output('info_tab', 'children'),
              Output('samp_tab', 'children'),
              Output('cluster_insight_graph', 'figure'),
              Input('cluster_names', 'value'),
              Input('ncluster_slider', 'value'),
              #Input('multi_channel_selector', 'value'),
             #Input('multi_product_selector', 'value'),
             )
def display_weighted(cluster_names, n_clusters):#, channels, products):
    filtered = processed_data.copy()
    #if channels:
    #    filtered = processed_data[processed_data['distributionchannel'].isin(channels)]
    #if products:
    #    filtered = processed_data[processed_data['productgroup'].isin(products)]
    
    if n_clusters == 2:
        CLV_M5 = load_csv_data('', 'CLV_M2.csv')
    elif n_clusters == 3:
        CLV_M5 = load_csv_data('', 'CLV_M3.csv')
    elif n_clusters == 4:
        CLV_M5 = load_csv_data('', 'CLV_M4.csv')
    elif n_clusters == 5:
        CLV_M5 = load_csv_data('', 'CLV_M5.csv')
    elif n_clusters == 6:
        CLV_M5 = load_csv_data('', 'CLV_M6.csv')
    elif n_clusters == 7:
        CLV_M5 = load_csv_data('', 'CLV_M7.csv')
    else: 
        CLV_M5 = load_csv_data('', 'CLV_M5.csv')

    rfm_fig = CLV_M5.copy()
    
    if cluster_names is not None:
        names = cluster_names.split(",")
        for i in range(0, len(names)):
            rfm_fig['Cluster'] = rfm_fig['Cluster'].astype(str).str.replace(str(i),names[i])
    fig = plot_rfm_fig(rfm_fig)
    col_names =  ['Fiscal Period', 'Bill Period', 'Total Customers', 'Total Orders' ]
    data_insight  = pd.DataFrame(columns = col_names)
    data_insight.loc[len(data_insight)] = [('2022010-2022012'), '2022-07-04-2022-09-30', len(processed_data['customer'].unique()),
                                            len(processed_data['customer'])]
    
    col_names =  ['Customer Cluster', 'Customers', 'Recency(Days-Avg)', 'Frequency(Count-Avg)', 'Monetary(KxEURO-Avg)']
    clust_insight  = pd.DataFrame(columns = col_names)
    clusters = rfm_fig.Cluster.unique().tolist()
    
    for i in range(0, len(clusters)):
        each_cluster = rfm_fig[rfm_fig['Cluster'].isin([clusters[i]])]
        each_cluster.reset_index(drop=True, inplace=True)
        cluster = each_cluster['Cluster'][0] #first value taken as customer ID
            # calculate recency
        TotalCustomers = len(each_cluster['customer'])
        recency = round(sum(each_cluster['recency'])/len(each_cluster['recency'])) 
        frequency = round(sum(each_cluster['frequency'])/len(each_cluster['frequency']))
        monetary = int(sum(each_cluster['monetary'])/len(each_cluster['monetary']))
        clust_insight.loc[len(clust_insight)] = [cluster, TotalCustomers, recency, frequency, monetary]
    fig2 = px.bar(clust_insight.sort_values(by=['Customers']), x='Customer Cluster', y='Customers', text_auto='.3s',
             hover_data=['Customers', 'Customer Cluster'], labels={'Customers': 'Total Customers', 'Customer Cluster': 'Customer Cluster'},height=400)

    col_names =  ['Sample Customers']
    samp_insight  = pd.DataFrame(columns = col_names)
    for i in range(0, len(clusters)):
        each_cluster = rfm_fig[rfm_fig['Cluster'].isin([clusters[i]])]
        samp = each_cluster.sample(1)
        samp.reset_index(drop=True, inplace=True)
        samp_example = str(samp.at[0, 'Cluster']) + ' customer:' + str(samp.at[0, 'customer']) + ', whose latest order was ' + str(samp.at[0, 'recency']) + ' days ago with a total of ' + str(samp.at[0, 'frequency']) + ' transactions and providing contribution of ' + str(samp.at[0, 'monetary']) + 'K Euros'
        samp_insight.loc[len(samp_insight)] = [samp_example]

    return fig, dash_table.DataTable(clust_insight.to_dict('records'), [{"name": i, "id": i} for i in clust_insight.columns], 
                                     id='tbl', editable=True), dash_table.DataTable(data_insight.to_dict('records'), [{"name": i, "id": i} for i in data_insight.columns]), dash_table.DataTable(
                                         samp_insight.to_dict('records'), [{"name": i, "id": i} for i in samp_insight.columns], style_cell={'textAlign': 'left'},), fig2
     
#This method Stores a CSV File
@app.callback(Output('saved1', 'children'),
              Input('incoming_order_dropdown', 'value'),
              Input('estimation_slider', 'value'),
              Input('estimation_slider2', 'value'))
def save_interaction(ord_dd, est_sld, rej_est):
    import csv   
    fields=['user', str(ord_dd), str(est_sld),str(rej_est)]
    with open(r'clv_est.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    return html.P('Saved')

#This method Stores a CSV File
@app.callback(Output('saved', 'children'),
              Input('ncluster_slider', 'value'),
              Input('clustering_slider', 'value'))
def save_interaction2(n_clusts, est_sld):
    import csv   
    fields=['user', str(n_clusts), str(est_sld)]
    with open(r'clust_est.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    return html.P('Saved')                

#@app.callback(
#    Output('tbl', 'data'),
#    Output('cluster_names', 'value'),
#    Input('tbl', 'data_timestamp'),
#    State('tbl', 'data'))
#def update_columns(timestamp, rows):
#    cluster_names = ''
#    for row in rows:
#        cluster_names += row['Cluster'] + ',' 
#    print(cluster_names)
#    return rows, cluster_names

#@app.callback(Output('message', 'children'),
#              Input('nrecency', 'value'),
#              Input('nfrequency', 'value'),
#              Input('nmonetary', 'value'))
#def display_weighted(rec, fre, mon):
#    if rec + fre + mon >= 100:
#        return html.H5('The sum of all three percentages cannot be greater than 100')
#    return html.H3("")

if __name__ == '__main__':
    app.run_server(debug=True)
