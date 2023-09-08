import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import dash
import plotly.express as px
from dash import Dash, dcc, html
import pandas as pd
import dash_table
from dash.dependencies import Input, Output ,State
from pandas.api.types import CategoricalDtype
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

df = pd.read_csv('cred_4.csv', low_memory=False)

df1 = df.groupby(['operator', 'query_id'], as_index = False).agg('size')
df1.cross = pd.crosstab(index=df['operator'], columns=df1['size'])
freqcount=df1.cross.sort_values(by = df1.cross.columns.tolist(), ascending = False)
freqcount=freqcount.reset_index()

tabscan_df = df[df['operator'] == 'TableScanOperator']
cols = ['query_id',

        'thread_duration_max', 'cost_percent_str_Values', 'parquet_task_cost_percent_Values',
        'parquet_reading_cost_percent_Values', 'read_io_time_percent_Values', 'filtering_cost_percent_Values',
        'open_time_percent_Values',

        'files', 'partitions', 'tasks', 'total_row_groups', 'parallelism', 'skipped_row_groups', 'row_count_in',
        'row_count_out', 'num_chunks', 'task_rowsInCount_0_50_75_90_100', 'RowsInPerThread_max',
        'InputRowsPerThread_0_50_75_90_100', 'read_io_count', 'read_io_bytes', 'seek_io_count'
         ]
tabscan_df = tabscan_df[cols]
# tabscan_df.info()

tabscan_df = tabscan_df.dropna(axis = 1, how = 'all')
col_ind = np.where(tabscan_df.isnull().sum() > 1000)[0]
print(col_ind)
tabscan_df = tabscan_df.drop(tabscan_df.columns[col_ind],axis = 1)
tabscan_df = tabscan_df.dropna()
print(len(tabscan_df))
tabscan_df = tabscan_df.reset_index(drop = True)
anom_files = np.where((tabscan_df['files'] < 1) | (tabscan_df['tasks'] < 1) |
                      (tabscan_df['row_count_in'] < 1))[0].tolist()
# print(len(anom_files))
tabscan_df.loc[anom_files, 'anomaly_type'] = 'files/tasks/rows = 0'
print(tabscan_df['anomaly_type'].value_counts())
# tabscan_df.iloc[anom_files]

anom_tasksVtrg = np.where(tabscan_df['tasks'] > tabscan_df['total_row_groups'])[0].tolist()
tabscan_df.loc[anom_tasksVtrg, 'anomaly_type'] = 'tasks > total_row_groups'
anom_tasksVpll = np.where(tabscan_df['tasks'] > tabscan_df['parallelism'])[0].tolist()
print(len(anom_tasksVpll))

tabscan_df['anomaly_type'].value_counts()

anom_tasksVpll = np.where(tabscan_df['tasks'] > tabscan_df['parallelism'])[0].tolist()
print(len(anom_tasksVpll))
anom_tasksVpll = [i for i in anom_tasksVpll if i not in anom_tasksVtrg]
print(len(anom_tasksVpll))
tabscan_df.loc[anom_tasksVpll, 'anomaly_type'] = 'tasks > parallelism'
tabscan_df['anomaly_type'].value_counts()

join_df = df[df['operator'] == 'JoinOperator']

if 'Join_build_row_count_in' not in join_df:
    join_df['Join_build_row_count_in']=''


cols = ['query_id', 'thread_duration_max',
       'row_count_in', 'Join_build_row_count_in', 'row_count_out', 'RowsInPerThread_max',
        'InputRowsPerThread_0_50_75_90_100', 'num_chunks', 'join_type']
join_df = join_df[cols]
join_df['join_type'] = join_df['join_type'].astype('category')
join_df['join_type'] = join_df['join_type'].cat.codes
join_df = join_df.dropna(axis = 1, how = 'all')
col_ind = np.where(join_df.isnull().sum() > 1000)[0]
join_df = join_df.drop(join_df.columns[col_ind],axis = 1)
join_df = join_df.dropna()
join_df = join_df.reset_index(drop = True)
join_df['anomaly_type'] = None

anom_files = np.where(( (join_df['thread_duration_max'] == 0) & (join_df['row_count_in'] > 0) &
             (join_df['Join_build_row_count_in'] > 0) ))[0].tolist()
print(len(anom_files))
join_df.loc[anom_files, 'anomaly_type'] = 'duration = 0 & probe,build rows > 0'
print(join_df['anomaly_type'].value_counts())

anom_files = np.where((join_df['row_count_in'] < join_df['Join_build_row_count_in']))[0].tolist()
print(len(anom_files))
join_df.loc[anom_files, 'anomaly_type'] = 'Probe rows < Build rows'
print(join_df['anomaly_type'].value_counts())

value_counts_df = pd.DataFrame(df['operator'].value_counts().reset_index())
value_counts_df.columns = ['Operator', 'Count']

# Filter the DataFrame to get data for "Sink Operator" entries
sink_operator_data = df[df['operator'] == 'SinkOperator']
sink_operator_data['Hour'] = sink_operator_data['Hour'].astype('category')
sink_operator_data['Month'] = sink_operator_data['Month'].astype('category')
sink_operator_data['Date'] = sink_operator_data['Date'].astype('category')


# Create a bar plot for the count of "Hour" occurrences using Plotly Express
# Create a bar plot for the count of "Hour" occurrences using Plotly Express
hourly_data = sink_operator_data.groupby('Hour').size().reset_index(name='count')
hourly_labels = [
    "12 AM", "1 AM", "2 AM", "3 AM", "4 AM", "5 AM", "6 AM", "7 AM", "8 AM", "9 AM", "10 AM", "11 AM",
    "12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM"
]
fig_hour = px.bar(
    hourly_data,
    x='Hour',  # X-axis: Capitalized "Hour"
    y='count',  # Y-axis: Count of occurrences
    title='Hourly Data for Sink Operator',  # Chart title
    labels={'Hour': 'Hour', 'count': 'Count'},  # Label customization
)
fig_hour.update_layout(
    xaxis_title='Hour',
    yaxis_title='Count',
    xaxis=dict(
        showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2,
        tickvals=list(range(24)), ticktext=hourly_labels,
        range=[0, 23]  # Set the X-axis range from 0 to 23
    ),
    yaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2),
    plot_bgcolor='#F2F2F2',  # Plot background color
    paper_bgcolor='white',   # Paper background color
    font=dict(family='Arial', size=14, color='black'),
)

# Create a bar plot for the count of "Month" occurrences using Plotly Express
# Create a bar plot for the count of "Month" occurrences using Plotly Express
monthly_data = sink_operator_data.groupby('Month').size().reset_index(name='count')
monthly_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
fig_month = px.bar(
    monthly_data,
    x='Month',  # X-axis: Capitalized "Month"
    y='count',  # Y-axis: Count of occurrences
    title='Monthly Data for Sink Operator',  # Chart title
    labels={'Month': 'Month', 'count': 'Count'},  # Label customization
)
fig_month.update_layout(
    xaxis_title='Month',
    yaxis_title='Count',
    xaxis=dict(
        showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2,
        tickvals=list(range(1, 13)), ticktext=monthly_labels,  # Adjusted tickvals and ticktext
        range=[1, 12]  # Set the X-axis range from 1 to 12
    ),
    yaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2),
    plot_bgcolor='#F2F2F2',  # Plot background color
    paper_bgcolor='white',   # Paper background color
    font=dict(family='Arial', size=14, color='black'),
)

daily_data = sink_operator_data.groupby('Date').size().reset_index(name='count')
daily_labels = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30","31",
]
fig_date = px.bar(
    daily_data,
    x='Date',  # X-axis: Capitalized "Month"
    y='count',  # Y-axis: Count of occurrences
    title='Day-wise Data for Sink Operator',  # Chart title
    labels={'Date': 'Date', 'count': 'Count'},  # Label customization
)
fig_date.update_layout(
    xaxis_title='Date',
    yaxis_title='Count',
    xaxis=dict(
        showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2,
        tickvals=list(range(1, 32)), ticktext=daily_labels,  # Adjusted tickvals and ticktext
        range=[0, 32]  # Set the X-axis range from 1 to 12
    ),
    yaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='black', linewidth=2),
    plot_bgcolor='#F2F2F2',  # Plot background color
    paper_bgcolor='white',   # Paper background color
    font=dict(family='Arial', size=14, color='black'),
)




queryleveldf=df[df["operator"]=="SinkOperator"]

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
# Define inline styles for the tabs, card, and dashboard
tab_style = {
    'color': '#FFFFFF',            # Text color for tabs
    'backgroundColor': '#007BFF',  # Background color for tabs
}
card_style = {
    'backgroundColor': '#F5F5F5',  # Background color for card
}
dashboard_style = {
    'backgroundColor': '#ECECEC',  # Background color for the entire dashboard
}


# Define the layout for the Operators window
operators_layout = html.Div([
    html.H2('Operators', style={'text-align': 'center'}),
    dcc.RadioItems(
        id='operator-type',
        options=[
            {'label': 'Table Scan', 'value': 'tablescan'},
            {'label': 'Join', 'value': 'join'}
        ],
        value='tablescan',  # Default value
        style={'text-align': 'center', 'margin-bottom': '20px'}
    ),
    dcc.Tabs(id='operator-tabs', value='A', children=[
        dcc.Tab(label='Independent/Input metrics:', value='A'),
        dcc.Tab(label='Dependent/Output metrics:', value='B'),
        dcc.Tab(label='Correlation', value='C'),
        dcc.Tab(label='Anomaly Types - Rule based', value='D'),
        dcc.Tab(label='Anomaly Types - Model based', value='E'),
    ]),
    html.Div(id='operator-content')
])

# Define the layout for the new window with tabs
temporal_layout = html.Div([
    html.H2('Temporal Data of Queries', style={'text-align': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Hourly Data', children=[
            dcc.Graph(figure=fig_hour),  # Add the "Hour" bar plot to the "Hourly Data" tab
        ]),
        dcc.Tab(label='Monthly Data', children=[
            dcc.Graph(figure=fig_month),  # Add the "Month" bar plot to the "Monthly Data" tab
        ]),
        dcc.Tab(label='Day-wise Data', children=[
            dcc.Graph(figure=fig_date),  # Add the "Hour" bar plot to the "Hourly Data" tab
        ]),
    ]),
])

analysis_layout = html.Div(
    children=[
        dbc.Card(
            children=[
                dbc.CardHeader(
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                label="Query Time Analysis",
                                tab_id="tab-1",
                                style=tab_style,
                            ),
                            dbc.Tab(
                                label="Distributions",
                                tab_id="tab-2",
                                style=tab_style,
                            ),
                        ],
                        id="tabs",
                        active_tab="tab-1",
                    )
                ),
                dbc.CardBody(html.Div(id="card-content", style={'color': 'black'})),
            ],
            style=card_style,
        ),
    ],
    style=dashboard_style,
)


Query_layout = html.Div([
    html.H4(f'No of Queries: {len(df["query_id"].unique())}', style={'text-align': 'centre'}),
    dcc.Tabs([
        dcc.Tab(label='Query Analysis', children=[analysis_layout
            # Content for Query Analysis tab
           ]),
        dcc.Tab(label='Query Searcher', children=[
            html.Div([
                dcc.Input(id='search-query-id', type='text', placeholder='Enter query_id'),
                html.Button('Search', id='search-button'),
                html.Div(id='searched-query-tabs',style={'overflowY': 'scroll','overflowX': 'scroll'})
            ])
            # Content for Query Searcher tab
        ]),
    ]),
])
@app.callback(
    Output('searched-query-tabs', 'children'),  # Output to update the tabs
    [Input('search-button', 'n_clicks')],
    [State('search-query-id', 'value')]
)
def update_searched_query_tabs(n_clicks, query_id):
    if n_clicks is None:
        return dash.no_update
    if query_id is None:
        return "Please enter a query_id and click 'Search'."
    # Filter the data based on the entered query_id
    queried_data = df[df['query_id'] == query_id]
    if queried_data.empty:
        return "No data found for the entered query_id."
    # Create and display the table for the queried data
    searched_table = dash_table.DataTable(
        id='searched-query-table-output',
        columns=[{'name': col, 'id': col} for col in queried_data.columns],
        data=queried_data.to_dict('records'),
        style_table={
            'textAlign': 'left',
            'margin': 'auto',
            'width': '100%',
            'border': '1px solid #ddd',
            'borderCollapse': 'collapse',
        },
        style_header={
            'backgroundColor': '#007BFF',
            'color': 'white',
            'fontWeight': 'bold',
        },
        style_cell={
            'textAlign': 'left',
            'border': '1px solid #ddd',
            'padding': '10px',
            'backgroundColor': '#F2F2F2',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#F9F9F9',
            },
        ],
    )
    # Create a div to display the value counts of the "operator" column
    operator_value_counts_div = html.Div([
        html.H4("Value Counts of 'operator' Column:"),
        dcc.Graph(
            id='operator-value-counts',
            figure={
                'data': [
                    {
                        'x': queried_data['operator'].value_counts().index,
                        'y': queried_data['operator'].value_counts().values,
                        'type': 'bar',
                    },
                ],
                'layout': {
                    'xaxis': {'title': 'Operator'},
                    'yaxis': {'title': 'Count'},
                    'title': 'Operator Value Counts',
                },
            },
        ),
    ])
    # Create two dropdowns for selecting graph type and metrics
    graph_type_dropdown = dcc.Dropdown(
        id='graph-type-dropdown',
        options=[
            {'label': 'Operator Value Counts', 'value': 'value_counts'},
            {'label': 'Stacked Bar Graph', 'value': 'stacked_bar'},
        ],
        value='value_counts',  # Default to showing value counts graph
        style={'margin-top': '20px'}
    )
    metrics_dropdown = dcc.Dropdown(
        id='metrics-dropdown',
        options=[
            {'label': 'Parsing Time', 'value': 'parsing_time'},
            {'label': 'Execution Time', 'value': 'execution_time'},
            {'label': 'Execution Queueing Time', 'value': 'executionQueueingTime'},
            {'label': 'Parsing Time', 'value': 'parsingTime'},
            {'label': 'Total Open Duration', 'value': 'totalOpenDuration'},
            {'label': 'Total Client Query Time', 'value': 'totalClientQueryTime'},
        ],
        multi=True,  # Allow multiple metric selection
        value=['parsing_time', 'execution_time'],  # Default metrics to show
    )
    # Create a div to display the stacked bar graph
    stacked_bar_graph_div = html.Div(id='stacked-bar-graph-div')
    # Create two tabs with the table in the first tab and the Analysis div in the second tab
    searched_tabs = dcc.Tabs([
        dcc.Tab(label='Table', children=[
            searched_table
        ]),
        dcc.Tab(label='Analysis', children=[
            graph_type_dropdown,
            metrics_dropdown,
            operator_value_counts_div,
            stacked_bar_graph_div
        ]),
    ])
    return searched_tabs
import plotly.express as px
# ... (Previous code remains unchanged)
@app.callback(
    Output('stacked-bar-graph-div', 'children'),
    [Input('graph-type-dropdown', 'value'),
     Input('metrics-dropdown', 'value')],
    [State('searched-query-table-output', 'data')]
)
def update_stacked_bar_graph(graph_type, selected_metrics, queried_data):
    if graph_type == 'stacked_bar':
        if queried_data:
            queried_data = pd.DataFrame(queried_data)  # Convert queried_data to DataFrame
            # Filter the data for the SinkOperator
            sink_operator_data = queried_data[queried_data['operator'] == 'SinkOperator']
            if not sink_operator_data.empty:
                # Create a stacked bar graph for the selected metrics
                fig = go.Figure()
                for metric in selected_metrics:
                    if metric != 'count':
                        fig.add_trace(go.Bar(
                            x=sink_operator_data['query_id'],
                            y=sink_operator_data[metric],
                            name=metric,
                            hovertext=f"{metric}: " + sink_operator_data[metric].astype(str),
                        ))
                if 'totalClientQueryTime' in selected_metrics:
                    fig.add_trace(go.Bar(
                        x=sink_operator_data['query_id'],
                        y=sink_operator_data['totalClientQueryTime'],
                        name='Total Client Query Time',
                        hovertext=f"Total Client Query Time: " + sink_operator_data['totalClientQueryTime'].astype(str),
                    ))
                fig.update_layout(
                    barmode='stack',  # Stack the bars
                    xaxis_title='Query ID',
                    yaxis_title='Time',
                    title='Stacked Bar Graph of Sink Operator Metrics',
                )
                return dcc.Graph(figure=fig)
            else:
                return "No SinkOperator data to display."
    return None
@app.callback(
    Output('value-counts-graph-div', 'children'),  # Output to update the value counts graph
    [Input('graph-type-dropdown', 'value')],
    [State('searched-query-table-output', 'data')]
)
def update_value_counts_graph(graph_type, queried_data):
    if graph_type == 'value_counts' and queried_data:
        queried_data = pd.DataFrame(queried_data)  # Convert queried_data to DataFrame
        # Create a value counts graph for the "operator" column
        value_counts_fig = px.bar(queried_data['operator'].value_counts(), x='index', y='operator')
        value_counts_fig.update_layout(
            xaxis_title='Operator',
            yaxis_title='Count',
            title='Operator Value Counts',
        )
        return dcc.Graph(figure=value_counts_fig)
    return None
@app.callback(
    Output('second-analysis-div', 'children'),
    [Input('first-analysis-dropdown', 'value')]
)
def update_second_analysis_dropdown(selected_value):
    if selected_value == 'operator_value_counts':
        # Create an empty div for the second analysis dropdown when operator value counts is selected
        return html.Div()
    # If you have additional analysis options, add the corresponding dropdowns and logic here


# Define the callback to update the tab content
@app.callback(Output("card-content", "children"), [Input("tabs", "active_tab")])
def render_content(active_tab):
    if active_tab == "tab-1":
        return html.Div([
            html.H3("Time Graphs"),
            # Add the dropdown and graph to Tab 1
            dcc.Dropdown(
                id='graph-dropdown',
                options=[
                    {'label': 'Execution time', 'value': 'graph1'},
                    {'label':'Total Client Query Time','value':'graph2'},
                    {'label':'Comaprison','value':'graph3'}
                ],
                value='graph1'  # Initial selection
            ),
            dcc.Graph(id='display-graph'),
        ])
    elif active_tab == "tab-2":
        return html.Div([
            html.H3("Distributions"),
            dcc.Dropdown(
                id='graph-dropdown',
                options=[
                    {'label': 'Row Count in', 'value': 'graph4'},
                    {'label':'Row Count out','value':'graph5'},
                    {'label':'Total Bytes','value':'graph6'}
                ],
                value='graph4'  # Initial selection
                ),
            dcc.Graph(id='display-graph'),
        ])
# Callback to update the graph based on the dropdown selection
@app.callback(
    Output('display-graph', 'figure'),
    [Input('graph-dropdown', 'value')]
)
def update_graph(selected_graph):
    if selected_graph == 'graph1':
        fig = px.line(queryleveldf, x='query_id', y='execution_time',title='Execution Time vs Query ID')
        fig.update_yaxes(range=[0, 70000])  # Adjust the range as needed
        fig.update_xaxes(title_text='', showticklabels=False)
        return fig
    elif selected_graph == 'graph2':
        fig=px.line(queryleveldf,x='query_id',y='totalClientQueryTime',title='Total Client Query Time vs Query ID')
        fig.update_xaxes(title_text='', showticklabels=False)
        fig.update_traces(line=dict(color="green"))
        fig.update_yaxes(range=[0, 100000])
        return fig
    elif selected_graph == 'graph3':
        fig = px.line(queryleveldf, x='query_id', y='execution_time')
        fig2=px.line(queryleveldf,x='query_id',y='totalClientQueryTime')
        fig.update_traces(line=dict(color="blue"))
        fig2.update_traces(line=dict(color="green"))
        fig.add_trace(fig2.data[0])
        fig.update_layout(
        xaxis_title='Query ID',
        yaxis_title='Time',
        legend_title_text='Data Sets',
        hovermode='x',
        hoverlabel=dict(bgcolor="white", font_size=14),
)
        fig.update_xaxes(title_text='', showticklabels=False)
        fig.update_yaxes(range=[0, 200000])  # Adjust the range as needed
        return fig  # Return an empty figure if no selection
    elif selected_graph=='graph4':
        fig=px.histogram(queryleveldf,x='row_count_in',nbins=1000)
        fig.update_xaxes(range=[0,200000])
        return fig
    elif selected_graph=='graph5':
        fig=px.histogram(queryleveldf,x='row_count_out',nbins=1000)
        fig.update_xaxes(range=[0,200000])
        return fig
    elif selected_graph=='graph6':
        fig=px.histogram(queryleveldf,x='totalBytes',nbins=10000)
        fig.update_xaxes(range=[0,500000000])
        return fig

app.layout = html.Div([
    html.H1(f'cred data', style={'text-align': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Query Data', children=[Query_layout]),
        dcc.Tab(label='Temporal Time', children=[temporal_layout]),  # Add the new window as a tab
        dcc.Tab(label='Operator Stats', children=[
            dcc.RadioItems(
                id='frequency-type',  # ID for radio items
                options=[
                    {'label': 'Operator Count', 'value': 'count'},
                    {'label': 'Operator Frequency', 'value': 'frequency'}
                ],
                value='count',  # Default value
                style={'text-align': 'center', 'margin-bottom': '20px'}
            ),
            html.Div(id='frequency-content')  # Content will be populated by the callback
        ]),
        dcc.Tab(label='Operators', children=[operators_layout]),  # Add the Operators window as a tab
    ])
])


def create_variable_table(variables):
    return dash_table.DataTable(
        id='variable-table',
        columns=[
            {'name': 'Variable', 'id': 'Variable'}
        ],
        data=[
            {'Variable': var} for var in variables
        ],
        style_table={
            'textAlign': 'left',
            'margin': 'auto',
            'width': '50%',
            'border': '1px solid #ddd',
            'borderCollapse': 'collapse',
        },
        style_header={
            'backgroundColor': '#007BFF',
            'color': 'white',
            'fontWeight': 'bold',
        },
        style_cell={
            'textAlign': 'left',
            'border': '1px solid #ddd',
            'padding': '10px',
            'backgroundColor': '#F2F2F2',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#F9F9F9',
            },
        ],
    )

# Define callback to update operator content based on selected radio item and tab
@app.callback(
    Output('frequency-content', 'children'),
    [Input('frequency-type', 'value')]
)
def update_frequency_content(selected_type):
    if selected_type == 'frequency':

        # Return the operator frequency table
        return dash_table.DataTable(
            id='freqcount-table',
            columns=[{'name': str(col), 'id': str(col)} for col in freqcount.columns],
            data=freqcount.to_dict('records'),
            style_table={
                'textAlign': 'left',
                'margin': 'centre',
                'width': '50%',
                'border': '1px solid #ddd',
                'borderCollapse': 'collapse',
            },
            style_header={
                'backgroundColor': '#007BFF',
                'color': 'white',
                'fontWeight': 'bold',
            },
            style_cell={
                'textAlign': 'left',
                'border': '1px solid #ddd',
                'padding': '10px',
                'backgroundColor': '#F2F2F2',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#F9F9F9',
                },
            ],
        )
    elif selected_type == 'count':
        # Return the operator count table
        return dash_table.DataTable(
            id='count-table',
            columns=[{'name': str(col), 'id': str(col)} for col in value_counts_df.columns],
            data=value_counts_df.to_dict('records'),
            style_table={
                'textAlign': 'left',
                'margin': 'auto',
                'width': '50%',
                'border': '1px solid #ddd',
                'borderCollapse': 'collapse',
            },
            style_header={
                'backgroundColor': '#007BFF',
                'color': 'white',
                'fontWeight': 'bold',
            },
            style_cell={
                'textAlign': 'left',
                'border': '1px solid #ddd',
                'padding': '10px',
                'backgroundColor': '#F2F2F2',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#F9F9F9',
                },
            ],
        )
    else:
        return None

@app.callback(
    Output('operator-content', 'children'),
    [Input('operator-type', 'value'),
     Input('operator-tabs', 'value')]
)
def update_operator_content(operator_type, tab_selected):
    operator_data = join_df

    if operator_type == 'tablescan':
        # Define your table data for 'tablescan' here
        operator_data = pd.DataFrame(tabscan_df)  # Define your data
    elif operator_type =='joinscan':
        # Define your table data for 'join' here
        operator_data = pd.DataFrame(join_df)  # Define your data

    if tab_selected == 'A':
        if operator_type == 'tablescan':
            operator_content = create_variable_table([
            'files', 'partitions', 'tasks', 'total_row_groups', 'parallelism',
            'skipped_row_groups', 'row_count_in', 'row_count_out', 'num_chunks',
            'task_rowsInCount_0_50_75_90_100', 'RowsInPerThread_max', 'seek_io_count',
            'InputRowsPerThread_0_50_75_90_100', 'read_io_count', 'read_io_bytes'
        ])
        else:
            operator_content = create_variable_table(
               ['row_count_in', 'Join_build_row_count_in', 'row_count_out', 'RowsInPerThread_max',
                'InputRowsPerThread_0_50_75_90_100', 'num_chunks', 'join_type'
               ] )


    elif tab_selected =='B':
        if operator_type == 'tablescan':
            operator_content=create_variable_table(
                ['thread_duration_max','cost_percent_str_Values','parquet_task_cost_percent_Values',
                'parquet_reading_cost_percent_Values','read_io_time_percent_Values',
                'filtering_cost_percent_Values','open_time_percent_Values'])

        else:
            operator_content=create_variable_table([
            'thread_duration_max'])

    elif tab_selected == 'C':

        # Assuming you have already computed the operator_data DataFrame (tabscan_df)

        numeric_columns = operator_data.select_dtypes(include=[float, int]).columns
        correlation_matrix = operator_data[numeric_columns].corr()

        # Create a heatmap using Plotly Express
        correlation_fig = px.imshow(correlation_matrix,
                                    x=correlation_matrix.columns,
                                    y=correlation_matrix.columns,
                                    color_continuous_scale='icefire')
        correlation_fig.update_layout(
            title="Correlation Matrix (With Labels and Color Bar)",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,  # Adjust the width as needed
            height=800  # Adjust the height as needed
        )
        operator_content = dcc.Graph(figure=correlation_fig)



    elif tab_selected == 'D':
        # Define your anomaly_type_table for both 'tablescan' and 'joinscan' here
        anomaly_type_table = dash_table.DataTable(
            id='anomaly-type-table',
           columns=[
                    {'name': col, 'id': col} for col in operator_data['anomaly_type'].value_counts().reset_index().columns
                ],
            data=operator_data['anomaly_type'].value_counts().reset_index().to_dict('records'),
            # Define the rest of the styles and configurations
            style_table={
                'textAlign': 'left',
                'margin': 'auto',
                'width': '50%',
                'border': '1px solid #ddd',
                'borderCollapse': 'collapse',
            },
            style_header={
                'backgroundColor': '#007BFF',
                'color': 'white',
                'fontWeight': 'bold',
            },
            style_cell={
                'textAlign': 'left',
                'border': '1px solid #ddd',
                'padding': '10px',
                'backgroundColor': '#F2F2F2',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#F9F9F9',
                },
            ],
        )
        operator_content = anomaly_type_table

    else:
        operator_content = html.Div()  # Placeholder for other tabs (E)

    return operator_content

if __name__ == '__main__':
    app.run_server(debug=True, port=8059)
