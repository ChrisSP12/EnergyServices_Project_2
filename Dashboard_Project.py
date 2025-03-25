import dash
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor as RFRegressor
from sklearn.tree import DecisionTreeRegressor
from dash.exceptions import PreventUpdate
import os

# Define external stylesheets (Bootstrap + custom theme)
external_stylesheets = ['https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css']

# Load data
df_CP_2017 = pd.read_csv('Forecast_Data_2017_CP.csv')
df_CP_2017['DateTime'] = pd.to_datetime(df_CP_2017['DateTime'])
df_CP_2017 = df_CP_2017.drop(columns=['DateTime'])  # Drop Date for feature selection

df_real_CP_2017 = pd.read_csv('Real_Results_2017_CP.csv')
df_real_CP_2017['Date'] = pd.to_datetime(df_real_CP_2017['Date'])

df_CP_2019 = pd.read_csv('Forecast_Data_2019_CP.csv')
df_CP_2019['Date'] = pd.to_datetime(df_CP_2019['Date'])
df_real_CP_2019 = pd.read_csv('Real_Results_2019_CP.csv')

# Load raw data for visualization
df_raw_2017 = pd.read_csv('Raw_Data_2017.csv')
df_raw_2017['DateTime'] = pd.to_datetime(df_raw_2017['DateTime'])

# Load 2019 raw data with proper datetime handling
df_raw_2019 = pd.read_csv('Raw_data_2019.csv')
# Ensure datetime column exists and is properly named
if 'DateTime' not in df_raw_2019.columns:
    # Try to find datetime column by type or common names
    datetime_cols = [col for col in df_raw_2019.columns if pd.api.types.is_datetime64_any_dtype(df_raw_2019[col])]
    if not datetime_cols:
        datetime_cols = [col for col in df_raw_2019.columns if 'date' in col.lower() or 'time' in col.lower()]
    if datetime_cols:
        df_raw_2019 = df_raw_2019.rename(columns={datetime_cols[0]: 'DateTime'})
    else:
        # If no datetime column found, try to parse first column as datetime
        try:
            df_raw_2019['DateTime'] = pd.to_datetime(df_raw_2019.iloc[:, 0])
        except:
            raise ValueError("Could not find or parse datetime column in 2019 data")
df_raw_2019['DateTime'] = pd.to_datetime(df_raw_2019['DateTime'])

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = "IST Energy Forecast"

# Store for keeping metrics
metrics_store = dcc.Store(id='metrics-store')

# Define CSS for light theme
app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'color': '#333'}, children=[
    html.Div([  
        html.Div([  
            html.Img(src='/assets/Project_1_pic.jpg', style={  
                'width': '100%',  
                'height': '100vh',  
                'objectFit': 'cover',  
                'filter': 'brightness(25%)'  
            }),  
        ], style={  
            'position': 'relative',  
            'overflow': 'hidden',  
        }),  
        html.Div([  
            html.A("IST Energy Forecast Tool", href="#dashboard-content", className="display-3 fw-bold text-white text-decoration-none", style={'cursor': 'pointer'}),  
            html.P("Click to explore IST's energy consumption and forecasting during the beginning of 2019", className="text-white mt-2")  
        ], className="position-absolute top-50 start-50 translate-middle text-center")  
    ], className="position-relative bg-light vh-100 d-flex align-items-center justify-content-center"),  

    html.Div(id="dashboard-content", children=[  
        metrics_store,
        dcc.Tabs(id='tabs', value='tab-0', children=[  
            dcc.Tab(label='Raw Data', value='tab-0'),
            dcc.Tab(label='Feature Analysis', value='tab-1'),
            dcc.Tab(label='Forecast', value='tab-2'),
            dcc.Tab(label='Model Metrics', value='tab-3'),
        ], style={'fontSize': '18px'}),  
        html.Div(id='tabs-content')  
    ])  
])

# Raw Data tab layout with date range picker and year selector
raw_data_tab = html.Div([
    html.Div([
        html.H4("Raw Data Visualization", className='mb-4 text-center'),
        dcc.RadioItems(
            id='year-selector',
            options=[
                {'label': '2017 Data', 'value': '2017'},
                {'label': '2019 Data', 'value': '2019'}
            ],
            value='2017',
            labelStyle={'display': 'inline-block', 'margin-right': '15px'},
            className='mb-3'
        ),
        dcc.Dropdown(
            id='raw-data-column',
            options=[],  # Will be updated in callback
            value=[],  # Will be updated in callback
            multi=True,
            className='mb-4'
        ),
        html.Label("Select Date Range:", className="fw-bold text-center d-block text-dark mb-3"),
        dcc.DatePickerRange(
            id='date-range-picker',
            min_date_allowed=df_raw_2017['DateTime'].min(),
            max_date_allowed=df_raw_2017['DateTime'].max(),
            start_date=df_raw_2017['DateTime'].min(),
            end_date=df_raw_2017['DateTime'].max(),
            display_format='YYYY-MM-DD',
            className='mb-4'
        ),
        dcc.Graph(id='raw-data-graph')
    ], className='container mt-4')
])

# Feature Analysis tab layout with three methods
feature_analysis_tab = html.Div([
    html.Div([
        html.H4("Feature Analysis (Takes a While to Load)", className='mb-4 text-center'),
        html.Label("Select Target Variable:", className="fw-bold text-center d-block text-dark mb-3"),
        dcc.Dropdown(
            id='target-variable',
            options=[{'label': col, 'value': col} for col in df_raw_2017.columns if col != 'DateTime'],
            value='Civil Building Demand (kW)',
            clearable=False,
            className='mb-4'
        ),
        html.Label("Number of Top Features to Select:", className="fw-bold text-center d-block text-dark mb-3"),
        dcc.Slider(
            id='k-features',
            min=1,
            max=10,
            step=1,
            value=3,
            marks={i: str(i) for i in range(1, 11)},
            className='mb-4'
        ),
        dcc.Tabs([
            dcc.Tab(label='Filter Method (KBest)', children=[
                dcc.Graph(id='filter-method-graph'),
                html.Div(id='filter-method-output', className='mt-4')
            ]),
            dcc.Tab(label='Wrapper Method (RFE)', children=[
                dcc.Graph(id='wrapper-method-graph'),
                html.Div(id='wrapper-method-output', className='mt-4')
            ]),
            dcc.Tab(label='Ensemble Method (RF Importance)', children=[
                dcc.Graph(id='ensemble-method-graph'),
                html.Div(id='ensemble-method-output', className='mt-4')
            ])
        ])
    ], className='container mt-4')
])

# Forecast tab layout with model selection and scatter plot
forecast_tab = html.Div([  
    html.Label("Select Features for Model Training:", className="fw-bold text-center d-block text-dark"),  
    dcc.Checklist(  
        id='feature-checklist',  
        options=[  
            {'label': 'GHI (W/m2)', 'value': 'GHI (W/m2)'},  
            {'label': 'Civil Building Demand (kW) - 1', 'value': 'Civil Building Demand (kW) - 1'},  
            {'label': 'Week Day', 'value': 'Week Day'},  
            {'label': 'Month', 'value': 'Month'},
            {'label': 'Sin Hour', 'value': 'Sin Hour'},  
        ],  
        value=['GHI (W/m2)', 'Civil Building Demand (kW) - 1'],  
        labelStyle={'display': 'inline-block', 'margin': '10px'},  
        style={'color': '#333'}  
    ),  
    html.Div([  
        dcc.Loading(  
            id="loading",  
            type="circle",  
            children=[
                html.Div([
                    html.Label("Select Models to Display:", className="fw-bold text-center d-block text-dark mb-2"),
                    dcc.Dropdown(
                        id='model-selector',
                        options=[
                            {'label': 'Linear Regression', 'value': 'LinearRegression'},
                            {'label': 'Random Forest', 'value': 'RandomForest'},
                            {'label': 'Neural Network', 'value': 'NeuralNetwork'}
                        ],
                        value=['LinearRegression', 'RandomForest', 'NeuralNetwork'],
                        multi=True,
                        className='mb-3'
                    ),
                    html.Button("Train Models", id='train-models-button', className='btn btn-primary')
                ])
            ]  
        ),  
    ], className="container text-center mt-4"),  
    html.Div([  
        dcc.Graph(id='forecast-graph'),  
    ], className="container text-center mt-5"),
    html.Div([  
        dcc.Graph(id='scatter-plot'),  
    ], className="container text-center mt-5"),  
])

# Metrics tab layout
metrics_tab = html.Div([
    html.Div([
        html.Label("Select Metrics to Display:", className="fw-bold text-center d-block text-dark mb-3"),
        dcc.Checklist(
            id='metrics-checklist',
            options=[
                {'label': 'MSE', 'value': 'mse'},
                {'label': 'RMSE', 'value': 'rmse'},
                {'label': 'MAE', 'value': 'mae'},
                {'label': 'cvRMSE (%)', 'value': 'cv_rmse'},
                {'label': 'NMBE (%)', 'value': 'nmbe'},
                {'label': 'R²', 'value': 'r2'}
            ],
            value=['mse', 'rmse', 'mae', 'cv_rmse', 'nmbe', 'r2'],
            labelStyle={'display': 'inline-block', 'margin': '5px 15px'},
            style={'color': '#333'},
            className='mb-4'
        ),
    ], className='container'),
    html.Div(id='metrics-output', children='Train models first to see metrics.')
])

# Callback to update content based on selected tab
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_tab_content(tab):
    if tab == 'tab-0':
        return raw_data_tab
    elif tab == 'tab-1':
        return feature_analysis_tab
    elif tab == 'tab-2':
        return forecast_tab
    elif tab == 'tab-3':
        return metrics_tab

# Callback to update dropdown options based on selected year
@app.callback(
    [Output('raw-data-column', 'options'),
     Output('raw-data-column', 'value')],
    [Input('year-selector', 'value')]
)
def update_dropdown_options(selected_year):
    if selected_year == '2017':
        df = df_raw_2017
    else:
        df = df_raw_2019
    
    options = [{'label': col, 'value': col} for col in df.columns if col != 'DateTime']
    value = [options[0]['value']] if options else []
    
    return options, value

# Updated callback for raw data graph with year selection and date filtering
@app.callback(
    [Output('raw-data-graph', 'figure'),
     Output('date-range-picker', 'min_date_allowed'),
     Output('date-range-picker', 'max_date_allowed'),
     Output('date-range-picker', 'start_date'),
     Output('date-range-picker', 'end_date')],
    [Input('raw-data-column', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('year-selector', 'value')]
)
def update_raw_data_graph(selected_columns, start_date, end_date, selected_year):
    if not selected_columns:
        return px.line(title="Please select at least one feature to display"), dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Select the appropriate dataframe based on year
        if selected_year == '2017':
            df = df_raw_2017.copy()
        else:
            df = df_raw_2019.copy()
        
        # Ensure DateTime column exists and is in datetime format
        if 'DateTime' not in df.columns:
            raise ValueError("DateTime column not found in the dataset")
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Update date picker range based on selected year
        min_date = df['DateTime'].min()
        max_date = df['DateTime'].max()
        
        # Convert start_date and end_date to datetime if they're not None
        start_date = pd.to_datetime(start_date) if start_date else min_date
        end_date = pd.to_datetime(end_date) if end_date else max_date
        
        # Filter data based on selected date range
        filtered_df = df[
            (df['DateTime'] >= start_date) & 
            (df['DateTime'] <= end_date)
        ].copy()
        
        # Ensure selected columns exist in the dataframe
        valid_columns = [col for col in selected_columns if col in df.columns]
        if not valid_columns:
            raise ValueError("No valid columns selected for visualization")
        
        fig = px.line(filtered_df, 
                     x='DateTime', 
                     y=valid_columns,
                     title=f"{selected_year} Raw Data Features Over Time ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
                     labels={'value': 'Value', 'variable': 'Feature', 'DateTime': 'Date/Time'},
                     template='plotly_dark')
        
        fig.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333',
            hovermode='x unified',
            legend_title_text='Features',
            xaxis_title='Date/Time'
        )
        
        fig.update_traces(line=dict(width=2))
        
        return fig, min_date, max_date, min_date, max_date
    
    except Exception as e:
        print(f"Error: {str(e)}")
        error_fig = px.line(title=f"Error loading data: {str(e)}")
        error_fig.update_layout(
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333'
        )
        return error_fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback for feature analysis with updated wrapper method using DecisionTreeRegressor
@app.callback(
    [Output('filter-method-graph', 'figure'),
     Output('filter-method-output', 'children'),
     Output('wrapper-method-graph', 'figure'),
     Output('wrapper-method-output', 'children'),
     Output('ensemble-method-graph', 'figure'),
     Output('ensemble-method-output', 'children')],
    [Input('target-variable', 'value'),
     Input('k-features', 'value')]
)
def update_feature_analysis(target, k_value):
    if target is None:
        raise PreventUpdate
    
    # Prepare data (exclude datetime and target)
    features = [col for col in df_raw_2017.columns if col not in ['DateTime', target]]
    X = df_raw_2017[features]
    y = df_raw_2017[target]
    
    # Handle NaN values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # 1. Filter Method (KBest)
    try:
        selector_kbest = SelectKBest(score_func=f_regression, k='all')
        selector_kbest.fit(X, y)
        scores = selector_kbest.scores_
        
        # Create KBest feature importance plot
        fig_kbest = go.Figure()
        fig_kbest.add_trace(go.Bar(
            x=scores,
            y=features,
            orientation='h',
            marker_color='#007bff'
        ))
        
        fig_kbest.update_layout(
            title='Filter Method: SelectKBest (F Regression)',
            xaxis_title='F-score',
            yaxis_title='Features',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333',
            height=600
        )
        
        # Get top K features from KBest
        selector_kbest_top = SelectKBest(score_func=f_regression, k=k_value)
        selector_kbest_top.fit(X, y)
        selected_features_kbest = np.array(features)[selector_kbest_top.get_support()]
        selected_scores_kbest = scores[selector_kbest_top.get_support()]
        
        # Create output text for KBest
        output_kbest = [
            html.H5(f"Top {k_value} Selected Features (Filter Method):", className='mt-4'),
            html.Ul([html.Li(f"{feature} (Score: {score:.2f})") 
                    for feature, score in zip(selected_features_kbest, selected_scores_kbest)])
        ]
    except Exception as e:
        fig_kbest = go.Figure()
        fig_kbest.update_layout(
            title='Error in Filter Method Calculation',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333',
            height=600
        )
        output_kbest = html.Div(f"Error calculating filter method: {str(e)}", className='text-danger')
    
    # 2. Wrapper Method (RFE) with DecisionTreeRegressor
    try:
        model = DecisionTreeRegressor()
        rfe = RFE(model, n_features_to_select=k_value)
        rfe.fit(X, y)
        
        # Create RFE feature importance plot
        fig_rfe = go.Figure()
        fig_rfe.add_trace(go.Bar(
            x=rfe.ranking_,
            y=features,
            orientation='h',
            marker_color='#28a745'
        ))
        
        fig_rfe.update_layout(
            title='Wrapper Method: Recursive Feature Elimination (RFE)',
            xaxis_title='Feature Ranking (1 = selected)',
            yaxis_title='Features',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333',
            height=600
        )
        
        # Get selected features from RFE
        selected_features_rfe = np.array(features)[rfe.support_]
        rankings_rfe = rfe.ranking_[rfe.support_]
        
        # Create output text for RFE
        output_rfe = [
            html.H5(f"Top {k_value} Selected Features (Wrapper Method):", className='mt-4'),
            html.Ul([html.Li(f"{feature} (Rank: {rank})") 
                    for feature, rank in zip(selected_features_rfe, rankings_rfe)])
        ]
    except Exception as e:
        fig_rfe = go.Figure()
        fig_rfe.update_layout(
            title='Error in Wrapper Method Calculation',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333',
            height=600
        )
        output_rfe = html.Div(f"Error calculating wrapper method: {str(e)}", className='text-danger')
    
    # 3. Ensemble Method (Random Forest Feature Importance)
    try:
        rf = RFRegressor(n_estimators=100)
        rf.fit(X, y)
        
        # Create RF feature importance plot
        fig_rf = go.Figure()
        fig_rf.add_trace(go.Bar(
            x=rf.feature_importances_,
            y=features,
            orientation='h',
            marker_color='#dc3545'
        ))
        
        fig_rf.update_layout(
            title='Ensemble Method: Random Forest Feature Importance',
            xaxis_title='Feature Importance Score',
            yaxis_title='Features',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333',
            height=600
        )
        
        # Get top K features from RF importance
        top_k_indices = np.argsort(rf.feature_importances_)[-k_value:][::-1]
        selected_features_rf = np.array(features)[top_k_indices]
        selected_scores_rf = rf.feature_importances_[top_k_indices]
        
        # Create output text for RF
        output_rf = [
            html.H5(f"Top {k_value} Selected Features (Ensemble Method):", className='mt-4'),
            html.Ul([html.Li(f"{feature} (Score: {score:.4f})") 
                    for feature, score in zip(selected_features_rf, selected_scores_rf)])
        ]
    except Exception as e:
        fig_rf = go.Figure()
        fig_rf.update_layout(
            title='Error in Ensemble Method Calculation',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font_color='#333',
            height=600
        )
        output_rf = html.Div(f"Error calculating ensemble method: {str(e)}", className='text-danger')
    
    return (fig_kbest, output_kbest, 
            fig_rfe, output_rfe, 
            fig_rf, output_rf)

# Callback to train models and update forecast graph and scatter plot
@app.callback(
    [Output('train-models-button', 'children'),
     Output('forecast-graph', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('metrics-store', 'data')],
    [Input('train-models-button', 'n_clicks')],
    [State('feature-checklist', 'value'),
     State('model-selector', 'value')],
    prevent_initial_call=True
)
def train_and_update(n_clicks, selected_features, selected_models):
    if n_clicks is None:
        raise PreventUpdate

    # Prepare training data (2017)
    X_train_2017 = df_CP_2017[selected_features].values
    y_train_2017 = df_real_CP_2017['Civil Building Demand (kW)'].values

    # Train models
    LR_model = LinearRegression()
    LR_model.fit(X_train_2017, y_train_2017)

    RF_model = RandomForestRegressor(n_estimators=100)
    RF_model.fit(X_train_2017, y_train_2017)

    NN_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    NN_model.fit(X_train_2017, y_train_2017)

    # Save models
    joblib.dump(LR_model, 'LR_model.sav')
    joblib.dump(RF_model, 'RF_model.sav')
    joblib.dump(NN_model, 'NN_model.sav')

    # Prepare input data for prediction (2019)
    X_2019 = df_CP_2019[selected_features].values
    y_actual_2019 = df_real_CP_2019['Civil Building Demand (kW)'].values

    # Make predictions using the trained models
    y_pred_LR = LR_model.predict(X_2019)
    y_pred_RF = RF_model.predict(X_2019)
    y_pred_NN = NN_model.predict(X_2019)

    # Add predictions to the DataFrame
    df_CP_2019['LinearRegression'] = y_pred_LR
    df_CP_2019['RandomForest'] = y_pred_RF
    df_CP_2019['NeuralNetwork'] = y_pred_NN
    df_CP_2019['Actual'] = y_actual_2019

    # Create time series forecast graph
    columns_to_display = ['Actual']
    display_names = ['Actual Data']
    
    if 'LinearRegression' in selected_models:
        columns_to_display.append('LinearRegression')
        display_names.append('Linear Regression')
    if 'RandomForest' in selected_models:
        columns_to_display.append('RandomForest')
        display_names.append('Random Forest')
    if 'NeuralNetwork' in selected_models:
        columns_to_display.append('NeuralNetwork')
        display_names.append('Neural Network')

    forecast_fig = px.line(df_CP_2019, 
                  x='Date', 
                  y=columns_to_display,
                  title="Forecast vs Actual Demand for 2019",
                  labels={'value': 'Demand (kW)', 'variable': 'Model'},
                  template='plotly_dark')

    for i, name in enumerate(display_names):
        forecast_fig.data[i].name = name

    forecast_fig.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f8f9fa',
        font_color='#333',
        legend_title_text='Models/Actual Data'
    )

    # Create scatter plot with diagonal line
    scatter_fig = go.Figure()
    
    # Add diagonal line
    max_val = max(max(y_actual_2019), max(y_pred_LR), max(y_pred_RF), max(y_pred_NN))
    scatter_fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='grey', dash='dash'),
        name='Perfect Prediction'
    ))
    
    # Add scatter plots for selected models
    if 'LinearRegression' in selected_models:
        scatter_fig.add_trace(go.Scatter(
            x=y_actual_2019,
            y=y_pred_LR,
            mode='markers',
            name='Linear Regression',
            marker=dict(color='blue')
        ))
    
    if 'RandomForest' in selected_models:
        scatter_fig.add_trace(go.Scatter(
            x=y_actual_2019,
            y=y_pred_RF,
            mode='markers',
            name='Random Forest',
            marker=dict(color='green')
        ))
    
    if 'NeuralNetwork' in selected_models:
        scatter_fig.add_trace(go.Scatter(
            x=y_actual_2019,
            y=y_pred_NN,
            mode='markers',
            name='Neural Network',
            marker=dict(color='red')
        ))
    
    scatter_fig.update_layout(
        title='Actual vs Predicted Demand',
        xaxis_title='Actual Demand (kW)',
        yaxis_title='Predicted Demand (kW)',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f8f9fa',
        font_color='#333',
        showlegend=True
    )
    
    scatter_fig.update_xaxes(range=[0, max_val])
    scatter_fig.update_yaxes(range=[0, max_val])

    # Calculate all metrics
    metrics = []
    for model, y_pred in zip(['Linear Regression', 'Random Forest', 'Neural Network'], 
                           [y_pred_LR, y_pred_RF, y_pred_NN]):
        mse = mean_squared_error(y_actual_2019, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_actual_2019 - y_pred))
        cv_rmse = (rmse / np.mean(y_actual_2019)) * 100  # Coefficient of Variation RMSE (%)
        nmbe = (np.mean(y_pred - y_actual_2019) / np.mean(y_actual_2019)) * 100  # Normalized Mean Bias Error (%)
        r2 = r2_score(y_actual_2019, y_pred)
        
        metrics.append({
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'cv_rmse': cv_rmse,
            'nmbe': nmbe,
            'r2': r2
        })
    
    return "Models Trained Successfully!", forecast_fig, scatter_fig, metrics

# Callback to update metrics tab when data is available
@app.callback(
    Output('metrics-output', 'children'),
    [Input('metrics-store', 'data'),
     Input('metrics-checklist', 'value')]
)
def update_metrics(data, selected_metrics):
    if data is None:
        return 'Train models first to see metrics.'
    
    # Define metric display names and formatting
    metric_display = {
        'mse': {'name': 'MSE', 'format': '{:.2f}'},
        'rmse': {'name': 'RMSE', 'format': '{:.2f}'},
        'mae': {'name': 'MAE', 'format': '{:.2f}'},
        'cv_rmse': {'name': 'cvRMSE (%)', 'format': '{:.2f}'},
        'nmbe': {'name': 'NMBE (%)', 'format': '{:.2f}'},
        'r2': {'name': 'R²', 'format': '{:.4f}'}
    }
    
    # Create table header
    header = [html.Th("Model", className='text-start')]
    for metric in selected_metrics:
        header.append(html.Th(metric_display[metric]['name'], className='text-center'))
    
    header = html.Tr(header, className='table-dark')

    # Create table rows
    rows = []
    for model_data in data:
        row = [html.Td(model_data['model'], className='text-start fw-bold')]
        for metric in selected_metrics:
            row.append(html.Td(
                metric_display[metric]['format'].format(model_data[metric]), 
                className='text-center'
            ))
        rows.append(html.Tr(row, className='border-bottom'))

    # Create the table
    metrics_table = html.Table([
        html.Thead(header),
        html.Tbody(rows)
    ], className='table table-striped table-bordered table-hover')

    return html.Div([
        html.H4("Model Performance Metrics", className='mb-4 text-center'),
        html.Div(metrics_table, className='table-responsive')
    ], className='container mt-4')

if __name__ == '__main__':
    app.run_server()
