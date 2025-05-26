import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import os

# Dictionary of region coordinates (latitude, longitude)
region_coordinates = {
    'North America': (40.0, -100.0),
    'South America': (-15.0, -60.0),
    'Europe': (50.0, 10.0),
    'Africa': (0.0, 20.0),
    'Middle East': (25.0, 45.0),
    'Asia': (30.0, 100.0),
    'Australia': (-25.0, 135.0),
    'Caribbean': (20.0, -75.0),
    'Central America': (15.0, -90.0),
    'Southeast Asia': (10.0, 105.0),
    'East Asia': (35.0, 120.0),
    'South Asia': (25.0, 80.0),
    'Central Asia': (45.0, 60.0),
    'Eastern Europe': (55.0, 30.0),
    'Northern Europe': (60.0, 15.0),
    'Southern Europe': (40.0, 15.0),
    'Western Europe': (48.0, 5.0),
    'Northern Africa': (25.0, 15.0),
    'Western Africa': (10.0, 0.0),
    'Eastern Africa': (5.0, 35.0),
    'Southern Africa': (-25.0, 25.0),
    'Central Africa': (0.0, 20.0),
    'Oceania': (-10.0, 150.0),
    'Pacific Islands': (0.0, 160.0)
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the data
def load_data():
    try:
        # Try to load from the specified path
        df = pd.read_csv('SummerInternVIT/app/simulated_transactions.csv')
    except FileNotFoundError:
        try:
            # Try alternative paths
            df = pd.read_csv('simulated_transactions.csv')
        except FileNotFoundError:
            # Create a sample dataset if file not found
            print("Warning: CSV file not found. Creating sample data.")
            df = create_sample_data()
    
    # Ensure required columns exist
    required_columns = ['send region', 'receiver region', 'amount', 'transaction_status', 'payment_method']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 'Unknown'
    
    if 'year' not in df.columns:
        df['year'] = 2023
        
    # Create amount categories if not present
    if 'amount_category' not in df.columns:
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=[0, 1e6, 5e6, 1e7, float('inf')], 
            labels=['< 1M', '1M-5M', '5M-10M', '> 10M']
        )
    
    return df

def create_sample_data():
    """Create sample data if CSV is not found"""
    regions = list(region_coordinates.keys())[:10]  # Use first 10 regions
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'sender': np.random.choice(['Bank_A', 'Bank_B', 'Bank_C', 'Bank_D'], n_samples),
        'receiver': np.random.choice(['Bank_E', 'Bank_F', 'Bank_G', 'Bank_H'], n_samples),
        'send region': np.random.choice(regions, n_samples),
        'receiver region': np.random.choice(regions, n_samples),
        'amount': np.random.lognormal(15, 1, n_samples),
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY'], n_samples),
        'mx/mt': np.random.choice(['MX', 'MT'], n_samples),
        'Mt': np.random.choice(['MT103', 'MT202', 'MT910'], n_samples),
        'direction': np.random.choice(['Inbound', 'Outbound'], n_samples),
        'transaction_status': np.random.choice(['Completed', 'Pending', 'Failed'], n_samples, p=[0.8, 0.15, 0.05]),
        'payment_method': np.random.choice(['Wire', 'ACH', 'Check'], n_samples),
        'year': np.random.choice([2021, 2022, 2023, 2024], n_samples)
    }
    
    return pd.DataFrame(data)

# Load data
df = load_data()

def create_sankey_diagram(filtered_df, send_region=None, receive_region=None, year=None):
    """Create Sankey diagram with maintained colors and flow animations"""
    
    # Filter data
    title_parts = []
    if send_region and send_region != 'All':
        filtered_df = filtered_df[filtered_df['send region'] == send_region]
        title_parts.append(f"from {send_region}")
    
    if receive_region and receive_region != 'All':
        filtered_df = filtered_df[filtered_df['receiver region'] == receive_region]
        title_parts.append(f"to {receive_region}")
    
    if year and year != 'All':
        filtered_df = filtered_df[filtered_df['year'] == year]
        title_parts.append(f"in {year}")
    
    title = f"Transactions {' '.join(title_parts)}" if title_parts else "All Regional Transactions"
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text=f"No data available for {title}",
            annotations=[dict(
                text="No transactions found for the selected criteria",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )]
        )
        return fig
    
    # Ensure string columns
    str_columns = ['sender', 'receiver', 'currency', 'mx/mt', 'Mt', 'direction', 
                  'transaction_status', 'payment_method', 'amount_category']
    for col in str_columns:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].astype(str)
    
    # Create flows
    flows = []
    flows.append(filtered_df.groupby(['sender', 'receiver']).size().reset_index(name='count'))
    flows.append(filtered_df.groupby(['receiver', 'currency']).size().reset_index(name='count'))
    flows.append(filtered_df.groupby(['currency', 'mx/mt']).size().reset_index(name='count'))
    flows.append(filtered_df.groupby(['mx/mt', 'Mt']).size().reset_index(name='count'))
    flows.append(filtered_df.groupby(['Mt', 'direction']).size().reset_index(name='count'))
    flows.append(filtered_df.groupby(['direction', 'transaction_status']).size().reset_index(name='count'))
    flows.append(filtered_df.groupby(['transaction_status', 'payment_method']).size().reset_index(name='count'))
    flows.append(filtered_df.groupby(['payment_method', 'amount_category']).size().reset_index(name='count'))
    
    # Get unique labels
    all_labels = []
    for flow in flows:
        all_labels.extend(flow.iloc[:, 0].tolist())
        all_labels.extend(flow.iloc[:, 1].tolist())
    
    labels = list(pd.Series(all_labels).unique())
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Prepare sources, targets, and values
    source, target, value = [], [], []
    link_labels = []
    
    flow_pairs = [
        ('sender', 'receiver'),
        ('receiver', 'currency'),
        ('currency', 'mx/mt'),
        ('mx/mt', 'Mt'),
        ('Mt', 'direction'),
        ('direction', 'transaction_status'),
        ('transaction_status', 'payment_method'),
        ('payment_method', 'amount_category')
    ]
    
    for i, (src_col, tgt_col) in enumerate(flow_pairs):
        for _, row in flows[i].iterrows():
            source.append(label_to_index[row[src_col]])
            target.append(label_to_index[row[tgt_col]])
            value.append(row['count'])
            link_labels.append(f"{row[src_col]} → {row[tgt_col]}<br>Count: {row['count']}")
    
    # Define consistent colors for each node type
    node_colors = ["rgba(31, 119, 180, 0.8)"] * len(labels)
    
    # Color mapping for different node types
    color_map = {
        'sender': "rgba(214, 39, 40, 0.8)",      # Red
        'receiver': "rgba(44, 160, 44, 0.8)",    # Green
        'currency': "rgba(255, 127, 14, 0.8)",   # Orange
        'mx/mt': "rgba(148, 103, 189, 0.8)",     # Purple
        'Mt': "rgba(140, 86, 75, 0.8)",          # Brown
        'direction': "rgba(23, 190, 207, 0.8)",  # Cyan
        'transaction_status': "rgba(188, 189, 34, 0.8)",  # Yellow-green
        'payment_method': "rgba(127, 127, 127, 0.8)",     # Gray
        'amount_category': "rgba(31, 119, 180, 0.8)"      # Blue
    }
    
    # Assign colors based on node type
    for i, label in enumerate(labels):
        if label in flows[0]['sender'].values:
            node_colors[i] = color_map['sender']
        elif label in flows[0]['receiver'].values:
            node_colors[i] = color_map['receiver']
        elif label in flows[1]['currency'].values:
            node_colors[i] = color_map['currency']
        elif label in flows[2]['mx/mt'].values:
            node_colors[i] = color_map['mx/mt']
        elif label in flows[3]['Mt'].values:
            node_colors[i] = color_map['Mt']
        elif label in flows[4]['direction'].values:
            node_colors[i] = color_map['direction']
        elif label in flows[5]['transaction_status'].values:
            node_colors[i] = color_map['transaction_status']
        elif label in flows[6]['payment_method'].values:
            node_colors[i] = color_map['payment_method']
        else:
            node_colors[i] = color_map['amount_category']
    
    # Create link colors based on source node colors
    link_colors = []
    for s in source:
        base_color = node_colors[s].replace("0.8", "0.4")
        link_colors.append(base_color)
    
    # Create the main Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors,
            hoverinfo="all"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            customdata=np.array(link_labels),
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])
    
    # Add flow animation frames
    frames = []
    for i in range(10):
        phase = i / 10.0
        flow_colors = []
        
        for s, t in zip(source, target):
            pos = (s + t) / (2 * len(labels))
            intensity = 0.5 + 0.4 * np.sin(2 * np.pi * (pos - phase))
            base_color = node_colors[s].replace("0.8", str(max(0.2, min(0.9, intensity))))
            flow_colors.append(base_color)
        
        frames.append(go.Frame(
            data=[go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=flow_colors,
                    customdata=np.array(link_labels),
                    hovertemplate='%{customdata}<extra></extra>'
                )
            )],
            name=f"frame{i}"
        ))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play Flow Animation",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                                transition=dict(duration=200)
                            )
                        ]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ],
                x=0.1,
                y=1.02,
            )
        ]
    )
    
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=800,
        width=1200,
        annotations=[
            dict(x=0.02, y=1.05, xref='paper', yref='paper', text='Sender', showarrow=False, font=dict(size=14)),
            dict(x=0.14, y=1.05, xref='paper', yref='paper', text='Receiver', showarrow=False, font=dict(size=14)),
            dict(x=0.25, y=1.05, xref='paper', yref='paper', text='Currency', showarrow=False, font=dict(size=14)),
            dict(x=0.37, y=1.05, xref='paper', yref='paper', text='MX/MT', showarrow=False, font=dict(size=14)),
            dict(x=0.48, y=1.05, xref='paper', yref='paper', text='MT', showarrow=False, font=dict(size=14)),
            dict(x=0.60, y=1.05, xref='paper', yref='paper', text='Direction', showarrow=False, font=dict(size=14)),
            dict(x=0.71, y=1.05, xref='paper', yref='paper', text='Status', showarrow=False, font=dict(size=14)),
            dict(x=0.83, y=1.05, xref='paper', yref='paper', text='Payment', showarrow=False, font=dict(size=14)),
            dict(x=0.94, y=1.05, xref='paper', yref='paper', text='Amount', showarrow=False, font=dict(size=14)),
        ]
    )
    
    return fig

def create_flow_map(df, animate=False):
    
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Aggregate transactions by sender and receiver regions
    region_flows = df.groupby(['send region', 'receiver region'])['amount'].sum().reset_index()
    
    # Filter out flows where sender and receiver are the same
    region_flows = region_flows[region_flows['send region'] != region_flows['receiver region']]
    
    # Create a figure with a map
    fig = go.Figure()
    
    # Add a basemap
    fig.add_trace(go.Scattergeo(
        lon=[],
        lat=[],
        mode='markers',
        marker=dict(
            size=1,
            color='rgba(0,0,0,0)'
        ),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Set up color scale for flows based on amount
    max_amount = region_flows['amount'].max()
    min_amount = region_flows['amount'].min()
    
    # Add region markers
    all_regions = set(region_flows['send region'].unique()).union(set(region_flows['receiver region'].unique()))
    
    # Add only regions that have coordinates
    valid_regions = [region for region in all_regions if region in region_coordinates]
    
    # Create a list of region names, longitudes, and latitudes
    region_names = []
    region_lons = []
    region_lats = []
    
    for region in valid_regions:
        if region in region_coordinates:
            region_names.append(region)
            region_lons.append(region_coordinates[region][1])  # lon
            region_lats.append(region_coordinates[region][0])  # lat
    
    # Add region markers
    fig.add_trace(go.Scattergeo(
        lon=region_lons,
        lat=region_lats,
        text=region_names,
        mode='markers+text',
        marker=dict(
            size=10,
            color='blue',
            line=dict(width=1, color='black')
        ),
        textposition="top center",
        name='Regions',
        hoverinfo='text'
    ))
    
    # Calculate normalized amounts for line widths and colors
    region_flows['normalized_amount'] = (region_flows['amount'] - min_amount) / (max_amount - min_amount)
    region_flows['width'] = 1 + 5 * region_flows['normalized_amount']  # Line width between 1 and 6
    
    # Add flow lines 
    # We'll create separate traces for each flow to enable better control over animations
    for idx, flow in region_flows.iterrows():
        sender = flow['send region']
        receiver = flow['receiver region']
        
        # Skip if we don't have coordinates for either region
        if sender not in region_coordinates or receiver not in region_coordinates:
            continue
        
        # Get coordinates
        sender_lat, sender_lon = region_coordinates[sender]
        receiver_lat, receiver_lon = region_coordinates[receiver]
        
        # Calculate control point for curved lines
        # We'll make a curved line by adding a midpoint that's offset
        mid_lon = (sender_lon + receiver_lon) / 2
        mid_lat = (sender_lat + receiver_lat) / 2
        
        # Add some curvature based on distance
        dist = np.sqrt((receiver_lon - sender_lon)**2 + (receiver_lat - sender_lat)**2)
        curve_strength = 0.1 * dist  # Adjust this factor for more/less curvature
        
        # Calculate perpendicular offset
        dx = receiver_lon - sender_lon
        dy = receiver_lat - sender_lat
        
        # Get perpendicular direction (rotate 90 degrees)
        perp_dx = -dy
        perp_dy = dx
        
        # Normalize and apply curve strength
        magnitude = np.sqrt(perp_dx**2 + perp_dy**2)
        if magnitude > 0:  # Avoid division by zero
            perp_dx = perp_dx / magnitude * curve_strength
            perp_dy = perp_dy / magnitude * curve_strength
        
        # Apply offset to midpoint
        mid_lon += perp_dx
        mid_lat += perp_dy
        
        # Create a smooth curve using multiple points
        lon_points = []
        lat_points = []
        num_points = 20
        
        for i in range(num_points):
            t = i / (num_points - 1)
            # Quadratic Bezier curve
            lon = (1-t)**2 * sender_lon + 2*(1-t)*t * mid_lon + t**2 * receiver_lon
            lat = (1-t)**2 * sender_lat + 2*(1-t)*t * mid_lat + t**2 * receiver_lat
            lon_points.append(lon)
            lat_points.append(lat)
        
        # Calculate color based on normalized amount (blue to red)
        color = f'rgba({int(255 * flow["normalized_amount"])}, 0, {int(255 * (1 - flow["normalized_amount"]))}, 0.8)'
        
        # Add the flow line
        fig.add_trace(go.Scattergeo(
            lon=lon_points,
            lat=lat_points,
            mode='lines',
            line=dict(
                width=flow['width'],
                color=color
            ),
            opacity=0.7,
            name=f"{sender} → {receiver}",
            text=f"{sender} → {receiver}: ${flow['amount']:,.2f}",
            hoverinfo='text',
            customdata=[{
                'sender': sender,
                'receiver': receiver,
                'amount': flow['amount'],
                'normalized_amount': flow['normalized_amount']
            }]
        ))
    
    # Configure the base map
    fig.update_geos(
        showcoastlines=True, coastlinecolor="Black",
        showland=True, landcolor="LightGreen",
        showocean=True, oceancolor="LightBlue",
        showlakes=True, lakecolor="Blue",
        showrivers=True, rivercolor="Blue",
        showcountries=True, countrycolor="Black"
    )
    
    # Add animation frames if requested
    if animate:
        frames = []
        
        for frame_idx in range(20):
            frame_data = []
            
            # First add the basemap and markers (unchanged)
            frame_data.append(fig.data[0])  # Basemap
            frame_data.append(fig.data[1])  # Region markers
            
            # For each flow line, create an animated version
            for flow_idx in range(2, len(fig.data)):
                flow_trace = fig.data[flow_idx]
                
                # Create a progress value that moves along the path
                progress = frame_idx / 20.0
                
                # For animation, we'll show only part of the path up to the current progress
                visible_points = int(progress * len(flow_trace.lon)) + 1
                visible_points = max(2, min(visible_points, len(flow_trace.lon)))
                
                
                animated_trace = go.Scattergeo(
                    lon=flow_trace.lon[:visible_points],
                    lat=flow_trace.lat[:visible_points],
                    mode='lines',
                    line=dict(
                        width=flow_trace.line.width,
                        color=flow_trace.line.color
                    ),
                    opacity=flow_trace.opacity,
                    name=flow_trace.name,
                    text=flow_trace.text,
                    hoverinfo=flow_trace.hoverinfo
                )
                
                frame_data.append(animated_trace)
            
            # Create the frame
            frames.append(go.Frame(
                data=frame_data,
                name=f"frame{frame_idx}"
            ))
        
        
        fig.frames = frames
        
       
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play Flow Animation",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=100, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                    transition=dict(duration=50)
                                )
                            ]
                        )
                    ],
                    x=0.1,
                    y=0.95,
                )
            ]
        )
    
    fig.update_layout(
        title="Global Banking Transaction Flows",
        height=600,
        showlegend=False,
        geo=dict(
            projection_type="natural earth",
            showframe=False,
            showcountries=True
        )
    )
    
    return fig

def create_heatmap(filtered_df):
    """Create heatmap visualization"""
    
    matrix = filtered_df.pivot_table(
        index='send region',
        columns='receiver region',
        values='amount',
        aggfunc='count',
        fill_value=0
    )
    
    fig = px.imshow(
        matrix,
        labels=dict(x="Receiver Region", y="Sender Region", color="Transaction Count"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale='Viridis',
        text_auto=True
    )
    
    fig.update_layout(
        title="Transaction Volume Heatmap",
        height=600
    )
    
    return fig

def create_time_series(filtered_df):
    """Create time series visualization"""
    
    grouped = filtered_df.groupby(['year', 'send region'])['amount'].sum().reset_index()
    
    fig = px.line(
        grouped,
        x='year',
        y='amount',
        color='send region',
        markers=True,
        title='Transaction Trends Over Time'
    )
    
    fig.update_layout(height=500)
    return fig

# Get unique values for dropdowns
all_regions = sorted(list(set(df['send region'].unique()).union(set(df['receiver region'].unique()))))
years = sorted(df['year'].unique())

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.H1("Banking Flows Analysis Dashboard", 
                style={'textAlign': 'center', 'color': '#10a37f', 'marginBottom': 30}),
        html.P("Explore global banking transaction flows with interactive visualizations",
               style={'textAlign': 'center', 'fontSize': 18, 'color': '#666'})
    ], style={'backgroundColor': '#1e1e1e', 'padding': 30, 'margin': 10, 
              'border': '1px solid #444', 'borderRadius': 10, 'color': 'white'}),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Sender Region:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='sender-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': region, 'value': region} for region in all_regions],
                value='All',
                style={'width': '300px'}
            )
        ], style={'display': 'inline-block', 'marginRight': 20}),
        
        html.Div([
            html.Label("Receiver Region:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='receiver-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': region, 'value': region} for region in all_regions],
                value='All',
                style={'width': '300px'}
            )
        ], style={'display': 'inline-block', 'marginRight': 20}),
        
        html.Div([
            html.Label("Year:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': str(year), 'value': year} for year in years],
                value='All',
                style={'width': '200px'}
            )
        ], style={'display': 'inline-block'})
    ], style={'textAlign': 'center', 'margin': 20}),
    
    # Tabs for different visualizations
    dcc.Tabs(id="tabs", value='flow-map', children=[
        dcc.Tab(label='Flow Map', value='flow-map'),
        dcc.Tab(label='Sankey Diagram', value='sankey'),
        dcc.Tab(label='Heatmap', value='heatmap'),
        dcc.Tab(label='Time Series', value='time-series'),
    ]),
    
    html.Div(id='tab-content', style={'margin': 20})
])

# Callbacks
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('sender-dropdown', 'value'),
     Input('receiver-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_tab_content(active_tab, sender_region, receiver_region, year):
    # Filter data
    filtered_df = df.copy()
    
    if sender_region != 'All':
        filtered_df = filtered_df[filtered_df['send region'] == sender_region]
    
    if receiver_region != 'All':
        filtered_df = filtered_df[filtered_df['receiver region'] == receiver_region]
    
    if year != 'All':
        filtered_df = filtered_df[filtered_df['year'] == year]
    
    if active_tab == 'flow-map':
        fig = create_flow_map(filtered_df, animate=True)
        return dcc.Graph(figure=fig, style={'height': '700px'})
    
    elif active_tab == 'sankey':
        fig = create_sankey_diagram(filtered_df, sender_region, receiver_region, year)
        return dcc.Graph(figure=fig, style={'height': '900px'})
    
    elif active_tab == 'heatmap':
        fig = create_heatmap(filtered_df)
        return dcc.Graph(figure=fig, style={'height': '700px'})
    
    elif active_tab == 'time-series':
        fig = create_time_series(filtered_df)
        return dcc.Graph(figure=fig, style={'height': '600px'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)