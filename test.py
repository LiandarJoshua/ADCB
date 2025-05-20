import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import json

# Define region coordinates (approximate centroids)
region_coordinates = {
    "North America": {"lat": 40.0, "lon": -100.0},
    "Europe": {"lat": 50.0, "lon": 10.0},
    "Asia Pacific": {"lat": 30.0, "lon": 120.0},
    "Middle East": {"lat": 27.0, "lon": 45.0},
    "Africa": {"lat": 5.0, "lon": 20.0},
    "Latin America": {"lat": -10.0, "lon": -60.0},
    "Caribbean": {"lat": 20.0, "lon": -75.0},
    "Central Asia": {"lat": 45.0, "lon": 65.0},
    "South Asia": {"lat": 20.0, "lon": 75.0},
    "Southeast Asia": {"lat": 10.0, "lon": 105.0}
}

# Function to prepare flow data between regions
def prepare_region_flow_data(df):
    # Group by sender and receiver regions
    grouped = df.groupby(['send region', 'receiver region'])['amount'].sum().reset_index()
    
    # Create a copy to avoid modifying the original
    flow_data = grouped.copy()
    
    # Add coordinates for source and target regions
    flow_data['source_lat'] = flow_data['send region'].apply(lambda x: region_coordinates[x]['lat'])
    flow_data['source_lon'] = flow_data['send region'].apply(lambda x: region_coordinates[x]['lon'])
    flow_data['target_lat'] = flow_data['receiver region'].apply(lambda x: region_coordinates[x]['lat'])
    flow_data['target_lon'] = flow_data['receiver region'].apply(lambda x: region_coordinates[x]['lon'])
    
    # Generate unique IDs for each flow
    flow_data['flow_id'] = flow_data['send region'] + '_to_' + flow_data['receiver region']
    
    return flow_data

# Function to create choropleth map with clickable flows
def create_clickable_flow_map(df):
    # Prepare flow data
    flow_data = prepare_region_flow_data(df)
    
    # Calculate total inflows and outflows for each region
    inflows = df.groupby('receiver region')['amount'].sum().reset_index()
    inflows.columns = ['region', 'inflow']
    
    outflows = df.groupby('send region')['amount'].sum().reset_index()
    outflows.columns = ['region', 'outflow']
    
    # Merge inflows and outflows
    region_flows = pd.merge(inflows, outflows, on='region', how='outer').fillna(0)
    
    # Calculate net flow (inflow - outflow)
    region_flows['net_flow'] = region_flows['inflow'] - region_flows['outflow']
    region_flows['total_volume'] = region_flows['inflow'] + region_flows['outflow']
    
    # Create coordinates dataframe for the choropleth
    region_coords = pd.DataFrame(region_coordinates).T.reset_index()
    region_coords.columns = ['region', 'lat', 'lon']
    
    # Merge coordinates with flow data
    region_data = pd.merge(region_flows, region_coords, on='region', how='left')
    
    # Normalize flow values for better visualization
    max_amount = flow_data['amount'].max()
    flow_data['normalized_amount'] = flow_data['amount'] / max_amount * 10
    
    # Create figure with mapbox background
    fig = go.Figure()
    
    # Add flows as custom shapes (arcs between regions)
    for _, row in flow_data.iterrows():
        # Skip flows below threshold to reduce clutter
        if row['normalized_amount'] < 0.5:
            continue
        
        # Calculate midpoint for the curve (with slight offset for visualization)
        lon_diff = row['target_lon'] - row['source_lon']
        lat_diff = row['target_lat'] - row['source_lat']
        
        # Adjust midpoint upward based on distance
        distance = np.sqrt(lon_diff**2 + lat_diff**2)
        midpoint_offset = min(distance * 0.15, 10)  # Cap the offset
        
        # Generate curve points
        curve_points = []
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            # Parametric curve formula
            x = (1-t)**2 * row['source_lon'] + 2*(1-t)*t * ((row['source_lon'] + row['target_lon'])/2) + t**2 * row['target_lon']
            y = (1-t)**2 * row['source_lat'] + 2*(1-t)*t * ((row['source_lat'] + row['target_lat'])/2 + midpoint_offset) + t**2 * row['target_lat']
            curve_points.append((x, y))
        
        # Extract lons and lats from points
        lons, lats = zip(*curve_points)
        
        # Create curve as a line - each flow has a unique customdata for identification
        # Changed from scattermapbox to scattermap
        fig.add_trace(go.Scattermap(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(
                width=row['normalized_amount'] * 1.5,
                color=f'rgba(70, 130, 180, {min(0.8, 0.3 + row["normalized_amount"] * 0.05)})'
            ),
            hoverinfo='text',
            hovertext=f"{row['send region']} → {row['receiver region']}<br>Amount: ${row['amount']:,.2f}",
            showlegend=False,
            customdata=[row['flow_id']],  # Add flow_id as customdata for click events
            name=f"Flow: {row['send region']} to {row['receiver region']}"
        ))
    
    # Add regions as points
    # Changed from scattermapbox to scattermap
    fig.add_trace(go.Scattermap(
        lon=region_data['lon'],
        lat=region_data['lat'],
        text=region_data['region'],
        customdata=np.dstack((
            region_data['inflow'].round(2),
            region_data['outflow'].round(2),
            region_data['net_flow'].round(2),
            region_data['region']  # Include region name in customdata
        ))[0],
        hovertemplate='<b>%{text}</b><br>Inflow: $%{customdata[0]:,.2f}<br>Outflow: $%{customdata[1]:,.2f}<br>Net Flow: $%{customdata[2]:,.2f}',
        mode='markers',
        marker=dict(
            size=region_data['total_volume'] / region_data['total_volume'].max() * 25 + 10,
            color=region_data['net_flow'],
            colorscale='RdBu',
            colorbar=dict(
                title='Net Flow<br>(Inflow - Outflow)',
                thickness=15
            ),
            cmid=0,  # Center colorscale at 0
            opacity=0.8
        ),
        name='Regions'
    ))
    
    # Set mapbox style
    mapbox = dict(
        style='carto-positron',  # Use Carto base map (no token needed)
        zoom=1.2,
        center=dict(lat=20, lon=0)
    )
    
    # Update layout
    fig.update_layout(
        title='Global Banking Transaction Flows Between Regions<br>Click on any flow to see detailed breakdown',
        mapbox=mapbox,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig, flow_data

# Function to create a Sankey diagram for a specific flow
def create_detailed_sankey(df, source_region, target_region):
    """
    Create a Sankey diagram showing the detailed breakdown of flows between two regions
    
    Parameters:
    df (DataFrame): The original transaction data
    source_region (str): The source region
    target_region (str): The target region
    
    Returns:
    A Plotly Figure object with the Sankey diagram
    """
    # Filter data for the specific flow
    flow_df = df[(df['send region'] == source_region) & (df['receiver region'] == target_region)]
    
    if flow_df.empty:
        # If no data is found, return a message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No detailed data available for flow: {source_region} to {target_region}",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract additional dimensions for detailed breakdown
    # Assuming df has columns like 'send bank', 'receiver bank', 'transaction_type', etc.
    # Adjust these dimensions based on your actual data structure
    
    # For this example, let's assume we have:
    # 1. Send bank
    # 2. Receiver bank
    # 3. Transaction type
    
    # First, prepare node names and indices
    nodes = []
    node_indices = {}
    
    # Source region is always the first node
    nodes.append(source_region)
    node_indices[source_region] = 0
    
    # Add send banks
    if 'send bank' in flow_df.columns:
        send_banks = flow_df['send bank'].unique()
        for bank in send_banks:
            bank_name = f"{bank} (Sender)"
            nodes.append(bank_name)
            node_indices[bank_name] = len(nodes) - 1
    else:
        # If bank data is not available, create a dummy node
        nodes.append(f"{source_region} Banks")
        node_indices[f"{source_region} Banks"] = len(nodes) - 1
        send_banks = ["Dummy"]
    
    # Add transaction types if available
    if 'transaction_type' in flow_df.columns:
        txn_types = flow_df['transaction_type'].unique()
        for txn_type in txn_types:
            txn_name = f"{txn_type}"
            nodes.append(txn_name)
            node_indices[txn_name] = len(nodes) - 1
    else:
        # If transaction type data is not available, create a dummy node
        nodes.append("Transactions")
        node_indices["Transactions"] = len(nodes) - 1
        txn_types = ["Dummy"]
    
    # Add receiver banks
    if 'receiver bank' in flow_df.columns:
        recv_banks = flow_df['receiver bank'].unique()
        for bank in recv_banks:
            bank_name = f"{bank} (Receiver)"
            nodes.append(bank_name)
            node_indices[bank_name] = len(nodes) - 1
    else:
        # If bank data is not available, create a dummy node
        nodes.append(f"{target_region} Banks")
        node_indices[f"{target_region} Banks"] = len(nodes) - 1
        recv_banks = ["Dummy"]
    
    # Target region is always the last node
    nodes.append(target_region)
    node_indices[target_region] = len(nodes) - 1
    
    # Now, prepare links
    source_indices = []
    target_indices = []
    values = []
    colors = []
    
    # Flow from source region to sending banks
    if 'send bank' in flow_df.columns:
        for bank in send_banks:
            bank_name = f"{bank} (Sender)"
            amount = flow_df[flow_df['send bank'] == bank]['amount'].sum()
            source_indices.append(node_indices[source_region])
            target_indices.append(node_indices[bank_name])
            values.append(amount)
            colors.append("rgba(44, 160, 44, 0.5)")  # Green
    else:
        # If bank data is not available, create a dummy link
        amount = flow_df['amount'].sum()
        source_indices.append(node_indices[source_region])
        target_indices.append(node_indices[f"{source_region} Banks"])
        values.append(amount)
        colors.append("rgba(44, 160, 44, 0.5)")  # Green
    
    # Flow from sending banks to transaction types
    if 'send bank' in flow_df.columns and 'transaction_type' in flow_df.columns:
        for bank in send_banks:
            bank_name = f"{bank} (Sender)"
            for txn_type in txn_types:
                bank_txn_df = flow_df[(flow_df['send bank'] == bank) & (flow_df['transaction_type'] == txn_type)]
                if not bank_txn_df.empty:
                    amount = bank_txn_df['amount'].sum()
                    source_indices.append(node_indices[bank_name])
                    target_indices.append(node_indices[txn_type])
                    values.append(amount)
                    colors.append("rgba(214, 39, 40, 0.5)")  # Red
    elif 'transaction_type' in flow_df.columns:
        # If only transaction type is available
        for txn_type in txn_types:
            amount = flow_df[flow_df['transaction_type'] == txn_type]['amount'].sum()
            source_indices.append(node_indices[f"{source_region} Banks"])
            target_indices.append(node_indices[txn_type])
            values.append(amount)
            colors.append("rgba(214, 39, 40, 0.5)")  # Red
    else:
        # If neither is available
        amount = flow_df['amount'].sum()
        source_indices.append(node_indices[f"{source_region} Banks"])
        target_indices.append(node_indices["Transactions"])
        values.append(amount)
        colors.append("rgba(214, 39, 40, 0.5)")  # Red
    
    # Flow from transaction types to receiving banks
    if 'receiver bank' in flow_df.columns and 'transaction_type' in flow_df.columns:
        for txn_type in txn_types:
            for bank in recv_banks:
                bank_name = f"{bank} (Receiver)"
                bank_txn_df = flow_df[(flow_df['receiver bank'] == bank) & (flow_df['transaction_type'] == txn_type)]
                if not bank_txn_df.empty:
                    amount = bank_txn_df['amount'].sum()
                    source_indices.append(node_indices[txn_type])
                    target_indices.append(node_indices[bank_name])
                    values.append(amount)
                    colors.append("rgba(31, 119, 180, 0.5)")  # Blue
    elif 'receiver bank' in flow_df.columns:
        # If only receiver bank is available
        for bank in recv_banks:
            bank_name = f"{bank} (Receiver)"
            amount = flow_df[flow_df['receiver bank'] == bank]['amount'].sum()
            source_indices.append(node_indices["Transactions"])
            target_indices.append(node_indices[bank_name])
            values.append(amount)
            colors.append("rgba(31, 119, 180, 0.5)")  # Blue
    else:
        # If neither is available
        amount = flow_df['amount'].sum()
        source_indices.append(node_indices["Transactions"])
        target_indices.append(node_indices[f"{target_region} Banks"])
        values.append(amount)
        colors.append("rgba(31, 119, 180, 0.5)")  # Blue
    
    # Flow from receiving banks to target region
    if 'receiver bank' in flow_df.columns:
        for bank in recv_banks:
            bank_name = f"{bank} (Receiver)"
            amount = flow_df[flow_df['receiver bank'] == bank]['amount'].sum()
            source_indices.append(node_indices[bank_name])
            target_indices.append(node_indices[target_region])
            values.append(amount)
            colors.append("rgba(255, 127, 14, 0.5)")  # Orange
    else:
        # If bank data is not available, create a dummy link
        amount = flow_df['amount'].sum()
        source_indices.append(node_indices[f"{target_region} Banks"])
        target_indices.append(node_indices[target_region])
        values.append(amount)
        colors.append("rgba(255, 127, 14, 0.5)")  # Orange
    
    # Create the figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=colors
        )
    )])
    
    # Update layout
    total_amount = flow_df['amount'].sum()
    fig.update_layout(
        title_text=f"Detailed Breakdown: {source_region} to {target_region} (Total: ${total_amount:,.2f})",
        font_size=10,
        height=600
    )
    
    return fig

# Create a Dash application to handle interactivity
def create_interactive_dashboard(df):
    """
    Create an interactive Dash application with the flow map and Sankey diagram
    
    Parameters:
    df (DataFrame): The transaction data
    
    Returns:
    A Dash application
    """
    # Create initial figures
    flow_map, flow_data = create_clickable_flow_map(df)
    
    # First flow to show as default in the Sankey
    if not flow_data.empty:
        first_flow = flow_data.iloc[0]
        initial_source = first_flow['send region']
        initial_target = first_flow['receiver region']
        initial_sankey = create_detailed_sankey(df, initial_source, initial_target)
    else:
        # Create an empty figure if no data
        initial_sankey = go.Figure()
        initial_sankey.add_annotation(
            text="Select a flow on the map to see detailed breakdown",
            showarrow=False,
            font=dict(size=16)
        )
    
    # Initialize the Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    # App layout
    app.layout = html.Div([
        html.H1("Interactive Banking Flow Analysis", style={'textAlign': 'center'}),
        
        # Main visualization container
        html.Div([
            # Flow map
            html.Div([
                dcc.Graph(
                    id='flow-map',
                    figure=flow_map,
                    style={'height': '700px'},
                    config={'displayModeBar': True}
                )
            ], style={'width': '100%', 'display': 'inline-block'}),
            
            # Sankey diagram
            html.Div([
                html.H3(id='sankey-title', children="Detailed Flow Breakdown", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='sankey-diagram',
                    figure=initial_sankey,
                    style={'height': '600px'}
                )
            ], style={'width': '100%', 'display': 'inline-block', 'marginTop': '20px'})
        ]),
        
        # Hidden div to store click data
        html.Div(id='click-data', style={'display': 'none'})
    ])
    
    # Callback to update the Sankey diagram when a flow is clicked
    @app.callback(
        [Output('sankey-diagram', 'figure'),
         Output('sankey-title', 'children')],
        [Input('flow-map', 'clickData')]
    )
    def update_sankey(click_data):
        if click_data is None:
            return initial_sankey, "Select a flow on the map to see detailed breakdown"
        
        # Check if clicked on a flow trace
        try:
            # Get the customdata from the clicked point
            custom_data = click_data['points'][0]['customdata']
            if isinstance(custom_data, list):
                flow_id = custom_data[0]
            else:
                # If it's not a flow trace, return the initial sankey
                return initial_sankey, "Select a flow on the map to see detailed breakdown"
                
            # Extract source and target regions from flow_id
            if "_to_" in flow_id:
                source_region, target_region = flow_id.split('_to_')
                
                # Create the Sankey diagram
                sankey_fig = create_detailed_sankey(df, source_region, target_region)
                
                # Update the title
                sankey_title = f"Detailed Breakdown: {source_region} to {target_region}"
                
                return sankey_fig, sankey_title
            else:
                # If the customdata is not in the expected format
                return initial_sankey, "Invalid flow selection. Please try again."
                
        except (KeyError, IndexError, AttributeError) as e:
            # If there's an error in accessing the data
            print(f"Error accessing click data: {e}")
            return initial_sankey, "Error processing selection. Please try again."
    
    return app

# Sample data creator for testing the visualization
def create_sample_data():
    """
    Create a sample dataset for testing the visualization
    
    Returns:
    A pandas DataFrame with sample banking flow data
    """
    # List of regions
    regions = list(region_coordinates.keys())
    
    # List of banks per region
    banks = {
        "North America": ["Bank of America", "JPMorgan Chase", "Wells Fargo", "Citibank"],
        "Europe": ["Deutsche Bank", "BNP Paribas", "Barclays", "Santander"],
        "Asia Pacific": ["MUFG Bank", "Bank of China", "Commonwealth Bank", "DBS Bank"],
        "Middle East": ["Emirates NBD", "Qatar National Bank", "National Bank of Kuwait"],
        "Africa": ["Standard Bank", "Ecobank", "Commercial Bank of Ethiopia"],
        "Latin America": ["Banco do Brasil", "Banco de Chile", "Bancolombia"],
        "Caribbean": ["National Commercial Bank Jamaica", "Republic Bank", "Scotia Bank"],
        "Central Asia": ["Kazkommertsbank", "Bank of Mongolia", "Halyk Bank"],
        "South Asia": ["State Bank of India", "ICICI Bank", "Habib Bank"],
        "Southeast Asia": ["Bangkok Bank", "Bank Rakyat Indonesia", "Maybank"]
    }
    
    # Transaction types
    transaction_types = ["Wire Transfer", "SWIFT", "ACH", "FX Settlement", "Trade Finance"]
    
    # Generate random flows
    rows = []
    
    # Generate some random flows between regions
    np.random.seed(42)  # For reproducibility
    
    for _ in range(200):
        # Select random source and target regions
        source_region = np.random.choice(regions)
        # Ensure target region is different from source
        target_regions = [r for r in regions if r != source_region]
        target_region = np.random.choice(target_regions)
        
        # Select random banks
        source_bank = np.random.choice(banks[source_region])
        target_bank = np.random.choice(banks[target_region])
        
        # Select random transaction type
        txn_type = np.random.choice(transaction_types)
        
        # Generate random amount (with some regions having larger flows)
        base_amount = np.random.exponential(100) * 10000
        if source_region in ["North America", "Europe", "Asia Pacific"]:
            base_amount *= 3
        
        # Add some randomness to the amount
        amount = base_amount * (0.5 + np.random.random())
        
        # Create a data row
        row = {
            "send region": source_region,
            "receiver region": target_region,
            "send bank": source_bank,
            "receiver bank": target_bank,
            "transaction_type": txn_type,
            "amount": amount
        }
        
        rows.append(row)
    
    # Create the DataFrame
    df = pd.DataFrame(rows)
    
    return df

def main():
    """
    Main function to run the interactive visualization
    """
    # Create sample data (replace this with your actual data loading)
    df = create_sample_data()
    
    # Create and run the Dash app
    app = create_interactive_dashboard(df)
    
    print("Starting the interactive dashboard...")
    print("Access the visualization at http://127.0.0.1:8050/ after the server starts")
    
    # Run the server - fixed to use app.run() instead of app.run_server()
    app.run(debug=True)

# Entry point
if __name__ == "__main__":
    main()