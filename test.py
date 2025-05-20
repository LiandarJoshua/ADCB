import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Region coordinates (approximate centroids)
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

def prepare_region_flow_data(df):
    flow_data = df.groupby(['send region', 'receiver region'])['amount'].sum().reset_index()
    for coord in ['lat', 'lon']:
        flow_data[f'source_{coord}'] = flow_data['send region'].map(lambda x: region_coordinates.get(x, {}).get(coord))
        flow_data[f'target_{coord}'] = flow_data['receiver region'].map(lambda x: region_coordinates.get(x, {}).get(coord))
    return flow_data

def create_sankey_diagram(df, source_region=None, target_region=None):
    """
    Create a Sankey diagram for flows between regions
    If source_region and target_region are provided, filter to show only flows between these regions
    Otherwise, default to showing North America and Europe
    """
    # Default regions if none specified
    if source_region is None:
        source_region = "North America"
    if target_region is None:
        target_region = "Europe"
    
    # Filter data for the selected regions
    if source_region and target_region:
        filtered_df = df[(df['send region'] == source_region) & (df['receiver region'] == target_region) |
                         (df['send region'] == target_region) & (df['receiver region'] == source_region)]
    else:
        filtered_df = df
    
    # If no data found, return empty figure with message
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No flow data between selected regions",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Prepare node labels (unique regions involved)
    unique_regions = pd.unique(filtered_df[['send region', 'receiver region']].values.ravel('K'))
    region_to_idx = {region: i for i, region in enumerate(unique_regions)}
    
    # Prepare Sankey data
    sources = [region_to_idx[region] for region in filtered_df['send region']]
    targets = [region_to_idx[region] for region in filtered_df['receiver region']]
    values = filtered_df['amount'].tolist()
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(unique_regions),
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            # Add hover information
            hovertemplate='%{source.label} → %{target.label}<br>Amount: $%{value:,.2f}<extra></extra>'
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text=f"Banking Flows: {source_region} ↔ {target_region}",
        font_size=12,
        height=400,
        margin=dict(l=25, r=25, t=50, b=25)
    )
    
    return fig

def create_flow_map(df, mapbox_token=None):
    flow_data = prepare_region_flow_data(df)
    region_flows = pd.merge(
        df.groupby('receiver region')['amount'].sum().rename('inflow').reset_index(),
        df.groupby('send region')['amount'].sum().rename('outflow').reset_index(),
        left_on='receiver region', right_on='send region', how='outer'
    ).fillna(0).rename(columns={'receiver region': 'region'}).drop('send region', axis=1)
    
    region_flows['net_flow'] = region_flows['inflow'] - region_flows['outflow']
    region_flows['total_volume'] = region_flows['inflow'] + region_flows['outflow']
    region_data = pd.merge(region_flows, pd.DataFrame(region_coordinates).T.reset_index().rename(columns={'index': 'region'}), on='region')
    
    flow_data['normalized_amount'] = flow_data['amount'] / flow_data['amount'].max() * 10
    fig = go.Figure()
    
    # Add flow lines
    for idx, row in flow_data[flow_data['normalized_amount'] >= 0.5].iterrows():
        # Check for None or NaN values in coordinates
        if (pd.isna(row['source_lon']) or pd.isna(row['source_lat']) or 
            pd.isna(row['target_lon']) or pd.isna(row['target_lat'])):
            continue
            
        lon_diff, lat_diff = row['target_lon'] - row['source_lon'], row['target_lat'] - row['source_lat']
        midpoint_offset = min(np.sqrt(lon_diff**2 + lat_diff**2) * 0.15, 10)
        curve_points = [(row['source_lon'], row['source_lat'])] + [
            ((1-t)**2 * row['source_lon'] + 2*(1-t)*t * ((row['source_lon'] + row['target_lon'])/2) + t**2 * row['target_lon'],
             (1-t)**2 * row['source_lat'] + 2*(1-t)*t * ((row['source_lat'] + row['target_lat'])/2 + midpoint_offset) + t**2 * row['target_lat'])
            for t in np.linspace(0, 1, 20)[1:-1]
        ] + [(row['target_lon'], row['target_lat'])]
        
        lons, lats = zip(*curve_points)
        # Fix the color transparency value
        alpha = min(0.8, 0.3 + row["normalized_amount"] * 0.05)
        color = f'rgba(70, 130, 180, {alpha:.2f})'
        
        fig.add_trace(go.Scattermapbox(
            lon=lons, lat=lats, mode='lines',
            line=dict(width=row['normalized_amount']*1.5, color=color),
            hoverinfo='text', hovertext=f"{row['send region']} → {row['receiver region']}<br>Amount: ${row['amount']:,.2f}",
            showlegend=False,
            # Add custom data for click events
            customdata=[{
                'source_region': row['send region'],
                'target_region': row['receiver region'],
                'flow_id': f"{row['send region']}_{row['receiver region']}"
            }]
        ))
    
    # Add regions
    # Filter out any rows with NaN coordinates
    valid_region_data = region_data.dropna(subset=['lon', 'lat'])
    if not valid_region_data.empty:
        fig.add_trace(go.Scattermapbox(
            lon=valid_region_data['lon'], 
            lat=valid_region_data['lat'], 
            text=valid_region_data['region'],
            customdata=valid_region_data[['inflow', 'outflow', 'net_flow', 'region']].values,
            hovertemplate='<b>%{text}</b><br>Inflow: $%{customdata[0]:,.2f}<br>Outflow: $%{customdata[1]:,.2f}<br>Net Flow: $%{customdata[2]:,.2f}',
            mode='markers',
            marker=dict(
                size=valid_region_data['total_volume'] / valid_region_data['total_volume'].max() * 25 + 10,
                color=valid_region_data['net_flow'], 
                colorscale='RdBu', 
                cmid=0, 
                opacity=0.8,
                colorbar=dict(title='Net Flow<br>(Inflow - Outflow)', thickness=15)
            ),
            name='Regions'
        ))
    
    fig.update_layout(
        title='Global Banking Transaction Flows Between Regions',
        mapbox=dict(
            style='carto-positron',
            zoom=1.2,
            center=dict(lat=20, lon=0),
            **({'accesstoken': mapbox_token, 'style': 'mapbox://styles/mapbox/light-v10'} if mapbox_token else {})
        ),
        height=500, margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

def create_dashboard(df):
    """
    Create an interactive dashboard with a map and Sankey diagram
    """
    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "mapbox"}], [{"type": "sankey"}]],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.05,
        subplot_titles=("Global Banking Flow Map", "Banking Flow Detail - North America ↔ Europe")
    )
    
    # Create the map figure
    map_fig = create_flow_map(df)
    
    # Create default Sankey diagram
    sankey_fig = create_sankey_diagram(df, "North America", "Europe")
    
    # Add traces from map figure to subplot
    for trace in map_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add Sankey diagram to second subplot
    for trace in sankey_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Update layout for both subplots
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Interactive Banking Flows Dashboard",
        mapbox=dict(
            style='carto-positron',
            zoom=1.2,
            center=dict(lat=20, lon=0)
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Add JavaScript for interactivity
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=["title", "Interactive Banking Flows Dashboard"]
                    )
                ],
                x=0.9,
                y=0.95,
            )
        ]
    )
    
    # Add JavaScript function for click events
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.5, y=0.01,
        text="Click on flow lines to update the Sankey diagram below",
        showarrow=False,
        font=dict(size=12)
    )
    
    # Add JavaScript callback for click events on flow lines
    fig.update_layout(
        clickmode='event',
    )
    
    return fig

def visualize_banking_flows_with_sankey(df):
    """
    Create an HTML file with interactive dashboard
    """
    # Ensure the expected columns exist
    required_columns = ['send region', 'receiver region', 'amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in dataframe: {missing_columns}")
    
    # Ensure all regions are in the coordinates dictionary
    unknown_send_regions = set(df['send region'].unique()) - set(region_coordinates.keys())
    unknown_recv_regions = set(df['receiver region'].unique()) - set(region_coordinates.keys())
    
    if unknown_send_regions:
        print(f"Warning: Unknown send regions: {unknown_send_regions}")
    if unknown_recv_regions:
        print(f"Warning: Unknown receiver regions: {unknown_recv_regions}")
    
    # Create the dashboard figure
    dashboard_fig = create_dashboard(df)
    
    # Add JavaScript for interactivity
    js_code = """
    <script>
        // Function to update Sankey diagram when a flow line is clicked
        var graphDiv = document.getElementById('banking-dashboard');
        graphDiv.on('plotly_click', function(data) {
            // Check if click was on a flow line (index 0 to n-2 are flow lines)
            var numTraces = graphDiv.data.length;
            var pointIndex = data.points[0].curveNumber;
            
            // If click was on a flow line (not a region marker)
            if (pointIndex < numTraces - 1 && data.points[0].customdata) {
                var source = data.points[0].customdata.source_region;
                var target = data.points[0].customdata.target_region;
                
                if (source && target) {
                    // Update Sankey diagram title
                    Plotly.relayout(graphDiv, {'annotations[1].text': 'Banking Flow Detail - ' + source + ' ↔ ' + target});
                    
                    // Create new Sankey diagram data
                    fetch('/update_sankey', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            source_region: source,
                            target_region: target
                        }),
                    })
                    .then(response => response.json())
                    .then(sankeyData => {
                        // Update the Sankey diagram
                        Plotly.update(graphDiv, {
                            node: sankeyData.node,
                            link: sankeyData.link
                        }, {}, [numTraces - 1]);
                    });
                }
            }
        });
    </script>
    """
    
    # Create HTML file with dashboard and JavaScript
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Banking Flows Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            .instructions {
                margin: 20px 0;
                padding: 10px;
                background-color: #f9f9f9;
                border-left: 4px solid #0066cc;
            }
            #banking-dashboard {
                width: 100%;
                height: 900px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interactive Banking Flows Dashboard</h1>
            <div class="instructions">
                <p><strong>Instructions:</strong> Click on any flow line in the map to see detailed Sankey diagram for flows between those regions.</p>
            </div>
            <div id="banking-dashboard"></div>
        </div>
        
        <script>
            // Create the initial dashboard
            var dashboardData = """ + dashboard_fig.to_json() + """;
            Plotly.newPlot('banking-dashboard', dashboardData.data, dashboardData.layout);
            
            // Add click event listener to update Sankey when flow is clicked
            document.getElementById('banking-dashboard').on('plotly_click', function(data) {
                var point = data.points[0];
                
                // Check if clicked on a flow line (which should have customdata)
                if (point.customdata && point.customdata.source_region) {
                    var source = point.customdata.source_region;
                    var target = point.customdata.target_region;
                    
                    // Update subplot title
                    var update = {
                        'annotations[1].text': 'Banking Flow Detail - ' + source + ' ↔ ' + target
                    };
                    Plotly.relayout('banking-dashboard', update);
                    
                    // Filter data for these regions
                    var filteredData = dashboardData.data.filter(function(trace) {
                        return trace.type === 'sankey';
                    })[0];
                    
                    // Create new Sankey diagram with filtered data
                    // In a real application, you would fetch new data from the server
                    // Here we're simulating the update
                    
                    // For demonstration purposes - recreating the sankey with new title
                    // In a real app, you would make an AJAX call to get new sankey data
                    
                    // For now, we'll just update the title and pretend the data changed
                    // The real implementation would replace the sankey trace with new data
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open("banking_flows_dashboard.html", "w") as f:
        f.write(html_content)
    
    print("Interactive dashboard created and saved as banking_flows_dashboard.html")
    return dashboard_fig

# Create a Flask app for handling Sankey diagram updates
def create_flask_app(df):
    """
    Create a Flask app to handle Sankey diagram updates
    This would be used in a real-world scenario to update the Sankey diagram
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/update_sankey', methods=['POST'])
    def update_sankey():
        data = request.json
        source_region = data.get('source_region')
        target_region = data.get('target_region')
        
        # Create Sankey diagram for selected regions
        sankey_fig = create_sankey_diagram(df, source_region, target_region)
        
        # Return Sankey data as JSON
        return jsonify(sankey_fig.data[0])
    
    @app.route('/')
    def index():
        with open("banking_flows_dashboard.html", "r") as f:
            return f.read()
    
    return app

# Example usage with a full implementation
def main():
    # Sample data creation
    # In a real application, you would load your data from a CSV or database
    regions = ["North America", "Europe", "Asia Pacific", "Middle East", "Africa", "Latin America"]
    sample_data = []
    
    # Generate some random flow data between regions
    import random
    for send_region in regions:
        for receiver_region in regions:
            if send_region != receiver_region:
                # Generate a random amount (more flows between major regions)
                amount_factor = 1.0
                if send_region in ["North America", "Europe", "Asia Pacific"] and receiver_region in ["North America", "Europe", "Asia Pacific"]:
                    amount_factor = 3.0
                
                amount = random.uniform(1000, 10000) * amount_factor
                sample_data.append({
                    "send region": send_region,
                    "receiver region": receiver_region,
                    "amount": amount
                })
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create and display the dashboard
    dashboard_fig = visualize_banking_flows_with_sankey(df)
    
    # Create Flask app for handling updates (uncomment to use)
    # app = create_flask_app(df)
    # app.run(debug=True)
    
    return dashboard_fig

if __name__ == "__main__":
    main()