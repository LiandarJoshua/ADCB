import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Banking Flows Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

@st.cache_data
def load_data(file_path):
    """Load and cache the transaction data"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def plot_transaction_time_series(df, value_column='amount', time_column='year', 
                                group_by='send bank', aggfunc='sum', 
                                title='Transaction Trends Over Time by Bank'):
    """Create a time series plot for transaction trends"""
    try:
        # Group and aggregate
        grouped = df.groupby([time_column, group_by])[value_column].agg(aggfunc).reset_index()
        grouped.rename(columns={value_column: 'value'}, inplace=True)

        fig = px.line(
            grouped,
            x=time_column,
            y='value',
            color=group_by,
            markers=True,
            title=title,
            labels={
                time_column: time_column.replace('_', ' ').title(),
                'value': value_column.title(),
                group_by: group_by.replace('_', ' ').title()
            }
        )

        fig.update_layout(
            xaxis=dict(dtick=1),
            yaxis_title=value_column.title(),
            legend_title=group_by.replace('_', ' ').title(),
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating time series plot: {e}")
        return None

def plot_transaction_heatmap(df, value_column='amount', aggfunc='sum', 
                           title='Heatmap of Transaction Volumes Between Regions'):
    """Create a heatmap showing transaction volumes between regions"""
    try:
        # Create pivot table
        matrix = df.pivot_table(
            index='send region',
            columns='receiver region',
            values=value_column,
            aggfunc=aggfunc,
            fill_value=0
        )

        # Generate heatmap
        fig = px.imshow(
            matrix,
            labels=dict(x="Receiver Region", y="Sender Region", color=value_column.title()),
            x=matrix.columns,
            y=matrix.index,
            color_continuous_scale='Viridis',
            text_auto=True
        )

        fig.update_layout(
            title=title,
            xaxis_title="Receiver Region",
            yaxis_title="Sender Region",
            height=600
        )

        return fig
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        return None

def create_sankey_for_regions(df, send_region=None, receive_region=None, year=None):
    """
    Create an enhanced Sankey diagram for transactions between specified regions,
    with hover animations and flow animations for the links.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing transaction data
    send_region : str, optional
        Region to filter senders by
    receive_region : str, optional
        Region to filter receivers by
    year : int, optional
        Year to filter transactions by
    
    Returns:
    --------
    plotly.graph_objects.Figure
        A Sankey diagram figure with enhanced animations
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    
    # Create a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Filter data if regions are specified
    title_parts = []
    
    if send_region:
        filtered_df = filtered_df[filtered_df['send region'] == send_region]
        title_parts.append(f"from {send_region}")
    
    if receive_region:
        filtered_df = filtered_df[filtered_df['receiver region'] == receive_region]
        title_parts.append(f"to {receive_region}")
    
    # Filter by year if specified
    if year and 'year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['year'] == year]
        title_parts.append(f"in {year}")
    
    if title_parts:
        title = f"Transactions {' '.join(title_parts)}"
    else:
        title = "All Regional Transactions"
    
    # Ensure amount categories exist (create if missing)
    if 'amount_category' not in filtered_df.columns:
        filtered_df['amount_category'] = pd.cut(
            filtered_df['amount'], 
            bins=[0, 1e6, 5e6, 1e7, float('inf')], 
            labels=['< 1M', '1M-5M', '5M-10M', '> 10M']
        )
    
    # Check if we have data after all the filters
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text=f"No data available for {title}",
            annotations=[dict(
                text="No transactions found for the selected criteria",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )]
        )
        return fig
    
    # Make sure all columns are properly formatted as strings
    str_columns = ['sender', 'receiver', 'currency', 'mx/mt', 'Mt', 'direction', 
                  'transaction_status', 'payment_method', 'amount_category']
    for col in str_columns:
        if col in filtered_df.columns and filtered_df[col].dtype != 'str':
            filtered_df[col] = filtered_df[col].astype(str)
    
    # Create the flows according to the specified path
    # sender → receiver
    flow1 = filtered_df.groupby(['sender', 'receiver']).size().reset_index(name='count')
    
    # receiver → currency
    flow2 = filtered_df.groupby(['receiver', 'currency']).size().reset_index(name='count')
    
    # currency → mx/mt
    flow3 = filtered_df.groupby(['currency', 'mx/mt']).size().reset_index(name='count')
    
    # mx/mt → mt
    flow4 = filtered_df.groupby(['mx/mt', 'Mt']).size().reset_index(name='count')
    
    # mt → direction
    flow5 = filtered_df.groupby(['Mt', 'direction']).size().reset_index(name='count')
    
    # direction → transaction_status (new flow)
    flow6 = filtered_df.groupby(['direction', 'transaction_status']).size().reset_index(name='count')
    
    # transaction_status → payment_method (new flow)
    flow7 = filtered_df.groupby(['transaction_status', 'payment_method']).size().reset_index(name='count')
    
    # payment_method → amount (new flow)
    flow8 = filtered_df.groupby(['payment_method', 'amount_category']).size().reset_index(name='count')
    
    # Get all unique labels
    labels = pd.unique(
        flow1['sender'].tolist() +
        flow1['receiver'].tolist() +
        flow2['currency'].tolist() +
        flow3['mx/mt'].tolist() +
        flow4['Mt'].tolist() +
        flow5['direction'].tolist() +
        flow6['transaction_status'].tolist() +
        flow7['payment_method'].tolist() +
        flow8['amount_category'].astype(str).tolist()
    )
    
    # Map labels to indices
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Prepare sources, targets, and values for links
    source, target, value = [], [], []
    link_labels = []
    
    # sender → receiver
    for _, row in flow1.iterrows():
        source.append(label_to_index[row['sender']])
        target.append(label_to_index[row['receiver']])
        value.append(row['count'])
        link_labels.append(f"{row['sender']} → {row['receiver']}<br>Count: {row['count']}")
    
    # receiver → currency
    for _, row in flow2.iterrows():
        source.append(label_to_index[row['receiver']])
        target.append(label_to_index[row['currency']])
        value.append(row['count'])
        link_labels.append(f"{row['receiver']} → {row['currency']}<br>Count: {row['count']}")
    
    # currency → mx/mt
    for _, row in flow3.iterrows():
        source.append(label_to_index[row['currency']])
        target.append(label_to_index[row['mx/mt']])
        value.append(row['count'])
        link_labels.append(f"{row['currency']} → {row['mx/mt']}<br>Count: {row['count']}")
    
    # mx/mt → mt
    for _, row in flow4.iterrows():
        source.append(label_to_index[row['mx/mt']])
        target.append(label_to_index[row['Mt']])
        value.append(row['count'])
        link_labels.append(f"{row['mx/mt']} → {row['Mt']}<br>Count: {row['count']}")
    
    # mt → direction
    for _, row in flow5.iterrows():
        source.append(label_to_index[row['Mt']])
        target.append(label_to_index[row['direction']])
        value.append(row['count'])
        link_labels.append(f"{row['Mt']} → {row['direction']}<br>Count: {row['count']}")
    
    # direction → transaction_status (new flow)
    for _, row in flow6.iterrows():
        source.append(label_to_index[row['direction']])
        target.append(label_to_index[row['transaction_status']])
        value.append(row['count'])
        link_labels.append(f"{row['direction']} → {row['transaction_status']}<br>Count: {row['count']}")
    
    # transaction_status → payment_method (new flow)
    for _, row in flow7.iterrows():
        source.append(label_to_index[row['transaction_status']])
        target.append(label_to_index[row['payment_method']])
        value.append(row['count'])
        link_labels.append(f"{row['transaction_status']} → {row['payment_method']}<br>Count: {row['count']}")
    
    # payment_method → amount_category (new flow)
    for _, row in flow8.iterrows():
        source.append(label_to_index[row['payment_method']])
        target.append(label_to_index[str(row['amount_category'])])
        value.append(row['count'])
        link_labels.append(f"{row['payment_method']} → {row['amount_category']}<br>Count: {row['count']}")
    
    # Set up colors for nodes and links
    num_nodes = len(labels)
    node_colors = ["rgba(31, 119, 180, 0.8)"] * num_nodes  # Default blue color
    
    # Create color groups for different node types
    sender_idx    = [label_to_index[label] for label in pd.Series(flow1['sender']).unique()]
    receiver_idx  = [label_to_index[label] for label in pd.Series(flow1['receiver']).unique()]
    currency_idx  = [label_to_index[label] for label in pd.Series(flow2['currency']).unique()]
    mxmt_idx      = [label_to_index[label] for label in pd.Series(flow3['mx/mt']).unique()]
    mt_idx        = [label_to_index[label] for label in pd.Series(flow4['Mt']).unique()]
    direction_idx = [label_to_index[label] for label in pd.Series(flow5['direction']).unique()]
    status_idx    = [label_to_index[label] for label in pd.Series(flow6['transaction_status']).unique()]
    payment_idx   = [label_to_index[label] for label in pd.Series(flow7['payment_method']).unique()]
    amount_idx    = [label_to_index[str(label)] for label in pd.Series(flow8['amount_category']).unique()]

    
    # Assign different colors to each node type
    for idx in sender_idx:
        node_colors[idx] = "rgba(214, 39, 40, 0.8)"  # Red for senders
    for idx in receiver_idx:
        node_colors[idx] = "rgba(44, 160, 44, 0.8)"  # Green for receivers
    for idx in currency_idx:
        node_colors[idx] = "rgba(255, 127, 14, 0.8)"  # Orange for currency
    for idx in mxmt_idx:
        node_colors[idx] = "rgba(148, 103, 189, 0.8)"  # Purple for mx/mt
    for idx in mt_idx:
        node_colors[idx] = "rgba(140, 86, 75, 0.8)"   # Brown for Mt
    for idx in direction_idx:
        node_colors[idx] = "rgba(23, 190, 207, 0.8)"  # Cyan for directions
    for idx in status_idx:
        node_colors[idx] = "rgba(188, 189, 34, 0.8)"  # Yellow-green for transaction status
    for idx in payment_idx:
        node_colors[idx] = "rgba(127, 127, 127, 0.8)"  # Gray for payment method
    for idx in amount_idx:
        node_colors[idx] = "rgba(31, 119, 180, 0.8)"  # Blue for amount categories
    
    # Create base and hover colors for links based on their source
    link_colors = []
    link_hover_colors = []
    
    for s in source:
        # Normal state: semi-transparent
        base_color = node_colors[s].replace("0.8", "0.4")  # More transparent for normal state
        # Hover state: more opaque and brighter
        hover_color = node_colors[s].replace("0.8", "0.9")  # More opaque for hover state
        
        link_colors.append(base_color)
        link_hover_colors.append(hover_color)
    
    # Create the Sankey diagram with enhanced styling
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(labels),
            color=node_colors,
            # Add hover effect to nodes
            hoverinfo="all",
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            # Add animated hover effect for links
            customdata=np.array(link_labels),
            hovertemplate='%{customdata}<extra></extra>',
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
    )])
    
    # Add flow animation using frames
    # We'll create frames to animate link opacity to simulate flow
    frames = []
    
    # Create several frames with different link opacities to simulate flow
    for i in range(10):
        # Generate a flow pattern that moves from left to right
        phase = i / 10.0
        flow_colors = []
        
        for s, t in zip(source, target):
            # Calculate distance from left to right (normalized)
            pos = (s + t) / (2 * num_nodes)
            
            # Create a wave pattern that moves across the links
            intensity = 0.5 + 0.4 * np.sin(2 * np.pi * (pos - phase))
            
            # Apply the intensity to the base color
            base_color = node_colors[s].replace("0.8", str(intensity))
            flow_colors.append(base_color)
        
        # Create a frame with the current flow pattern
        frames.append(go.Frame(
            data=[go.Sankey(
                link=dict(color=flow_colors)
            )],
            name=f"frame{i}"
        ))
    
    # Add frames to the figure
    fig.frames = frames
    
    # Add play button for animation
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play Flow",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                                transition=dict(duration=200)
                            )
                        ]
                    )
                ],
                x=0.1,
                y=1,
            )
        ]
    )
    
    # Add annotations to describe each column
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=800,  # Increased height to accommodate more nodes
        width=1200,  # Increased width to accommodate more columns
        # Add hover mode settings to improve interactivity
        hovermode="closest",
        hoverdistance=10,
        # Add transition settings for smoother animations
        transition=dict(
            duration=1000,
            easing="cubic-in-out"
        ),
        annotations=[
            dict(x=0.02, y=1, xref='paper', yref='paper', text='Sender', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.14, y=1, xref='paper', yref='paper', text='Receiver', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.25, y=1, xref='paper', yref='paper', text='Currency', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.37, y=1, xref='paper', yref='paper', text='MX/MT', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.48, y=1, xref='paper', yref='paper', text='MT', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.60, y=1, xref='paper', yref='paper', text='Direction', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.71, y=1, xref='paper', yref='paper', text='Status', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.83, y=1, xref='paper', yref='paper', text='Payment', showarrow=False, font=dict(size=14, color='black')),
            dict(x=0.94, y=1, xref='paper', yref='paper', text='Amount', showarrow=False, font=dict(size=14, color='black')),
        ]
    )
    
    # Add custom CSS for hover effects
    fig.update_layout(
        hoverlabel_bgcolor="white",
        hoverlabel_font_size=14,
        hoverlabel_font_family="Arial"
    )
    
    # Add configuration options for better interactivity
    fig.update_layout(
        dragmode="pan",
        clickmode="event+select"
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

def main():
    # App title and description
    st.title("🏦 Banking Flows Analysis Dashboard")
    st.markdown("Analyze transaction flows between regions with interactive visualizations")
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filters")
    
    # Default file path
    default_path = "app/simulated_transactions.csv"

    
    # Load data from default path
    df = None
    try:
        df = pd.read_csv(default_path)
        st.sidebar.success("✅ Data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {e}")
        st.error(f"Could not load data from: {default_path}")
        st.stop()
    
    # Display basic info about the dataset
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Rows:** {len(df):,}")
    st.sidebar.write(f"**Columns:** {len(df.columns)}")
    
    # Data preprocessing
    if 'year' not in df.columns and 'date' in df.columns:
        try:
            df['year'] = pd.to_datetime(df['date']).dt.year
        except:
            df['year'] = 2023
    elif 'year' not in df.columns:
        df['year'] = 2023
    
    # Ensure amount categories exist
    if 'amount_category' not in df.columns and 'amount' in df.columns:
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=[0, 1e6, 5e6, 1e7, float('inf')], 
            labels=['< 1M', '1M-5M', '5M-10M', '> 10M']
        )
    
    # Get unique values for filters
    regions = ['All'] + sorted(list(set(df['send region'].unique()).union(set(df['receiver region'].unique()))))
    years = ['All'] + sorted(df['year'].unique()) if 'year' in df.columns else ['All']
    
    # Region filters
    send_region = st.sidebar.selectbox("Sender Region", regions)
    receive_region = st.sidebar.selectbox("Receiver Region", regions)
    year_filter = st.sidebar.selectbox("Year", years)
    
    # Apply filters
    filtered_df = df.copy()
    
    if send_region != 'All':
        filtered_df = filtered_df[filtered_df['send region'] == send_region]
    
    if receive_region != 'All':
        filtered_df = filtered_df[filtered_df['receiver region'] == receive_region]
    
    if year_filter != 'All':
        filtered_df = filtered_df[filtered_df['year'] == year_filter]
    
    # Display filtered data info
    st.sidebar.write(f"**Filtered rows:** {len(filtered_df):,}")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Flow Map", "🔄 Sankey Diagram", "🔥 Heatmap", "📈 Time Series"])
    
    with tab1:
        st.subheader("Global Transaction Flow Map")
        if len(filtered_df) > 0:
            flow_map = create_flow_map(filtered_df,True)
            if flow_map:
                st.plotly_chart(flow_map, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab2:
        st.subheader("Transaction Flow Sankey Diagram")
        if len(filtered_df) > 0:
            sankey_fig = create_sankey_for_regions(
                filtered_df, 
                send_region if send_region != 'All' else None,
                receive_region if receive_region != 'All' else None,
                year_filter if year_filter != 'All' else None
            )
            if sankey_fig:
                st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab3:
        st.subheader("Regional Transaction Heatmap")
        if len(filtered_df) > 0:
            heatmap_fig = plot_transaction_heatmap(filtered_df)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab4:
        st.subheader("Transaction Trends Over Time")
        if len(filtered_df) > 0 and 'year' in filtered_df.columns:
            # Time series options
            col1, col2 = st.columns(2)
            with col1:
                value_col = st.selectbox("Value Column", ['amount'] + [col for col in filtered_df.columns if col != 'amount'])
            with col2:
                group_by = st.selectbox("Group By", ['send region', 'receiver region', 'send bank', 'receiver bank'] if 'send bank' in filtered_df.columns else ['send region', 'receiver region'])
            
            time_series_fig = plot_transaction_time_series(
                filtered_df,
                value_column=value_col,
                time_column='year',
                group_by=group_by
            )
            if time_series_fig:
                st.plotly_chart(time_series_fig, use_container_width=True)
        else:
            st.warning("No data available for time series analysis.")
    
    # Summary statistics
    if len(filtered_df) > 0:
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(filtered_df):,}")
        
        with col2:
            if 'amount' in filtered_df.columns:
                total_amount = filtered_df['amount'].sum()
                st.metric("Total Amount", f"${total_amount:,.2f}")
        
        with col3:
            unique_regions = len(set(filtered_df['send region'].unique()).union(set(filtered_df['receiver region'].unique())))
            st.metric("Unique Regions", unique_regions)
        
        with col4:
            if 'amount' in filtered_df.columns:
                avg_amount = filtered_df['amount'].mean()
                st.metric("Average Amount", f"${avg_amount:,.2f}")



if __name__ == "__main__":
    main()