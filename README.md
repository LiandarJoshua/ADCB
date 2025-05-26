Liandar Joshua
Kaggle Notebook Link: https://www.kaggle.com/code/liandarjoshua/transactiondashboard


Steamlit App: https://transactiondashboardsample.streamlit.app/
=======
# Banking Flows Analysis Dashboard - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Function Reference](#function-reference)
3. [Data Structures](#data-structures)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Performance Considerations](#performance-considerations)
7. [Development Guidelines](#development-guidelines)

## Architecture Overview

### Application Structure
```
banking_flows_dashboard.py
├── Configuration & Setup
├── Data Loading & Caching
├── Visualization Functions
│   ├── Time Series Analysis
│   ├── Heatmap Generation
│   ├── Sankey Diagrams
│   └── Geographical Flow Maps
├── Main Application Logic
└── Streamlit Interface
```

### Technology Stack
- **Frontend**: Streamlit (Web UI)
- **Visualization**: Plotly (Interactive charts)
- **Data Processing**: Pandas (Data manipulation)
- **Numerical Computing**: NumPy (Mathematical operations)

### Data Flow
1. CSV data loaded and cached
2. User applies filters via sidebar
3. Data processed and aggregated
4. Visualizations generated dynamically
5. Interactive results displayed in tabs

## Function Reference

### Data Loading Functions

#### `load_data(file_path: str) -> pd.DataFrame`
Loads and caches transaction data from CSV file.

**Parameters:**
- `file_path` (str): Path to CSV file containing transaction data

**Returns:**
- `pd.DataFrame`: Loaded transaction data or None if error

**Features:**
- Uses `@st.cache_data` for performance optimization
- Comprehensive error handling
- Automatic data validation

**Example:**
```python
df = load_data("data/transactions.csv")
if df is not None:
    print(f"Loaded {len(df)} transactions")
```

### Visualization Functions

#### `plot_transaction_time_series(df, value_column='amount', time_column='year', group_by='send bank', aggfunc='sum', title='Transaction Trends Over Time by Bank') -> plotly.graph_objects.Figure`

Creates interactive time series visualizations for transaction trends.

**Parameters:**
- `df` (pd.DataFrame): Transaction data
- `value_column` (str): Column to aggregate (default: 'amount')
- `time_column` (str): Temporal column for x-axis (default: 'year')
- `group_by` (str): Grouping column for series (default: 'send bank')
- `aggfunc` (str): Aggregation function (default: 'sum')
- `title` (str): Chart title

**Returns:**
- `plotly.graph_objects.Figure`: Interactive line chart

**Features:**
- Multiple series support
- Customizable aggregation
- Interactive hover information
- Responsive design

#### `plot_transaction_heatmap(df, value_column='amount', aggfunc='sum', title='Heatmap of Transaction Volumes Between Regions') -> plotly.graph_objects.Figure`

Generates heatmap visualization of transaction volumes between regions.

**Parameters:**
- `df` (pd.DataFrame): Transaction data
- `value_column` (str): Column to aggregate
- `aggfunc` (str): Aggregation function
- `title` (str): Chart title

**Returns:**
- `plotly.graph_objects.Figure`: Interactive heatmap

**Features:**
- Pivot table generation
- Color scale customization
- Text annotations on cells
- Interactive hover details

#### `create_sankey_for_regions(df, send_region=None, receive_region=None, year=None) -> plotly.graph_objects.Figure`

Creates animated Sankey diagram showing multi-stage transaction flows.

**Parameters:**
- `df` (pd.DataFrame): Transaction data
- `send_region` (str, optional): Filter by sender region
- `receive_region` (str, optional): Filter by receiver region
- `year` (int, optional): Filter by year

**Returns:**
- `plotly.graph_objects.Figure`: Interactive Sankey diagram with animation

**Flow Stages:**
1. Sender → Receiver
2. Receiver → Currency
3. Currency → MX/MT
4. MX/MT → MT
5. MT → Direction
6. Direction → Transaction Status
7. Transaction Status → Payment Method
8. Payment Method → Amount Category

**Features:**
- Multi-stage flow visualization
- Animated flow effects
- Color-coded node categories
- Interactive hover information
- Custom link animations

#### `create_flow_map(df, animate=False) -> plotly.graph_objects.Figure`

Generates geographical visualization of transaction flows between regions.

**Parameters:**
- `df` (pd.DataFrame): Transaction data
- `animate` (bool): Enable flow animation (default: False)

**Returns:**
- `plotly.graph_objects.Figure`: Interactive geographical map

**Features:**
- Curved flow lines
- Line thickness proportional to volume
- Color coding by amount
- Regional markers
- Optional flow animation
- Natural Earth projection

## Data Structures

### Required DataFrame Schema

#### Mandatory Columns
```python
{
    'send region': str,      # Source region
    'receiver region': str,  # Destination region
    'amount': float,         # Transaction amount
}
```

#### Optional Columns
```python
{
    'sender': str,           # Individual sender ID
    'receiver': str,         # Individual receiver ID
    'currency': str,         # Transaction currency
    'mx/mt': str,           # Message exchange/type
    'Mt': str,              # Message type classification
    'direction': str,        # Transaction direction
    'transaction_status': str, # Transaction status
    'payment_method': str,   # Payment method
    'year': int,            # Year (auto-generated from date)
    'date': str,            # Transaction date
    'send bank': str,       # Sending bank
    'receiver bank': str,   # Receiving bank
}
```

### Processed Data Enhancements

#### Auto-Generated Columns
- `amount_category`: Categorical breakdown of amounts
  - `< 1M`: Less than $1,000,000
  - `1M-5M`: $1,000,000 - $5,000,000
  - `5M-10M`: $5,000,000 - $10,000,000
  - `> 10M`: Greater than $10,000,000

#### Data Transformations
- Date to year extraction
- String normalization
- Null value handling
- Coordinate mapping for regions

## Configuration

### Region Coordinates Dictionary
```python
region_coordinates = {
    'Region Name': (latitude, longitude),
    # Example:
    'North America': (40.0, -100.0),
    'Europe': (50.0, 10.0),
    # ... additional regions
}
```

### Streamlit Configuration
```python
st.set_page_config(
    page_title="Banking Flows Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Color Schemes

#### Node Colors (Sankey)
- Senders: `rgba(214, 39, 40, 0.8)` (Red)
- Receivers: `rgba(44, 160, 44, 0.8)` (Green)
- Currency: `rgba(255, 127, 14, 0.8)` (Orange)
- MX/MT: `rgba(148, 103, 189, 0.8)` (Purple)
- MT: `rgba(140, 86, 75, 0.8)` (Brown)
- Direction: `rgba(23, 190, 207, 0.8)` (Cyan)
- Status: `rgba(188, 189, 34, 0.8)` (Yellow-green)
- Payment: `rgba(127, 127, 127, 0.8)` (Gray)
- Amount: `rgba(31, 119, 180, 0.8)` (Blue)

## API Reference

### Main Application Function
```python
def main():
    """
    Main application entry point.
    Handles UI setup, data loading, filtering, and visualization rendering.
    """
```

### Utility Functions

#### Data Processing
```python
# Auto-generate year from date
if 'year' not in df.columns and 'date' in df.columns:
    df['year'] = pd.to_datetime(df['date']).dt.year

# Create amount categories
df['amount_category'] = pd.cut(
    df['amount'], 
    bins=[0, 1e6, 5e6, 1e7, float('inf')], 
    labels=['< 1M', '1M-5M', '5M-10M', '> 10M']
)
```

#### Filtering Logic
```python
# Apply region filters
if send_region != 'All':
    filtered_df = filtered_df[filtered_df['send region'] == send_region]

if receive_region != 'All':
    filtered_df = filtered_df[filtered_df['receiver region'] == receive_region]

if year_filter != 'All':
    filtered_df = filtered_df[filtered_df['year'] == year_filter]
```

## Performance Considerations

### Optimization Strategies

#### Data Caching
- Use `@st.cache_data` for expensive operations
- Cache data loading and processing
- Minimize redundant calculations

#### Memory Management
- Filter data early in pipeline
- Use efficient pandas operations
- Avoid unnecessary data copies

#### Rendering Performance
- Limit maximum data points in visualizations
- Use sampling for large datasets
- Implement progressive loading

### Scalability Limits

#### Data Size Recommendations
- **Optimal**: < 100,000 transactions
- **Acceptable**: 100,000 - 1,000,000 transactions
- **Challenging**: > 1,000,000 transactions

#### Browser Limitations
- Maximum plotly figure size: ~50MB
- Interactive elements: < 10,000 points optimal
- Animation frames: < 50 for smooth performance

## Development Guidelines

### Code Organization

#### Function Structure
```python
def visualization_function(df, **kwargs):
    """
    Brief description of function purpose.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Description of expected data structure
    **kwargs : dict
        Additional parameters
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Description of return value
        
    Raises:
    -------
    Exception
        Description of potential errors
    """
    try:
        # Implementation
        return fig
    except Exception as e:
        st.error(f"Error: {e}")
        return None
```

#### Error Handling Patterns
```python
try:
    # Risky operation
    result = process_data(df)
    return result
except Exception as e:
    st.error(f"Error in function_name: {e}")
    return None
```

### Testing Strategies

#### Data Validation
```python
# Check required columns
required_cols = ['send region', 'receiver region', 'amount']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Validate data types
if not pd.api.types.is_numeric_dtype(df['amount']):
    raise TypeError("Amount column must be numeric")
```

#### Performance Testing
```python
import time

start_time = time.time()
result = expensive_function(data)
execution_time = time.time() - start_time

if execution_time > 5.0:
    st.warning(f"Slow operation: {execution_time:.2f}s")
```

### Extension Points

#### Adding New Visualizations
1. Create function following naming convention
2. Implement error handling
3. Add to appropriate tab
4. Update documentation

#### Custom Filters
1. Add UI element in sidebar
2. Apply filter to `filtered_df`
3. Update summary statistics
4. Test with various data combinations

#### New Data Sources
1. Extend `load_data()` function
2. Add data validation logic
3. Update column mapping
4. Test compatibility

---

This documentation provides comprehensive technical details for developers working with or extending the Banking Flows Analysis Dashboard.
>>>>>>> 7e580db (app.py)
