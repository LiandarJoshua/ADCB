
## Deployed App Link

https://transactiondashboardsample.streamlit.app/

# Banking Flows Analysis Dashboard 🏦

A simple, interactive web app to visualize banking transactions between different regions around the world.

## What Does This App Do?

This dashboard helps you understand banking transaction patterns by showing:
- **Where money flows** between different regions on a world map
- **How transactions move** through different stages (banks, currencies, payment methods)
- **Which regions** have the most transaction activity
- **How transaction patterns change** over time

## Quick Start

### Step 1: Install Requirements
You need Python installed on your computer. Then run:
```bash
pip install streamlit pandas plotly numpy
```

### Step 2: Run the App
```bash
streamlit run banking_flows_dashboard.py
```

### Step 3: Open in Browser
The app will automatically open at `http://localhost:8501`

## How to Use the Dashboard

### 📊 Main Features

**🗺️ Flow Map Tab**
- See transaction flows on a world map
- Thicker lines = more money flowing
- Different colors = different transaction amounts
- Click "Play Flow Animation" to see money moving

**🔄 Sankey Diagram Tab**
- Follow how transactions move step-by-step
- From sender → receiver → currency → payment method
- Hover over connections to see details
- Click "Play Flow" for animation

**🔥 Heatmap Tab**
- Quick overview of which regions trade most
- Darker colors = more transactions
- Easy to spot busy trading corridors

**📈 Time Series Tab**
- See how transactions change over time
- Compare different regions or banks
- Spot trends and patterns

### 🔍 Filtering Your Datas

Use the sidebar controls to focus on specific data:
- **Sender Region**: Show only transactions from a specific region
- **Receiver Region**: Show only transactions to a specific region  
- **Year**: Look at data from a specific year

## Your Data File

### Required Format
Your data should be a CSV file with these columns:

**Must Have:**
- `send region` - Where money comes from (e.g., "North America")
- `receiver region` - Where money goes to (e.g., "Europe")
- `amount` - How much money (numbers only)

**Nice to Have:**
- `sender` - Individual bank or person sending
- `receiver` - Individual bank or person receiving
- `currency` - USD, EUR, etc.
- `year` or `date` - When the transaction happened
- `payment_method` - Wire transfer, SWIFT, etc.

### Where to Put Your File
Place your CSV file at `app/simulated_transactions.csv` or change this line in the code:
```python
default_path = "your/file/path.csv"
```

## Common Questions

**Q: The app won't start**
- Make sure Python is installed
- Check you installed all required packages
- Verify your CSV file exists at the specified path

**Q: No data appears**
- Check your filters aren't too restrictive (try "All" for all filters)
- Make sure your CSV has the required columns
- Check for typos in region names

**Q: App is slow**
- Large files (>100,000 rows) may be slow
- Try filtering to smaller date ranges
- Close other browser tabs to free memory

**Q: Map doesn't show my regions**
- The app has pre-defined region coordinates
- Make sure your region names match the standard ones
- Contact support to add new regions

## Supported Regions

The app recognizes these regions:
- North America, South America, Europe, Asia, Africa
- Middle East, Australia, Caribbean, Central America
- Southeast Asia, East Asia, South Asia, Central Asia

## Getting Help

1. Check this guide first
2. Try the troubleshooting section
3. Make sure your data format matches the requirements
4. Test with a small sample file first

## What's Included

- Interactive world map with transaction flows
- Step-by-step transaction flow diagrams
- Regional comparison heatmaps  
- Time-based trend analysis
- Real-time filtering and updates
- Animation effects for better understanding

---

