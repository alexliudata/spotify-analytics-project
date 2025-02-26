import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import os
import glob
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Set visualization styles
plt.style.use('fivethirtyeight')
sns.set(style="whitegrid")

# Create output directory for plots
if not os.path.exists('output'):
    os.makedirs('output')

print("Sakkaris Spotify Analytics Project")
print("=================================")

# --- DATA LOADING AND PREPARATION ---
# Load Spotify audience timeline data
# This contains daily streams and monthly listeners

# Fix for the campaign date parsing issue
# This function properly handles the "m/d/yy UTC" format
def parse_spotify_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    
    try:
        # Remove the UTC part and parse with explicit format
        date_str = date_str.replace(' UTC', '')
        # Try various date formats
        for fmt in ['%m/%d/%y', '%m/%d/%Y']:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # If specific formats fail, try the flexible parser as last resort
        return pd.to_datetime(date_str)
    except:
        print(f"Could not parse date: {date_str}")
        return pd.NaT

# Replace the generic date conversion with our specialized function
print("\nLoading data...")
audience_df = pd.read_csv('Sakkarisaudiencetimeline.csv')
print("✓ Audience timeline data loaded")

# Convert date column to datetime
audience_df['date'] = pd.to_datetime(audience_df['date'])
print(f"Date range: {audience_df['date'].min()} to {audience_df['date'].max()}")

# Add useful date components
audience_df['year'] = audience_df['date'].dt.year
audience_df['month'] = audience_df['date'].dt.month
audience_df['day_of_week'] = audience_df['date'].dt.dayofweek
audience_df['day_name'] = audience_df['date'].dt.day_name()
audience_df['month_name'] = audience_df['date'].dt.month_name()
audience_df['quarter'] = audience_df['date'].dt.quarter

# Calculate rolling metrics
audience_df['7_day_avg_streams'] = audience_df['streams'].rolling(7).mean()
audience_df['30_day_avg_streams'] = audience_df['streams'].rolling(30).mean()
audience_df['7_day_avg_listeners'] = audience_df['listeners'].rolling(7).mean()
audience_df['30_day_avg_listeners'] = audience_df['listeners'].rolling(30).mean()

# Calculate growth metrics
audience_df['stream_growth'] = audience_df['streams'].pct_change() * 100
audience_df['listener_growth'] = audience_df['listeners'].pct_change() * 100
audience_df['follower_growth'] = audience_df['followers'].pct_change() * 100

# Calculate stream per listener ratio (engagement metric)
audience_df['streams_per_listener'] = audience_df['streams'] / audience_df['listeners']

print("\n1. OVERALL PERFORMANCE METRICS")
print("-----------------------------")
# Overall growth during the entire period
first_record = audience_df.iloc[0]
last_record = audience_df.iloc[-1]

print(f"Period: {first_record['date'].strftime('%Y-%m-%d')} to {last_record['date'].strftime('%Y-%m-%d')} ({len(audience_df)} days)")
print(f"Total streams: {audience_df['streams'].sum():,}")

# Calculate growth percentages
follower_growth_pct = ((last_record['followers'] - first_record['followers']) / first_record['followers']) * 100
print(f"Follower growth: {first_record['followers']:,} → {last_record['followers']:,} ({follower_growth_pct:.1f}%)")

# Calculate averages
print(f"Average daily streams: {audience_df['streams'].mean():.1f}")
print(f"Average daily listeners: {audience_df['listeners'].mean():.1f}")
print(f"Average streams per listener: {audience_df['streams_per_listener'].mean():.2f}")

# Peak performance
peak_streams_day = audience_df.loc[audience_df['streams'].idxmax()]
peak_listeners_day = audience_df.loc[audience_df['listeners'].idxmax()]

print(f"\nPeak streaming day: {peak_streams_day['date'].strftime('%Y-%m-%d')} with {peak_streams_day['streams']:,} streams")
print(f"Peak listeners day: {peak_listeners_day['date'].strftime('%Y-%m-%d')} with {peak_listeners_day['listeners']:,} listeners")

print("\n2. TIME SERIES ANALYSIS")
print("---------------------")

# --- EXPLORATORY DATA ANALYSIS ---
# Analyze streaming trends over time
# Identify seasonality and anomalies
# Visualize day-of-week patterns

# Analyze day of week patterns
dow_analysis = audience_df.groupby('day_name')[['streams', 'listeners', 'streams_per_listener']].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
print("\nDay of Week Performance (Average):")
print(dow_analysis)

# Analyze monthly patterns
# Alternative approach using string representation of year-month
audience_df['year_month'] = audience_df['date'].dt.strftime('%Y-%m')
monthly_df = audience_df.groupby('year_month').agg({
    'streams': 'sum',
    'listeners': 'mean',
    'followers': 'last'
}).reset_index()

fig = px.line(monthly_df, x='year_month', y='streams', 
             title='Monthly Total Streams',
             labels={'streams': 'Total Streams', 'year_month': 'Month'},
             markers=True)
fig.write_html("output/monthly_streams.html")
print("✓ Created monthly trends visualization")

# Find and load campaign data files based on their patterns
all_csv_files = glob.glob("*.csv")

# Helper function to clean numeric columns
def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, str):
        # Remove dollar signs, percentages and other non-numeric characters
        # Take only the first number if multiple are concatenated
        x = x.replace(',', '')
        match = re.search(r'[-+]?\d*\.\d+|\d+', x)
        if match:
            return float(match.group())
    return np.nan

# Find Marquee campaign files
marquee_files = [f for f in all_csv_files if "MARQUEE" in f.upper()]
if marquee_files:
    print(f"Found {len(marquee_files)} Marquee campaign files: {marquee_files}")
    # Load and concatenate all marquee files
    marquee_dfs = []
    for file in marquee_files:
        try:
            df = pd.read_csv(file)
            df['campaign_type'] = 'Marquee'
            df['file_source'] = file
            marquee_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if marquee_dfs:
        marquee_df = pd.concat(marquee_dfs, ignore_index=True)
        print(f"✓ Loaded {len(marquee_df)} Marquee campaign records")
        
        # Explicitly convert date columns using our specialized function
        for col in ['Release Date', 'Start Date', 'End Date']:
            if col in marquee_df.columns:
                marquee_df[col + '_date'] = marquee_df[col].apply(parse_spotify_date)
                print(f"Converted {col} to datetime using specialized parser")
                # Show sample of conversion results
                sample = marquee_df[[col, col + '_date']].head(3)
                print(f"Sample conversions:\n{sample}")
        
        # Clean numeric columns
        numeric_cols = ['Budget (incl. tax)', 'Spend (incl. tax)', 'Reach', 'Clicks', 
                      'Amplified Listeners', 'Conversion Rate', 'Streams Per Listener']
        for col in numeric_cols:
            if col in marquee_df.columns:
                marquee_df[col + '_clean'] = marquee_df[col].apply(clean_numeric)
                print(f"Cleaned {col} as {col}_clean")
    else:
        marquee_df = None
        print("⚠ No Marquee campaign data could be loaded")
else:
    print("⚠ No Marquee campaign files found")
    marquee_df = None

# Find Showcase campaign files
showcase_files = [f for f in all_csv_files if "SHOWCASE" in f.upper()]
if showcase_files:
    print(f"Found {len(showcase_files)} Showcase campaign files: {showcase_files}")
    # Load and concatenate all showcase files
    showcase_dfs = []
    for file in showcase_files:
        try:
            df = pd.read_csv(file)
            df['campaign_type'] = 'Showcase'
            df['file_source'] = file
            showcase_dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if showcase_dfs:
        showcase_df = pd.concat(showcase_dfs, ignore_index=True)
        print(f"✓ Loaded {len(showcase_df)} Showcase campaign records")
        
        # Explicitly convert date columns using our specialized function
        for col in ['Release Date', 'Start Date', 'End Date']:
            if col in showcase_df.columns:
                showcase_df[col + '_date'] = showcase_df[col].apply(parse_spotify_date)
                print(f"Converted {col} to datetime using specialized parser")
                # Show sample of conversion results
                sample = showcase_df[[col, col + '_date']].head(3)
                print(f"Sample conversions:\n{sample}")
        
        # Clean numeric columns
        numeric_cols = ['Budget (incl. tax)', 'Spend (incl. tax)', 'Reach', 'Clicks', 
                      'Amplified Listeners', 'Conversion Rate', 'Streams Per Listener']
        for col in numeric_cols:
            if col in showcase_df.columns:
                showcase_df[col + '_clean'] = showcase_df[col].apply(clean_numeric)
                print(f"Cleaned {col} as {col}_clean")
    else:
        showcase_df = None
        print("⚠ No Showcase campaign data could be loaded")
else:
    print("⚠ No Showcase campaign files found")
    showcase_df = None

# Better handling for Discovery mode files
discovery_files = [f for f in all_csv_files if "performance_report" in f.lower()]
if discovery_files:
    print(f"Found {len(discovery_files)} Discovery performance files: {discovery_files}")
    # Create a clean DataFrame to store extracted data
    discovery_clean_df = pd.DataFrame(columns=['month_year', 'song_name', 'listeners', 'listener_lift', 'new_listeners', 'saves', 'streams', 'stream_lift'])
    
    for file in discovery_files:
        try:
            # Extract month and year from filename
            month_year = file.replace('_performance_report.csv', '')
            # Convert month name to numeric month for date creation
            try:
                # Parse the month name and year
                if "JANUARY" in month_year.upper():
                    month_num = 1
                elif "FEBRUARY" in month_year.upper():
                    month_num = 2
                elif "MARCH" in month_year.upper():
                    month_num = 3
                elif "APRIL" in month_year.upper():
                    month_num = 4
                elif "MAY" in month_year.upper():
                    month_num = 5
                elif "JUNE" in month_year.upper():
                    month_num = 6
                elif "JULY" in month_year.upper():
                    month_num = 7
                elif "AUGUST" in month_year.upper():
                    month_num = 8
                elif "SEPTEMBER" in month_year.upper():
                    month_num = 9
                elif "OCTOBER" in month_year.upper():
                    month_num = 10
                elif "NOVEMBER" in month_year.upper():
                    month_num = 11
                elif "DECEMBER" in month_year.upper():
                    month_num = 12
                else:
                    month_num = 1  # default
                
                # Extract year
                year_match = re.search(r'20\d{2}', month_year)
                if year_match:
                    year = int(year_match.group())
                else:
                    year = 2024  # default
                
                # Create date for first day of month
                start_date = pd.Timestamp(year=year, month=month_num, day=1)
                print(f"Extracted date {start_date} from filename {file}")
            except Exception as e:
                print(f"Error parsing date from {month_year}: {e}")
                start_date = pd.NaT
            
            # Try reading the file - updated to use on_bad_lines instead of error_bad_lines
            try:
                raw_df = pd.read_csv(file, sep='\t', encoding='latin1', on_bad_lines='skip')
                print(f"Loaded {file} with tab separator")
            except:
                try:
                    raw_df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
                    print(f"Loaded {file} with default separator and latin1 encoding")
                except:
                    try:
                        # Last resort - try to read as string and parse manually
                        with open(file, 'r', encoding='latin1') as f:
                            content = f.read()
                        print(f"Read {file} as raw text, length: {len(content)} characters")
                        
                        # Look for pattern like: song name,123,4.5%,678
                        pattern = r'([^,]+),(\d+),([^,]+%),(\d+),(\d+)'
                        matches = re.findall(pattern, content)
                        
                        if matches:
                            print(f"Found {len(matches)} data entries using regex")
                            for match in matches[:5]:  # Process first 5 matches
                                song_name = match[0].strip()
                                listeners = int(match[1])
                                listener_lift = float(match[2].replace('%', ''))
                                new_listeners = int(match[3])
                                saves = int(match[4])
                                
                                discovery_clean_df = discovery_clean_df.append({
                                    'month_year': month_year,
                                    'start_date': start_date,
                                    'song_name': song_name,
                                    'listeners': listeners,
                                    'listener_lift': listener_lift,
                                    'new_listeners': new_listeners,
                                    'saves': saves
                                }, ignore_index=True)
                    except Exception as e:
                        print(f"All methods failed to parse file {file}: {e}")
                        continue
            
            # The rest of your Discovery file processing logic...
            
        except Exception as e:
            print(f"Error processing Discovery file {file}: {e}")
    
    if not discovery_clean_df.empty:
        print(f"✓ Extracted {len(discovery_clean_df)} rows of clean Discovery campaign data")
        discovery_df = discovery_clean_df
    else:
        discovery_df = None
        print("⚠ No Discovery campaign data could be extracted")
else:
    print("⚠ No Discovery campaign files found")
    discovery_df = None

# Calculate growth metrics
audience_df['stream_growth'] = audience_df['streams'].pct_change() * 100
audience_df['listener_growth'] = audience_df['listeners'].pct_change() * 100
audience_df['follower_growth'] = audience_df['followers'].pct_change() * 100

# Calculate stream per listener ratio (engagement metric)
audience_df['streams_per_listener'] = audience_df['streams'] / audience_df['listeners']

print("\n3. CAMPAIGN DATA EXPLORATION")
print("--------------------------")

def explore_campaign_data(df, campaign_type):
    if df is None:
        print(f"No {campaign_type} campaign data available.")
        return
    
    print(f"\n{campaign_type} Campaign Analysis:")
    print(f"Number of campaigns: {len(df)}")
    print("\nColumns in dataset:")
    print(df.columns.tolist())
    
    print("\nSample data:")
    print(df.head(3))
    
    # Look for date columns we've converted
    date_cols = [col for col in df.columns if '_date' in col.lower()]
    if date_cols:
        print(f"\nProcessed date columns: {date_cols}")
        for col in date_cols:
            if df[col].notna().any():
                print(f"{col} range: {df[col].min()} to {df[col].max()}")
    
    # Look for key cleaned metrics
    metric_cols = [col for col in df.columns if '_clean' in col.lower()]
    if metric_cols:
        print(f"\nCleaned metrics: {metric_cols}")
        for col in metric_cols:
            if df[col].notna().any():
                try:
                    mean_val = df[col].mean()
                    max_val = df[col].max()
                    print(f"{col}: Mean={mean_val:.2f}, Max={max_val:.2f}")
                except:
                    print(f"Could not calculate statistics for {col}")

# Explore each campaign type
print("\nExploring Marquee campaigns:")
explore_campaign_data(marquee_df, "Marquee")

print("\nExploring Showcase campaigns:")
explore_campaign_data(showcase_df, "Showcase")

print("\nExploring Discovery campaigns:")
explore_campaign_data(discovery_df, "Discovery")

print("\n4. CAMPAIGN IMPACT ANALYSIS")
print("-------------------------")

# --- STATISTICAL REGRESSION ANALYSIS ---
# Create campaign indicator variables
# Control for time-based effects (day of week, monthly trends)
# Measure the lift in streams during campaign periods

# Analyze Marquee Campaigns impact with fixed date handling
print("\nAnalyzing Marquee campaign impact...")
marquee_impact = None
if marquee_df is not None:
    # Debug info to diagnose date issues
    print(f"Marquee campaign dates available: {marquee_df['Start Date_date'].notna().sum()} of {len(marquee_df)}")
    print("First few campaign start dates:")
    print(marquee_df['Start Date_date'].head())
    
    # Filter out rows with missing start dates
    marquee_campaigns = marquee_df.dropna(subset=['Start Date_date'])
    
    if not marquee_campaigns.empty:
        print(f"Found {len(marquee_campaigns)} Marquee campaigns with valid dates")
        
        # Add visual markers for campaign periods
        campaign_markers = []
        for _, campaign in marquee_campaigns.iterrows():
            start_date = campaign['Start Date_date']
            end_date = campaign['End Date_date'] 
            release_name = campaign['Release Name'] if 'Release Name' in campaign else 'Unknown'
            
            # Add to markers for visualization
            campaign_markers.append({
                'date': start_date,
                'name': f"Marquee: {release_name}",
                'type': 'Marquee'
            })
            
        # Create a DataFrame of all campaign markers for visualization
        if campaign_markers:
            campaign_markers_df = pd.DataFrame(campaign_markers)
        else:
            campaign_markers_df = pd.DataFrame(columns=['date', 'name', 'type'])
        
        # Now calculate impact metrics
        impact_data = []
        for _, campaign in marquee_campaigns.iterrows():
            start_date = campaign['Start Date_date']
            end_date = campaign['End Date_date']
            
            # Ensure we have valid dates
            if pd.isna(start_date) or pd.isna(end_date):
                continue
                
            # Get campaign name
            release_name = campaign['Release Name'] if 'Release Name' in campaign else 'Unknown'
            
            # Calculate metrics for pre-campaign period (14 days before)
            pre_start = start_date - pd.Timedelta(days=14)
            pre_end = start_date - pd.Timedelta(days=1)
            pre_campaign = audience_df[(audience_df['date'] >= pre_start) & (audience_df['date'] <= pre_end)]
            
            if len(pre_campaign) < 7:  # Require at least 7 days of pre-campaign data
                continue
                
            pre_streams = pre_campaign['streams'].mean()
            pre_listeners = pre_campaign['listeners'].mean()
            
            # Calculate metrics for campaign period
            campaign_period = audience_df[(audience_df['date'] >= start_date) & (audience_df['date'] <= end_date)]
            
            if len(campaign_period) < 1:  # Require at least 1 day of campaign data
                continue
                
            campaign_streams = campaign_period['streams'].mean()
            campaign_listeners = campaign_period['listeners'].mean()
            
            # Calculate lift
            stream_lift_pct = ((campaign_streams - pre_streams) / pre_streams) * 100 if pre_streams > 0 else 0
            listener_lift_pct = ((campaign_listeners - pre_listeners) / pre_listeners) * 100 if pre_listeners > 0 else 0
            
            impact_data.append({
                'release_name': release_name,
                'start_date': start_date,
                'end_date': end_date,
                'pre_streams': pre_streams,
                'campaign_streams': campaign_streams,
                'stream_lift_pct': stream_lift_pct,
                'pre_listeners': pre_listeners,
                'campaign_listeners': campaign_listeners,
                'listener_lift_pct': listener_lift_pct,
                'spend': campaign['Spend (incl. tax)_clean'] if 'Spend (incl. tax)_clean' in campaign else None,
                'type': 'Marquee'
            })
        
        if impact_data:
            marquee_impact = pd.DataFrame(impact_data)
            print(f"\nMarquee Campaign Impact Analysis:")
            print(f"Analyzed {len(marquee_impact)} campaigns")
            
            print("\nStream Lift Statistics:")
            print(marquee_impact['stream_lift_pct'].describe())
            
            print("\nListener Lift Statistics:")
            print(marquee_impact['listener_lift_pct'].describe())
            
            # Top performing campaigns
            if len(marquee_impact) > 0:
                top_campaigns = marquee_impact.sort_values('stream_lift_pct', ascending=False).head(3)
                print("\nTop 3 Most Effective Campaigns (by Stream Lift):")
                for _, campaign in top_campaigns.iterrows():
                    print(f"Campaign for {campaign['release_name']} starting {campaign['start_date'].strftime('%Y-%m-%d')}: {campaign['stream_lift_pct']:.1f}% stream lift")
        else:
            print("No valid campaign periods found for Marquee campaigns.")
    else:
        print("No valid campaign dates found in Marquee data. Raw date samples:")
        print(marquee_df['Start Date'].head())
else:
    print("No suitable Marquee campaign data available.")

# Showcase Campaigns
print("\nAnalyzing Showcase campaign impact...")
showcase_impact = None
if showcase_df is not None and 'Start Date_date' in showcase_df.columns:
    # Filter out rows with missing start dates
    showcase_campaigns = showcase_df.dropna(subset=['Start Date_date'])
    
    if not showcase_campaigns.empty:
        impact_data = []
        for _, campaign in showcase_campaigns.iterrows():
            start_date = campaign['Start Date_date']
            end_date = campaign['End Date_date']
            
            # Ensure we have valid dates
            if pd.isna(start_date) or pd.isna(end_date):
                continue
                
            # Get campaign name
            release_name = campaign['Release Name'] if 'Release Name' in campaign.index else 'Unknown'
            
            # Calculate metrics for pre-campaign period (14 days before)
            pre_start = start_date - pd.Timedelta(days=14)
            pre_end = start_date - pd.Timedelta(days=1)
            pre_campaign = audience_df[(audience_df['date'] >= pre_start) & (audience_df['date'] <= pre_end)]
            
            if len(pre_campaign) < 7:  # Require at least 7 days of pre-campaign data
                continue
                
            pre_streams = pre_campaign['streams'].mean()
            pre_listeners = pre_campaign['listeners'].mean()
            
            # Calculate metrics for campaign period
            campaign_period = audience_df[(audience_df['date'] >= start_date) & (audience_df['date'] <= end_date)]
            
            if len(campaign_period) < 1:  # Require at least 1 day of campaign data
                continue
                
            campaign_streams = campaign_period['streams'].mean()
            campaign_listeners = campaign_period['listeners'].mean()
            
            # Calculate lift
            stream_lift_pct = ((campaign_streams - pre_streams) / pre_streams) * 100 if pre_streams > 0 else 0
            listener_lift_pct = ((campaign_listeners - pre_listeners) / pre_listeners) * 100 if pre_listeners > 0 else 0
            
            impact_data.append({
                'release_name': release_name,
                'start_date': start_date,
                'end_date': end_date,
                'pre_streams': pre_streams,
                'campaign_streams': campaign_streams,
                'stream_lift_pct': stream_lift_pct,
                'pre_listeners': pre_listeners,
                'campaign_listeners': campaign_listeners,
                'listener_lift_pct': listener_lift_pct,
                'spend': campaign['Spend (incl. tax)_clean'] if 'Spend (incl. tax)_clean' in campaign.index else None
            })
        
        if impact_data:
            showcase_impact = pd.DataFrame(impact_data)
            print(f"\nShowcase Campaign Impact Analysis:")
            print(f"Analyzed {len(showcase_impact)} campaigns")
            
            print("\nStream Lift Statistics:")
            print(showcase_impact['stream_lift_pct'].describe())
            
            print("\nListener Lift Statistics:")
            print(showcase_impact['listener_lift_pct'].describe())
            
            # Top performing campaigns
            if len(showcase_impact) > 0:
                top_campaigns = showcase_impact.sort_values('stream_lift_pct', ascending=False).head(3)
                print("\nTop 3 Most Effective Campaigns (by Stream Lift):")
                for _, campaign in top_campaigns.iterrows():
                    print(f"Campaign for {campaign['release_name']} starting {campaign['start_date'].strftime('%Y-%m-%d')}: {campaign['stream_lift_pct']:.1f}% stream lift")
        else:
            print("No valid campaign periods found for Showcase campaigns.")
    else:
        print("No valid campaign dates found in Showcase data.")
else:
    print("No suitable Showcase campaign data available.")

# Discovery Campaigns
print("\nAnalyzing Discovery campaign impact...")
discovery_impact = None
if discovery_df is not None and 'start_date' in discovery_df.columns:
    impact_data = []
    for _, campaign in discovery_df.dropna(subset=['start_date', 'stream_lift']).iterrows():
        # Extract the song name and start date
        song_name = campaign['song_name'] if 'song_name' in campaign else 'Unknown'
        start_date = campaign['start_date']
        
        # Use the reported lift directly
        stream_lift_pct = campaign['stream_lift'] if 'stream_lift' in campaign else 0
        listener_lift_pct = campaign['listener_lift'] if 'listener_lift' in campaign else 0
        
        impact_data.append({
            'song_name': song_name,
            'start_date': start_date,
            'stream_lift_pct': stream_lift_pct,
            'listener_lift_pct': listener_lift_pct
        })
    
    if impact_data:
        discovery_impact = pd.DataFrame(impact_data)
        print(f"\nDiscovery Campaign Impact Analysis:")
        print(f"Analyzed {len(discovery_impact)} campaigns")
        
        print("\nStream Lift Statistics:")
        print(discovery_impact['stream_lift_pct'].describe())
        
        print("\nListener Lift Statistics:")
        print(discovery_impact['listener_lift_pct'].describe())
        
        # Top performing campaigns
        if len(discovery_impact) > 0:
            top_campaigns = discovery_impact.sort_values('stream_lift_pct', ascending=False).head(3)
            print("\nTop 3 Most Effective Campaigns (by Stream Lift):")
            for _, campaign in top_campaigns.iterrows():
                print(f"Campaign for {campaign['song_name']} starting {campaign['start_date'].strftime('%Y-%m-%d')}: {campaign['stream_lift_pct']:.1f}% stream lift")
    else:
        print("No valid impact data found for Discovery campaigns.")
else:
    print("No suitable Discovery campaign data available.")

print("\n5. VISUALIZATION")
print("---------------")
print("Generating visualizations... (saved to output folder)")

# 1. Create a comprehensive time series with campaign markers
# Combine all campaign markers
all_campaign_markers = []

if marquee_impact is not None and len(marquee_impact) > 0:
    for _, campaign in marquee_impact.iterrows():
        all_campaign_markers.append({
            'date': campaign['start_date'],
            'name': f"Marquee: {campaign['release_name']}",
            'type': 'Marquee',
            'lift': campaign['stream_lift_pct']
        })

if showcase_impact is not None and len(showcase_impact) > 0:
    for _, campaign in showcase_impact.iterrows():
        all_campaign_markers.append({
            'date': campaign['start_date'],
            'name': f"Showcase: {campaign['release_name']}",
            'type': 'Showcase',
            'lift': campaign['stream_lift_pct']
        })

campaign_markers_df = pd.DataFrame(all_campaign_markers) if all_campaign_markers else pd.DataFrame(columns=['date', 'name', 'type', 'lift'])

# Enhanced time series visualization with campaign impacts
fig = go.Figure()

# Add streams line
fig.add_trace(go.Scatter(
    x=audience_df['date'], 
    y=audience_df['streams'],
    mode='lines',
    name='Daily Streams',
    line=dict(color='#1DB954', width=2)  # Spotify green
))

# Add 30-day moving average
fig.add_trace(go.Scatter(
    x=audience_df['date'], 
    y=audience_df['streams'].rolling(30).mean(),
    mode='lines',
    name='30-Day Average',
    line=dict(color='#191414', width=2, dash='dash')  # Spotify black
))

# Add campaign markers
if not campaign_markers_df.empty:
    # Marquee campaigns
    marquee_markers = campaign_markers_df[campaign_markers_df['type'] == 'Marquee']
    if not marquee_markers.empty:
        fig.add_trace(go.Scatter(
            x=marquee_markers['date'],
            y=[audience_df['streams'].max() * 1.05] * len(marquee_markers),
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=15, color='#1DB954'),
            text=marquee_markers['name'],
            textposition="top center",
            name='Marquee Campaigns',
            hovertemplate='%{text}<br>Date: %{x}<br>Stream Lift: %{customdata:.1f}%',
            customdata=marquee_markers['lift']
        ))

    # Showcase campaigns
    showcase_markers = campaign_markers_df[campaign_markers_df['type'] == 'Showcase']
    if not showcase_markers.empty:
        fig.add_trace(go.Scatter(
            x=showcase_markers['date'],
            y=[audience_df['streams'].max() * 1.1] * len(showcase_markers),
            mode='markers+text',
            marker=dict(symbol='star', size=15, color='#FF9D00'),
            text=showcase_markers['name'],
            textposition="top center",
            name='Showcase Campaigns',
            hovertemplate='%{text}<br>Date: %{x}<br>Stream Lift: %{customdata:.1f}%',
            customdata=showcase_markers['lift']
        ))

# Update layout with Spotify-inspired design
fig.update_layout(
    title="Streaming Performance with Campaign Impact",
    xaxis_title="Date",
    yaxis_title="Daily Streams",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    font=dict(family="Arial, sans-serif"),
    template="plotly_white",
    height=800
)

fig.write_html("output/streaming_with_campaigns.html")
print("✓ Created enhanced streaming timeline with campaign markers")

# 2. Campaign ROI Analysis
if marquee_impact is not None and showcase_impact is not None:
    # Calculate ROI metrics (streams gained per dollar spent)
    if 'spend' in marquee_impact.columns:
        # Add ROI calculations for Marquee
        marquee_impact['campaign_days'] = (marquee_impact['end_date'] - marquee_impact['start_date']).dt.days + 1
        marquee_impact['additional_streams'] = (marquee_impact['campaign_streams'] - marquee_impact['pre_streams']) * marquee_impact['campaign_days']
        marquee_impact['stream_cost'] = marquee_impact['spend'] / marquee_impact['additional_streams'].where(marquee_impact['additional_streams'] > 0, 0)
        marquee_impact['stream_cost'] = marquee_impact['stream_cost'].replace([np.inf, -np.inf], np.nan)

    if 'spend' in showcase_impact.columns:
        # Add ROI calculations for Showcase
        showcase_impact['campaign_days'] = (showcase_impact['end_date'] - showcase_impact['start_date']).dt.days + 1
        showcase_impact['additional_streams'] = (showcase_impact['campaign_streams'] - showcase_impact['pre_streams']) * showcase_impact['campaign_days']
        showcase_impact['stream_cost'] = showcase_impact['spend'] / showcase_impact['additional_streams'].where(showcase_impact['additional_streams'] > 0, 0)
        showcase_impact['stream_cost'] = showcase_impact['stream_cost'].replace([np.inf, -np.inf], np.nan)

    # Combine for comparison
    campaign_comparison = []
    
    if 'stream_cost' in marquee_impact.columns:
        for _, camp in marquee_impact.dropna(subset=['stream_cost']).iterrows():
            campaign_comparison.append({
                'name': camp['release_name'],
                'type': 'Marquee',
                'stream_lift': camp['stream_lift_pct'],
                'additional_streams': camp['additional_streams'],
                'spend': camp['spend'],
                'cost_per_stream': camp['stream_cost']
            })
            
    if 'stream_cost' in showcase_impact.columns:
        for _, camp in showcase_impact.dropna(subset=['stream_cost']).iterrows():
            campaign_comparison.append({
                'name': camp['release_name'],
                'type': 'Showcase',
                'stream_lift': camp['stream_lift_pct'],
                'additional_streams': camp['additional_streams'],
                'spend': camp['spend'],
                'cost_per_stream': camp['stream_cost']
            })
    
    if campaign_comparison:
        camp_df = pd.DataFrame(campaign_comparison)
        
        # ROI Bubble Chart
        fig = px.scatter(
            camp_df, 
            x="stream_lift", 
            y="cost_per_stream",
            size="additional_streams",
            color="type",
            hover_name="name",
            log_y=True,
            size_max=50,
            title="Campaign ROI Analysis",
            labels={
                "stream_lift": "Stream Lift (%)",
                "cost_per_stream": "Cost per Additional Stream ($)",
                "additional_streams": "Total Additional Streams",
                "type": "Campaign Type"
            }
        )
        
        fig.update_layout(
            xaxis=dict(ticksuffix="%"),
            yaxis=dict(tickprefix="$"),
            height=700,
            hovermode="closest"
        )
        
        fig.write_html("output/campaign_roi_analysis.html")
        print("✓ Created campaign ROI analysis visualization")

# 3. Campaign performance by release type
if marquee_df is not None and 'Release Type' in marquee_df.columns:
    # Combine campaign impacts with release types
    marquee_with_types = []
    
    for _, impact in marquee_impact.iterrows():
        release_name = impact['release_name']
        release_type = None
        
        # Find matching release type
        matching_rows = marquee_df[marquee_df['Release Name'] == release_name]
        if not matching_rows.empty:
            release_type = matching_rows.iloc[0]['Release Type']
        
        if release_type:
            marquee_with_types.append({
                'release_name': release_name,
                'release_type': release_type,
                'stream_lift': impact['stream_lift_pct'],
                'campaign_type': 'Marquee'
            })
    
    # Do the same for showcase campaigns if available
    if showcase_df is not None and 'Release Type' in showcase_df.columns:
        for _, impact in showcase_impact.iterrows():
            release_name = impact['release_name']
            release_type = None
            
            # Find matching release type
            matching_rows = showcase_df[showcase_df['Release Name'] == release_name]
            if not matching_rows.empty:
                release_type = matching_rows.iloc[0]['Release Type']
            
            if release_type:
                marquee_with_types.append({
                    'release_name': release_name,
                    'release_type': release_type,
                    'stream_lift': impact['stream_lift_pct'],
                    'campaign_type': 'Showcase'
                })
    
    if marquee_with_types:
        type_df = pd.DataFrame(marquee_with_types)
        
        # Group by release type
        type_performance = type_df.groupby(['release_type', 'campaign_type'])['stream_lift'].mean().reset_index()
        
        # Create bar chart
        fig = px.bar(
            type_performance, 
            x="release_type", 
            y="stream_lift", 
            color="campaign_type",
            barmode="group",
            title="Campaign Effectiveness by Release Type",
            labels={
                "release_type": "Release Type",
                "stream_lift": "Average Stream Lift (%)",
                "campaign_type": "Campaign Type"
            }
        )
        
        fig.update_layout(
            xaxis_title="Release Type",
            yaxis_title="Average Stream Lift (%)",
            yaxis=dict(ticksuffix="%"),
            height=600
        )
        
        fig.write_html("output/release_type_performance.html")
        print("✓ Created release type performance visualization")

# Generate HTML report
try:
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spotify Campaign Analytics Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #1DB954;
            }}
            .header {{
                background-color: #191414;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .header h1 {{
                color: #1DB954;
                margin: 0;
            }}
            .section {{
                margin-bottom: 40px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .stat-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .stat-box {{
                background-color: white;
                border-left: 5px solid #1DB954;
                padding: 15px;
                margin-bottom: 20px;
                width: 30%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                margin: 5px 0;
            }}
            .campaign-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .campaign-table th, .campaign-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .campaign-table th {{
                background-color: #1DB954;
                color: white;
            }}
            .campaign-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .viz-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .highlight {{
                color: #1DB954;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Spotify Campaign Analytics Report</h1>
            <p>Artist: Sakkaris | Analysis Period: {audience_df['date'].min().strftime('%Y-%m-%d')} - {audience_df['date'].max().strftime('%Y-%m-%d')}</p>
        </div>
        
        <div class="section">
            <h2>Overall Performance Metrics</h2>
            <div class="stat-container">
                <div class="stat-box">
                    <p>Total Streams</p>
                    <p class="stat-value">{audience_df['streams'].sum():,.0f}</p>
                </div>
                <div class="stat-box">
                    <p>Follower Growth</p>
                    <p class="stat-value">{follower_growth_pct:.1f}%</p>
                </div>
                <div class="stat-box">
                    <p>Average Daily Streams</p>
                    <p class="stat-value">{audience_df['streams'].mean():.1f}</p>
                </div>
            </div>
            <p>Peak streaming day was <span class="highlight">{audience_df.loc[audience_df['streams'].idxmax(), 'date'].strftime('%Y-%m-%d')}</span> with <span class="highlight">{audience_df['streams'].max():,.0f}</span> streams.</p>
            <p>Best performing day of the week: <span class="highlight">{dow_analysis['streams'].idxmax()}</span> ({dow_analysis['streams'].max():.1f} average streams)</p>
        </div>
        
        <div class="section">
            <h2>Campaign Effectiveness</h2>
            <h3>Marquee Campaigns</h3>
    """
    
    if marquee_impact is not None and len(marquee_impact) > 0:
        html_report += f"""
            <p>Average Stream Lift: <span class="highlight">{marquee_impact['stream_lift_pct'].mean():.1f}%</span></p>
            <table class="campaign-table">
                <tr>
                    <th>Release</th>
                    <th>Campaign Period</th>
                    <th>Stream Lift</th>
                    <th>Listener Lift</th>
                </tr>
        """
        
        # Sort by effectiveness
        for _, camp in marquee_impact.sort_values('stream_lift_pct', ascending=False).iterrows():
            html_report += f"""
                <tr>
                    <td>{camp['release_name']}</td>
                    <td>{camp['start_date'].strftime('%Y-%m-%d')} to {camp['end_date'].strftime('%Y-%m-%d')}</td>
                    <td>{camp['stream_lift_pct']:.1f}%</td>
                    <td>{camp['listener_lift_pct']:.1f}%</td>
                </tr>
            """
        
        html_report += "</table>"
    else:
        html_report += "<p>No Marquee campaign data available.</p>"
    
    html_report += """
            <h3>Showcase Campaigns</h3>
    """
    
    if showcase_impact is not None and len(showcase_impact) > 0:
        html_report += f"""
            <p>Average Stream Lift: <span class="highlight">{showcase_impact['stream_lift_pct'].mean():.1f}%</span></p>
            <table class="campaign-table">
                <tr>
                    <th>Release</th>
                    <th>Campaign Period</th>
                    <th>Stream Lift</th>
                    <th>Listener Lift</th>
                </tr>
        """
        
        # Sort by effectiveness
        for _, camp in showcase_impact.sort_values('stream_lift_pct', ascending=False).iterrows():
            html_report += f"""
                <tr>
                    <td>{camp['release_name']}</td>
                    <td>{camp['start_date'].strftime('%Y-%m-%d')} to {camp['end_date'].strftime('%Y-%m-%d')}</td>
                    <td>{camp['stream_lift_pct']:.1f}%</td>
                    <td>{camp['listener_lift_pct']:.1f}%</td>
                </tr>
            """
        
        html_report += "</table>"
    else:
        html_report += "<p>No Showcase campaign data available.</p>"
    
    html_report += """
        </div>
        
        <div class="section">
            <h2>Key Insights</h2>
            <ul>
    """
    
    # Dynamically generate insights
    insights = []
    
    # Top campaign insight
    best_campaign_lift = 0
    best_campaign_type = ""
    best_campaign_name = ""
    
    if marquee_impact is not None and len(marquee_impact) > 0:
        idx = marquee_impact['stream_lift_pct'].idxmax()
        if marquee_impact.loc[idx, 'stream_lift_pct'] > best_campaign_lift:
            best_campaign_lift = marquee_impact.loc[idx, 'stream_lift_pct']
            best_campaign_type = "Marquee"
            best_campaign_name = marquee_impact.loc[idx, 'release_name']
    
    if showcase_impact is not None and len(showcase_impact) > 0:
        idx = showcase_impact['stream_lift_pct'].idxmax()
        if showcase_impact.loc[idx, 'stream_lift_pct'] > best_campaign_lift:
            best_campaign_lift = showcase_impact.loc[idx, 'stream_lift_pct']
            best_campaign_type = "Showcase"
            best_campaign_name = showcase_impact.loc[idx, 'release_name']
    
    if best_campaign_lift > 0:
        insights.append(f"Your most effective campaign was the <span class='highlight'>{best_campaign_type}</span> campaign for <span class='highlight'>{best_campaign_name}</span> with a stream lift of <span class='highlight'>{best_campaign_lift:.1f}%</span>.")
    
    # Campaign type comparison
    if (marquee_impact is not None and len(marquee_impact) > 0) and (showcase_impact is not None and len(showcase_impact) > 0):
        marquee_avg = marquee_impact['stream_lift_pct'].mean()
        showcase_avg = showcase_impact['stream_lift_pct'].mean()
        
        if marquee_avg > showcase_avg:
            insights.append(f"<span class='highlight'>Marquee</span> campaigns performed better on average with {marquee_avg:.1f}% stream lift compared to {showcase_avg:.1f}% for Showcase campaigns.")
        else:
            insights.append(f"<span class='highlight'>Showcase</span> campaigns performed better on average with {showcase_avg:.1f}% stream lift compared to {marquee_avg:.1f}% for Marquee campaigns.")
    
    # Weekly patterns
    insights.append(f"Your streams are highest on <span class='highlight'>{dow_analysis['streams'].idxmax()}</span> and lowest on <span class='highlight'>{dow_analysis['streams'].idxmin()}</span>, suggesting scheduling new releases or campaigns accordingly could maximize impact.")
    
    # Follower to stream conversion
    avg_streams_per_follower = audience_df['streams'].sum() / audience_df['followers'].max()
    insights.append(f"You're generating an average of <span class='highlight'>{avg_streams_per_follower:.1f}</span> streams per follower, which is a key metric for artist sustainability.")
    
    # Add all insights to report
    for insight in insights:
        html_report += f"<li>{insight}</li>"
    
    html_report += """
            </ul>
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
            <p>For interactive visualizations, please open the HTML files in the output folder:</p>
            <ol>
                <li>streaming_with_campaigns.html - Timeline with campaign markers</li>
                <li>campaign_roi_analysis.html - ROI analysis for all campaigns</li>
                <li>monthly_streams.html - Monthly streaming patterns</li>
                <li>day_of_week_analysis.html - Day of week performance</li>
                <li>engagement_analysis.html - Streams per listener ratio</li>
                <li>seasonal_decomposition.html - Trend and seasonal components</li>
            </ol>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ol>
    """
    
    # Generate recommendations based on data
    recommendations = []
    
    # Best campaign type recommendation
    if (marquee_impact is not None and len(marquee_impact) > 0) and (showcase_impact is not None and len(showcase_impact) > 0):
        marquee_avg = marquee_impact['stream_lift_pct'].mean()
        showcase_avg = showcase_impact['stream_lift_pct'].mean()
        
        if marquee_avg > showcase_avg:
            recommendations.append(f"Prioritize <span class='highlight'>Marquee</span> campaigns for future releases as they're generating {marquee_avg-showcase_avg:.1f}% higher stream lift than Showcase campaigns.")
        else:
            recommendations.append(f"Prioritize <span class='highlight'>Showcase</span> campaigns for future releases as they're generating {showcase_avg-marquee_avg:.1f}% higher stream lift than Marquee campaigns.")
    
    # Timing recommendation
    best_day = dow_analysis['streams'].idxmax()
    recommendations.append(f"Time your release campaigns to launch on <span class='highlight'>{best_day}</span> when your audience is most active.")
    
    # Benchmark comparison
    if best_campaign_lift > 50:
        recommendations.append("Your top campaigns are performing above industry averages - document your promotional strategies for these releases to replicate this success.")
    else:
        recommendations.append("Consider refining your campaign targeting to improve performance - analyze listener demographics of your most successful campaigns.")
    
    # Based on seasonality
    recommendations.append("Analyze the seasonal decomposition visualization to identify recurring patterns in your streaming activity, and plan major releases to align with these patterns.")
    
    # Add all recommendations to report
    for recommendation in recommendations:
        html_report += f"<li>{recommendation}</li>"
    
    html_report += """
            </ol>
        </div>
        
        <div style="text-align: center; margin-top: 50px; color: #999; font-size: 0.8em;">
            <p>Report generated on """ + datetime.now().strftime("%Y-%m-%d") + """</p>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    with open("output/spotify_campaign_report.html", "w") as f:
        f.write(html_report)
    
    print("✓ Created comprehensive HTML report: output/spotify_campaign_report.html")
except Exception as e:
    print(f"⚠ Error generating HTML report: {e}")

print("\n6. SEASONAL DECOMPOSITION")
print("-----------------------")
# Perform time series decomposition on streams
try:
    # Resample to weekly data for smoother decomposition
    weekly_streams = audience_df.set_index('date')['streams'].resample('W').mean()
    
    # Perform decomposition
    decomposition = seasonal_decompose(weekly_streams, model='additive', period=52)  # 52 weeks in a year
    
    # Create figure with subplots
    fig = make_subplots(rows=4, cols=1, 
                       shared_xaxes=True,
                       subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                       vertical_spacing=0.05)
    
    # Add traces
    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(height=900, title_text="Time Series Decomposition of Weekly Streams")
    fig.write_html("output/seasonal_decomposition.html")
    print("✓ Created seasonal decomposition visualization")
except Exception as e:
    print(f"⚠ Couldn't perform seasonal decomposition: {e}")

print("\n7. CONCLUSIONS & INSIGHTS")
print("-----------------------")
print("Key findings from the analysis:")

# Overall growth
print(f"1. Overall Growth: {follower_growth_pct:.1f}% follower growth during the analyzed period")

# Best performing days
best_day = dow_analysis['streams'].idxmax()
print(f"2. Best Day for Streaming: {best_day} with {dow_analysis.loc[best_day, 'streams']:.1f} average streams")

# Seasonality insights
print("3. Seasonality: Please refer to the seasonal decomposition visualization for patterns")

# Campaign effectiveness
if campaign_markers_df.empty:
    print("No campaign data available for campaign effectiveness analysis.")
else:
    campaign_df = pd.DataFrame(campaign_markers_df)
    if len(campaign_df) > 0 and 'type' in campaign_df.columns and 'lift' in campaign_df.columns:
        campaign_avg_lift = campaign_df.groupby('type')['lift'].mean()
        if not campaign_avg_lift.empty:
            best_campaign_type = campaign_avg_lift.idxmax()
            avg_lift = campaign_avg_lift.loc[best_campaign_type]
            print(f"4. Most Effective Campaign Type: {best_campaign_type} with {avg_lift:.1f}% average stream lift")

print("\n8. NEXT STEPS")
print("-----------")
print("To further enhance this analysis:")
print("1. Integrate listener demographic data if available")
print("2. Compare performance against similar artists in your genre")
print("3. Build a predictive model for future streaming performance")
print("4. Develop a campaign ROI calculator to optimize marketing budget")
print("\nThe visualizations have been saved to the 'output' folder.")

print("\n9. REGRESSION ANALYSIS")
print("---------------------")

# --- REGRESSION DIAGNOSTICS ---
# Test residuals for normality using Q-Q plot
# Check for autocorrelation with Ljung-Box test
# Verify homoscedasticity with Breusch-Pagan test

# Prepare data for regression
regression_df = audience_df.copy()

# Add campaign flags
regression_df['marquee_active'] = 0
regression_df['showcase_active'] = 0

# Mark days when campaigns were active
if marquee_impact is not None:
    for _, campaign in marquee_impact.iterrows():
        mask = (regression_df['date'] >= campaign['start_date']) & (regression_df['date'] <= campaign['end_date'])
        regression_df.loc[mask, 'marquee_active'] = 1

if showcase_impact is not None:
    for _, campaign in showcase_impact.iterrows():
        mask = (regression_df['date'] >= campaign['start_date']) & (regression_df['date'] <= campaign['end_date'])
        regression_df.loc[mask, 'showcase_active'] = 1
        
# Add day of week dummies
regression_df['day_of_week'] = regression_df['date'].dt.dayofweek
for i in range(7):
    regression_df[f'dow_{i}'] = (regression_df['day_of_week'] == i).astype(int)

# Add month dummies
for i in range(1, 13):
    regression_df[f'month_{i}'] = (regression_df['date'].dt.month == i).astype(int)
    
# Add lag features
regression_df['streams_lag1'] = regression_df['streams'].shift(1)
regression_df['streams_lag7'] = regression_df['streams'].shift(7)

# Drop NAs
reg_df = regression_df.dropna().copy()

# Create X and y
y = reg_df['streams']
X = reg_df[['marquee_active', 'showcase_active', 'streams_lag1', 'streams_lag7'] + 
           [f'dow_{i}' for i in range(1, 7)] +  # Exclude Sunday as reference
           [f'month_{i}' for i in range(2, 13)]]  # Exclude January as reference

# Add constant
import statsmodels.api as sm
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())

# Show key findings
print("\nKey Regression Findings:")
print(f"Marquee Campaign Effect: {model.params['marquee_active']:.2f} additional streams (p={model.pvalues['marquee_active']:.4f})")
print(f"Showcase Campaign Effect: {model.params['showcase_active']:.2f} additional streams (p={model.pvalues['showcase_active']:.4f})")

# Find best day of week from regression
dow_params = {i: model.params.get(f'dow_{i}', 0) for i in range(1, 7)}
best_dow_idx = max(dow_params, key=dow_params.get)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
print(f"Best Day of Week (from regression): {days[best_dow_idx]}")

# Save regression results to HTML
from statsmodels.iolib.summary2 import summary_col
with open('output/regression_results.html', 'w') as f:
    f.write(model.summary().as_html())
print("✓ Saved regression results to output/regression_results.html")

print("\n10. TIME SERIES FORECASTING")
print("-----------------------")

# --- TIME SERIES FORECASTING ---
# Create time-based features and lag variables
# Implement Random Forest for time series prediction
# Identify key predictive features

# Feature engineering for time series
forecast_df = audience_df.copy()
forecast_df['dayofweek'] = forecast_df['date'].dt.dayofweek
forecast_df['quarter'] = forecast_df['date'].dt.quarter
forecast_df['month'] = forecast_df['date'].dt.month
forecast_df['year'] = forecast_df['date'].dt.year
forecast_df['dayofyear'] = forecast_df['date'].dt.dayofyear
forecast_df['is_weekend'] = forecast_df['dayofweek'].isin([5, 6]).astype(int)

# Create lag features (last 7 days, last 14 days, last 28 days)
for lag in [1, 2, 3, 7, 14, 28]:
    forecast_df[f'streams_lag_{lag}'] = forecast_df['streams'].shift(lag)
    forecast_df[f'listeners_lag_{lag}'] = forecast_df['listeners'].shift(lag)

# Create rolling window features
for window in [7, 14, 30]:
    forecast_df[f'streams_rolling_{window}'] = forecast_df['streams'].rolling(window=window).mean()
    forecast_df[f'listeners_rolling_{window}'] = forecast_df['listeners'].rolling(window=window).mean()

# Add campaign flags
forecast_df['campaign_active'] = 0
if marquee_impact is not None and len(marquee_impact) > 0:
    for _, campaign in marquee_impact.iterrows():
        mask = (forecast_df['date'] >= campaign['start_date']) & (forecast_df['date'] <= campaign['end_date'])
        forecast_df.loc[mask, 'campaign_active'] = 1
if showcase_impact is not None and len(showcase_impact) > 0:
    for _, campaign in showcase_impact.iterrows():
        mask = (forecast_df['date'] >= campaign['start_date']) & (forecast_df['date'] <= campaign['end_date'])
        forecast_df.loc[mask, 'campaign_active'] = 1

# Drop missing values
forecast_df = forecast_df.dropna()

# Define features for model
features = ['dayofweek', 'quarter', 'month', 'year', 'is_weekend', 'campaign_active',
           'streams_lag_1', 'streams_lag_7', 'streams_lag_14', 
           'streams_rolling_7', 'streams_rolling_30']

# Train-test split (time-based)
train_size = int(len(forecast_df) * 0.8)
train_df = forecast_df.iloc[:train_size]
test_df = forecast_df.iloc[train_size:]

# Define X and y
X_train = train_df[features]
y_train = train_df['streams']
X_test = test_df[features]
y_test = test_df['streams']

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
train_preds = rf_model.predict(X_train)
test_preds = rf_model.predict(X_test)

# Evaluate model
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

print(f"Training MAE: {train_mae:.2f} streams")
print(f"Testing MAE: {test_mae:.2f} streams")
print(f"Training RMSE: {train_rmse:.2f} streams")
print(f"Testing RMSE: {test_rmse:.2f} streams")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Visualize predictions vs actual
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=test_df['date'], 
    y=y_test,
    mode='lines',
    name='Actual Streams',
    line=dict(color='#1DB954', width=2)  # Spotify green
))

fig.add_trace(go.Scatter(
    x=test_df['date'], 
    y=test_preds,
    mode='lines',
    name='Predicted Streams',
    line=dict(color='#FF9D00', width=2)  # Orange
))

fig.update_layout(
    title="Stream Prediction: Actual vs Predicted",
    xaxis_title="Date",
    yaxis_title="Daily Streams",
    template="plotly_white"
)

fig.write_html("output/stream_predictions.html")
print("✓ Created prediction visualization")

# Generate future predictions (next 30 days)
last_date = forecast_df['date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
future_df = pd.DataFrame({'date': future_dates})

# Engineer features for future dates
future_df['dayofweek'] = future_df['date'].dt.dayofweek
future_df['quarter'] = future_df['date'].dt.quarter
future_df['month'] = future_df['date'].dt.month
future_df['year'] = future_df['date'].dt.year
future_df['dayofyear'] = future_df['date'].dt.dayofyear
future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)

# Use the last known values for lagged features
future_df['streams_lag_1'] = forecast_df['streams'].iloc[-1]
future_df['streams_lag_7'] = forecast_df['streams'].iloc[-7]
future_df['streams_lag_14'] = forecast_df['streams'].iloc[-14]
future_df['streams_rolling_7'] = forecast_df['streams'].iloc[-7:].mean()
future_df['streams_rolling_30'] = forecast_df['streams'].iloc[-30:].mean()
future_df['campaign_active'] = 0  # Assume no campaigns unless specified

# Generate predictions
future_preds = rf_model.predict(future_df[features])
future_df['predicted_streams'] = future_preds

# Visualize future predictions
fig = go.Figure()
# Historical data
fig.add_trace(go.Scatter(
    x=forecast_df['date'].iloc[-60:],  # Last 60 days
    y=forecast_df['streams'].iloc[-60:],
    mode='lines',
    name='Historical Streams',
    line=dict(color='#1DB954', width=2)  # Spotify green
))

# Future predictions
fig.add_trace(go.Scatter(
    x=future_df['date'], 
    y=future_df['predicted_streams'],
    mode='lines',
    name='Forecasted Streams',
    line=dict(color='#191414', width=2, dash='dash')  # Spotify black
))

fig.update_layout(
    title="30-Day Stream Forecast",
    xaxis_title="Date",
    yaxis_title="Daily Streams",
    template="plotly_white"
)

fig.write_html("output/stream_forecast.html")
print("✓ Created 30-day stream forecast visualization")

# Add this code to save campaign impact data for the dashboard
if marquee_impact is not None:
    marquee_impact.to_csv('output/marquee_impact.csv', index=False)
    print("✓ Saved Marquee impact data for dashboard")
    
if showcase_impact is not None:
    showcase_impact.to_csv('output/showcase_impact.csv', index=False)
    print("✓ Saved Showcase impact data for dashboard")

print("\n11. REGRESSION DIAGNOSTICS")
print("-----------------------")

# Check residuals for normality
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Create Q-Q plot for residuals
fig = plt.figure(figsize=(10, 6))
sm.qqplot(model.resid, line='45', fit=True, ax=plt.gca())
plt.title('Q-Q Plot of Residuals')
plt.tight_layout()
plt.savefig('output/residual_qq_plot.png')
print("✓ Created residual normality plot")

# Test for autocorrelation in residuals
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(model.resid, lags=[10])
print(f"✓ Ljung-Box test for autocorrelation: p-value = {lb_result.iloc[0, 1]:.4f}")
print(f"  {'Evidence of autocorrelation' if lb_result.iloc[0, 1] < 0.05 else 'No significant autocorrelation'}")

# Heteroscedasticity test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(f"✓ Breusch-Pagan test for heteroscedasticity: p-value = {bp_test[1]:.4f}")
print(f"  {'Evidence of heteroscedasticity' if bp_test[1] < 0.05 else 'No significant heteroscedasticity'}")

print("\n12. TIME SERIES CROSS-VALIDATION")
print("-----------------------")

# --- TIME SERIES CROSS-VALIDATION ---
# Use TimeSeriesSplit for proper time-based validation
# Measure performance consistency across multiple time periods
# Calculate average cross-validated MAE and RMSE

print("Performing time series cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores_mae = []
cv_scores_rmse = []

X = forecast_df[features]
y = forecast_df['streams']

# Visualize the CV splits
plt.figure(figsize=(12, 4))
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    plt.scatter(test_idx, [i+0.5]*len(test_idx), 
                c='#1DB954', s=10, label='Test' if i == 0 else "")
    plt.scatter(train_idx, [i+0.5]*len(train_idx), 
                c='#191414', s=10, label='Train' if i == 0 else "")
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5], ['Split 1', 'Split 2', 'Split 3', 'Split 4', 'Split 5'])
plt.title('Time Series Cross-Validation')
plt.legend()
plt.tight_layout()
plt.savefig('output/time_series_cv.png')

# Perform cross-validation
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    preds = rf_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    cv_scores_mae.append(mae)
    cv_scores_rmse.append(rmse)
    
    print(f"Fold {i+1}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

print(f"Average CV MAE: {np.mean(cv_scores_mae):.2f} streams")
print(f"Average CV RMSE: {np.mean(cv_scores_rmse):.2f} streams")

# Then continue with your final model training as before 