import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image

st.image(Image.open("cornell_tech_logo.jpg"), width=600, use_container_width=True)
st.title("Cornell Tech Weather")
st.subheader("Explore temperature patterns from 1950 to 2021")

st.sidebar.header("Settings")
temp_unit = st.sidebar.radio("Temperature Unit", ["Fahrenheit", "Celsius"])

@st.cache_data
def load_data():
    df = pd.read_csv("weather.csv")
    
    df['date'] = pd.to_datetime(df['time'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    if 'Ktemp' in df.columns:
        # Convert Kelvin to both Celsius and Fahrenheit
        df['Ctemp'] = df['Ktemp'] - 273.15
        df['Ftemp'] = df['Ctemp'] * (9/5) + 32
    return df


df = load_data()
min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_year = st.slider("Select a year", min_year, max_year, max_year)

year_data = df[df['year'] == selected_year]

temp_col = 'Ftemp' if temp_unit == "Fahrenheit" else 'Ctemp'
temp_symbol = "°F" if temp_unit == "Fahrenheit" else "°C"

monthly_avg = year_data.groupby('month')[temp_col].mean().reset_index()
historical_avg = df.groupby('month')[temp_col].mean().reset_index()
show_historical = st.checkbox("Show historical average (1950-present)")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly_avg['month'],
    y=monthly_avg[temp_col],
    mode='lines+markers',
    name=f'{selected_year}',
    line=dict(color='firebrick', width=3),
    marker=dict(size=10)
))

if show_historical:
    fig.add_trace(go.Scatter(
        x=historical_avg['month'],
        y=historical_avg[temp_col],
        mode='lines+markers',
        name='Historical Average',
        line=dict(color='royalblue', width=2, dash='dash'),
        marker=dict(size=8)
    ))

fig.update_layout(
    title=f'Monthly Average Temperatures for {selected_year}',
    xaxis=dict(
        title='Month',
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ),
    yaxis=dict(title=f'Temperature ({temp_symbol})'),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Temperature Statistics")
cols = st.columns(4)
with cols[0]:
    st.metric("Annual Average", f"{year_data[temp_col].mean():.1f}{temp_symbol}")
with cols[1]:
    st.metric("Highest Temp", f"{year_data[temp_col].max():.1f}{temp_symbol}")
with cols[2]:
    st.metric("Lowest Temp", f"{year_data[temp_col].min():.1f}{temp_symbol}")
with cols[3]:
    historical_mean = df[temp_col].mean()
    year_mean = year_data[temp_col].mean()
    delta = year_mean - historical_mean
    st.metric("vs. Historical Avg", f"{year_mean:.1f}{temp_symbol}", f"{delta:+.1f}{temp_symbol}")

st.markdown("---")
st.header("When will Cornell Tech finally be warm?")


threshold_f = 55
threshold_c = (threshold_f - 32) * 5/9  
threshold = threshold_f if temp_unit == "Fahrenheit" else threshold_c

yearly_avg_temp = df.groupby('year')[temp_col].mean().reset_index()
yearly_avg_temp['above_threshold'] = yearly_avg_temp[temp_col] > threshold

warm_years = yearly_avg_temp[yearly_avg_temp['above_threshold']]
if not warm_years.empty:
    first_warm_year = warm_years.iloc[0]['year']
    first_warm_temp = warm_years.iloc[0][temp_col]
    st.success(f"The first year when the average temperature exceeded {threshold:.1f}{temp_symbol} was **{int(first_warm_year)}** with an average temperature of **{first_warm_temp:.2f}{temp_symbol}**.")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=yearly_avg_temp['year'],
    y=yearly_avg_temp[temp_col],
    mode='lines+markers',
    name='Yearly Average',
    line=dict(color='firebrick', width=2),
    marker=dict(size=6)
))

fig2.add_trace(go.Scatter(
    x=[yearly_avg_temp['year'].min(), yearly_avg_temp['year'].max()],
    y=[threshold, threshold],
    mode='lines',
    name=f'{threshold:.1f}{temp_symbol} Threshold',
    line=dict(color='green', width=2, dash='dash')
))

if not warm_years.empty:
    fig2.add_trace(go.Scatter(
        x=[first_warm_year],
        y=[first_warm_temp],
        mode='markers',
        name=f'First Year Above {threshold:.1f}{temp_symbol} ({int(first_warm_year)})',
        marker=dict(color='green', size=12, symbol='star')
    ))

fig2.update_layout(
    title='Yearly Average Temperatures (1950-Present)',
    xaxis=dict(title='Year'),
    yaxis=dict(title=f'Temperature ({temp_symbol})'),
    hovermode='x unified'
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("Temperature Trend Analysis")
years_array = yearly_avg_temp['year'].values
temps_array = yearly_avg_temp[temp_col].values
slope, intercept = np.polyfit(years_array, temps_array, 1)

temp_change_per_decade = slope * 10
st.write(f"Temperature is changing at a rate of **{temp_change_per_decade:.2f}{temp_symbol} per decade**.")

df['decade'] = (df['year'] // 10) * 10
decade_avg = df.groupby('decade')[temp_col].mean().reset_index()
decade_avg['decade_label'] = decade_avg['decade'].astype(str) + 's'
df['season'] = pd.cut(
    df['month'], 
    bins=[0, 3, 6, 9, 12], 
    labels=['Winter', 'Spring', 'Summer', 'Fall'],
    include_lowest=True
)

season_decade_avg = df.groupby(['decade', 'season'])[temp_col].mean().reset_index()
pivot_data = season_decade_avg.pivot(index='decade', columns='season', values=temp_col)
fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=yearly_avg_temp['year'],
    y=yearly_avg_temp[temp_col],
    mode='markers',
    name='Annual Average',
    marker=dict(
        size=10,
        color=yearly_avg_temp[temp_col],
        colorscale='RdBu_r',
        colorbar=dict(
            title=f'Temperature ({temp_symbol})',
            thickness=20,
            len=0.9,
            y=0.5,
            yanchor="middle",
            x=1.05,
            xanchor="left"
        ),
        showscale=True,
        cmin=yearly_avg_temp[temp_col].min(),
        cmax=yearly_avg_temp[temp_col].max()
    ),
    hovertemplate='Year: %{x}<br>Temperature: %{y:.2f}' + temp_symbol + '<extra></extra>'
))

years_range = np.array([years_array.min(), years_array.max()])
trendline_y = slope * years_range + intercept

fig_trend.add_trace(
    go.Scatter(
        x=years_range, 
        y=trendline_y, 
        mode='lines', 
        name='Trend',
        line=dict(color='black', width=2, dash='dash')
    )
)

fig_trend.add_trace(
    go.Scatter(
        x=years_range,
        y=[threshold, threshold],
        mode='lines',
        name=f'{threshold:.1f}{temp_symbol} Threshold',
        line=dict(color='green', width=2)
    )
)

fig_trend.update_layout(
    height=500,
    xaxis=dict(title='Year'),
    yaxis=dict(title=f'Temperature ({temp_symbol})'),
    title='Temperature Trends (1950-Present)',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1
    ),
    margin=dict(r=150)  # Add right margin to make room for the colorbar
)

st.plotly_chart(fig_trend, use_container_width=True)

df['season'] = pd.cut(
    df['month'], 
    bins=[0, 3, 6, 9, 12], 
    labels=['Winter', 'Spring', 'Summer', 'Fall'],
    include_lowest=True
)

season_months = {
    'Winter': 'Dec-Feb',
    'Spring': 'Mar-May',
    'Summer': 'Jun-Aug',
    'Fall': 'Sep-Nov'
}

season_decade_avg = df.groupby(['decade', 'season'])[temp_col].mean().reset_index()

pivot_data = season_decade_avg.pivot(index='decade', columns='season', values=temp_col)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=pivot_data.values,
    x=pivot_data.columns,
    y=[f"{int(decade)}s" for decade in pivot_data.index],
    colorscale='RdBu_r',
    zmin=pivot_data.values.min(),
    zmax=pivot_data.values.max(),
    text=[[f"{val:.1f}{temp_symbol}" for val in row] for row in pivot_data.values],
    texttemplate="%{text}",
    textfont={"size":12},
    colorbar=dict(
        title=f'Temp ({temp_symbol})',
        thickness=20,
        len=0.9,
        y=0.5,
        yanchor="middle",
        x=1.05,
        xanchor="left"
    )
))

season_labels = [f"{season}<br><span style='font-size: 10px'>{season_months[season]}</span>" for season in pivot_data.columns]

fig_heatmap.update_layout(
    height=450,  
    xaxis_title='Season',
    yaxis_title='Decade',
    title='Seasonal Temperature Changes by Decade',
    margin=dict(r=150, b=60),  
    xaxis=dict(
        tickvals=list(range(len(pivot_data.columns))),
        ticktext=season_labels
    )
)

st.plotly_chart(fig_heatmap, use_container_width=True)


st.subheader("Decade-by-Decade Comparison")

earliest_decade = decade_avg['decade'].min()
latest_decade = decade_avg['decade'].max()

earliest_temp = decade_avg[decade_avg['decade'] == earliest_decade][temp_col].values[0]
latest_temp = decade_avg[decade_avg['decade'] == latest_decade][temp_col].values[0]

temp_change = latest_temp - earliest_temp
percent_change = (temp_change / earliest_temp) * 100

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        f"Average Temp ({int(earliest_decade)}s)", 
        f"{earliest_temp:.2f}{temp_symbol}", 
        delta=None
    )

with col2:
    st.metric(
        f"Average Temp ({int(latest_decade)}s)", 
        f"{latest_temp:.2f}{temp_symbol}", 
        delta=f"{temp_change:+.2f}{temp_symbol} vs {int(earliest_decade)}s",
        delta_color="inverse"  # Red for increasing temperature
    )

with col3:
    # Calculate threshold for projection based on temperature unit
    target_temp_f = 60
    target_temp_c = (target_temp_f - 32) * 5/9
    target_temp = target_temp_f if temp_unit == "Fahrenheit" else target_temp_c
    
    # Calculate years to reach target temperature at current rate
    years_to_target = (target_temp - latest_temp) / (slope)
    target_year = int(latest_decade + years_to_target)
    
    st.metric(
        f"Projected Year to Reach {target_temp:.1f}{temp_symbol}", 
        f"{target_year}",
        delta=f"{int(years_to_target)} years from {int(latest_decade)}"
    )

# Add explanatory text
st.info(f"""
This analysis shows the clear warming trend at Cornell Tech over the decades. The data reveals:
1. The rate of temperature increase has been {temp_change_per_decade:.2f}{temp_symbol} per decade
2. Seasonal variations show warming across all seasons
3. If trends continue, Cornell Tech will reach {target_temp:.1f}{temp_symbol} by {target_year}
""")


def add_cornell_tech_map():
    cornell_tech_lat = 40.7559768629782
    cornell_tech_lon = -73.95584979896849
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=[cornell_tech_lat],
        lon=[cornell_tech_lon],
        mode='markers',
        marker=dict(
            size=10,
            color='#B31B1B',  # Cornell red
            opacity=0.8
        ),
        text=['Cornell Tech'],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map", 
            center=dict(lat=cornell_tech_lat, lon=cornell_tech_lon),
            zoom=14
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Cornell Tech Location"
    )
    
    return fig

st.subheader("Come visit Cornell Tech on Roosevelt Island, NYC!")

st.plotly_chart(add_cornell_tech_map(), use_container_width=True)
