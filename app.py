import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(page_title="Obesity Healthcare Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for blue cards and improved visuals
st.markdown("""
    <style>
    .blue-metric-card {
        background: linear-gradient(135deg, #2196F3 80%, #1976D2 100%);
        border-radius: 12px;
        padding: 22px 18px 18px 18px;
        margin: 10px 0;
        color: #fff !important;
        box-shadow: 0 4px 16px rgba(33, 150, 243, 0.15);
        font-family: 'Segoe UI', sans-serif;
    }
    
    .red-metric-card {
        background: linear-gradient(135deg, #f44336 80%, #d32f2f 100%);
        border-radius: 12px;
        padding: 22px 18px 18px 18px;
        margin: 10px 0;
        color: #fff !important;
        box-shadow: 0 4px 16px rgba(244, 67, 54, 0.15);
        font-family: 'Segoe UI', sans-serif;
    }
    
    .purple-metric-card {
        background: linear-gradient(135deg, #9c27b0 80%, #7b1fa2 100%);
        border-radius: 12px;
        padding: 22px 18px 18px 18px;
        margin: 10px 0;
        color: #fff !important;
        box-shadow: 0 4px 16px rgba(156, 39, 176, 0.15);
        font-family: 'Segoe UI', sans-serif;
    }
    
    .pink-metric-card {
        background: linear-gradient(135deg, #e91e63 80%, #c2185b 100%);
        border-radius: 12px;
        padding: 22px 18px 18px 18px;
        margin: 10px 0;
        color: #fff !important;
        box-shadow: 0 4px 16px rgba(233, 30, 99, 0.15);
        font-family: 'Segoe UI', sans-serif;
    }
    .blue-metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E3F2FD;
st.markdown("<div class='graph-title-container'>Correlation Between Health Factors</div>", unsafe_allow_html=True)
st.markdown("<div class='graph-content'>", unsafe_allow_html=True)      margin-bottom: 6px;
    }
    .blue-metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 2px;
    }
    .blue-metric-delta {
        font-size: 1rem;
        color: #B3E5FC;
    }
    .dashboard-section {
        padding: 20px 0;
        border-bottom: 1px solid #E3F2FD;
        margin-bottom: 30px;
    }
    .graph-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1565C0;
        margin-bottom: 5px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #E8EAF6;
    }
    
    /* Style sidebar headers */
    .sidebar .block-container h1, 
    .sidebar .block-container h2,
    .sidebar .block-container h3 {
        color: #3F51B5;
    }
    
    /* Style sidebar filters - buttons */
    .stButton > button {
        background-color: #3F51B5 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Style sliders */
    .stSlider > div > div > div {
        background-color: #3F51B5 !important;
    }
    
    /* Style multiselect */
    .stMultiSelect > div > div > div {
        background-color: #E8EAF6;
        border-radius: 4px;
        border: 1px solid #9FA8DA;
    }
    
    /* Style multiselect dropdown items */
    .stMultiSelect span[data-baseweb="tag"] {
        background-color: #3F51B5 !important;
        color: white !important;
    }
    
    /* Improved graph container with title inside */
    .graph-container {
        background: white;
        border-radius: 10px;
        padding: 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        overflow: hidden;
    }
    
    .graph-title-container {
        background: linear-gradient(90deg, #3f51b5, #5c6bc0);
        color: white;
        padding: 10px 15px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .graph-content {
        padding: 15px;
    }
    
    /* Improved insights box */
    .insights-box {
        background: #E8EAF6;
        padding: 12px 15px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 3px solid #3F51B5;
        font-size: 0.9rem;
    }
    
    .insights-title {
        color: #303F9F;
        font-weight: 600;
        margin-bottom: 5px;
        font-size: 0.95rem;
    }
    
    /* Section header styling */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3F51B5;
        margin: 35px 0 5px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #9FA8DA;
        text-shadow: 1px 1px 2px rgba(63, 81, 181, 0.2);
    }
    
    /* Lifestyle section special styling */
    .lifestyle-section-header {
        background: linear-gradient(90deg, #673ab7, #9575cd);
        color: white;
        padding: 15px 25px;
        border-radius: 12px 12px 0 0;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 40px 0 0 0;
        box-shadow: 0 4px 12px rgba(103, 58, 183, 0.25);
        display: flex;
        align-items: center;
    }
    
    .lifestyle-section-header .emoji {
        font-size: 2.4rem;
        margin-right: 15px;
    }
    
    .lifestyle-section-subtitle {
        background: #EDE7F6;
        color: #311B92;
        padding: 12px 25px;
        border-radius: 0 0 12px 12px;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0 0 25px 0;
        border-left: 1px solid #B39DDB;
        border-right: 1px solid #B39DDB;
        border-bottom: 1px solid #B39DDB;
        box-shadow: 0 4px 12px rgba(103, 58, 183, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Color palettes based on the Material Design palette shared
# Main colors from the palette
PRIMARY_COLORS = {
    'RED': '#f44336', 
    'PINK': '#e91e63', 
    'PURPLE': '#9c27b0', 
    'DEEP_PURPLE': '#673ab7', 
    'INDIGO': '#3f51b5', 
    'BLUE': '#2196f3'
}

# Lighter shades (50, 100, 200)
LIGHT_COLORS = {
    'RED_50': '#ffebee',
    'PINK_50': '#fce4ec',
    'PURPLE_50': '#f3e5f5',
    'DEEP_PURPLE_50': '#ede7f6',
    'INDIGO_50': '#e8eaf6',
    'BLUE_50': '#e3f2fd',
    
    'RED_100': '#ffcdd2',
    'PINK_100': '#f8bbd0',
    'PURPLE_100': '#e1bee7',
    'DEEP_PURPLE_100': '#d1c4e9',
    'INDIGO_100': '#c5cae9',
    'BLUE_100': '#bbdefb',
    
    'RED_200': '#ef9a9a',
    'PINK_200': '#f48fb1',
    'PURPLE_200': '#ce93d8',
    'DEEP_PURPLE_200': '#b39ddb',
    'INDIGO_200': '#9fa8da',
    'BLUE_200': '#90caf9'
}

# Darker shades (300, 400, 500)  
DARK_COLORS = {
    'RED_300': '#e57373',
    'PINK_300': '#f06292',
    'PURPLE_300': '#ba68c8',
    'DEEP_PURPLE_300': '#9575cd',
    'INDIGO_300': '#7986cb',
    'BLUE_300': '#64b5f6',
    
    'RED_400': '#ef5350',
    'PINK_400': '#ec407a',
    'PURPLE_400': '#ab47bc',
    'DEEP_PURPLE_400': '#7e57c2',
    'INDIGO_400': '#5c6bc0',
    'BLUE_400': '#42a5f5',
    
    'RED_500': '#f44336',
    'PINK_500': '#e91e63',
    'PURPLE_500': '#9c27b0',
    'DEEP_PURPLE_500': '#673ab7',
    'INDIGO_500': '#3f51b5',
    'BLUE_500': '#2196f3'
}

# Palette for main chart sequences
CHART_COLORS = [
    PRIMARY_COLORS['BLUE'],
    PRIMARY_COLORS['PURPLE'],
    PRIMARY_COLORS['PINK'],
    PRIMARY_COLORS['RED'],
    PRIMARY_COLORS['DEEP_PURPLE'],
    PRIMARY_COLORS['INDIGO']
]

# Palette for obesity levels (consistent across all charts)
OBESITY_COLORS = {
    'Insufficient Weight': DARK_COLORS['BLUE_300'],
    'Normal Weight': DARK_COLORS['INDIGO_300'],
    'Overweight': DARK_COLORS['PURPLE_300'],  
    'Obesity': DARK_COLORS['RED_300']
}

# Gender colors
GENDER_COLORS = {
    'Male': DARK_COLORS['BLUE_400'],
    'Female': DARK_COLORS['PINK_400']
}

# Title and description
st.title("🏥 Obesity Analytics Dashboard 2025")
st.markdown("""
This interactive dashboard explores the relationships between demographic characteristics, lifestyle factors, and obesity classes.
Use the filters in the sidebar to customize your analysis and gain valuable insights into obesity patterns.
""")

# Load and process the data
@st.cache_data
def load_data():
    df = pd.read_excel('data/Obesity_Dataset.xlsx')
    
    # Map all categorical variables
    df['Sex'] = df['Sex'].map({1: 'Male', 2: 'Female'})
    df['Obesity_Level'] = df['Class'].map({
        1: 'Insufficient Weight', 2: 'Normal Weight',
        3: 'Overweight', 4: 'Obesity'
    })
    df['Physical_Activity_Level'] = df['Physical_Excercise'].map({
        1: 'No Activity', 2: '1-2 days/week',
        3: '3-4 days/week', 4: '5-6 days/week',
        5: '6+ days/week'
    })
    df['Technology_Usage'] = df['Schedule_Dedicated_to_Technology'].map({
        1: '0-2 hours', 2: '3-5 hours', 3: '5+ hours'
    })
    df['Transportation'] = df['Type_of_Transportation_Used'].map({
        1: 'Automobile', 2: 'Motorbike', 3: 'Bike',
        4: 'Public Transport', 5: 'Walking'
    })
    df['Vegetable_Consumption'] = df['Frequency_of_Consuming_Vegetables'].map({
        1: 'Rarely', 2: 'Sometimes', 3: 'Always'
    })
    df['Liquid_Intake'] = df['Liquid_Intake_Daily'].map({
        1: '< 1L', 2: '1-2L', 3: '> 2L'
    })
    
    # Map other categorical variables
    df['Fast_Food'] = df['Consumption_of_Fast_Food'].map({
        1: 'Yes', 2: 'No'
    })
    
    df['Family_History'] = df['Overweight_Obese_Family'].map({
        1: 'Yes', 2: 'No'
    })
    
    df['Tracks_Calories'] = df['Calculation_of_Calorie_Intake'].map({
        1: 'Yes', 2: 'No'
    })
    
    df['Smoker'] = df['Smoking'].map({
        1: 'Yes', 2: 'No'
    })
    
    df['Meal_Count'] = df['Number_of_Main_Meals_Daily'].map({
        1: '1-2 meals', 2: '3 meals', 3: 'More than 3'
    })
    
    df['Snacking'] = df['Food_Intake_Between_Meals'].map({
        1: 'Rarely', 2: 'Sometimes', 3: 'Frequently', 4: 'Always'
    })
    
    # Create age groups
    bins = [17, 25, 35, 45, 55]
    labels = ['18-25', '26-35', '36-45', '46+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    
    return df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("📊 Analysis Filters")
    
    # Age range filter
    st.subheader("Age Range")
    age_range = st.slider("Select age range", 
                         min_value=int(df['Age'].min()),
                         max_value=int(df['Age'].max()),
                         value=(int(df['Age'].min()), int(df['Age'].max())))
    
    # Gender filter
    st.subheader("Gender")
    selected_gender = st.multiselect("Select gender",
                                   options=df['Sex'].unique(),
                                   default=df['Sex'].unique())
    
    # Activity level filter
    st.subheader("Physical Activity")
    selected_activity = st.multiselect("Select activity level",
                                     options=df['Physical_Activity_Level'].unique(),
                                     default=df['Physical_Activity_Level'].unique())
    
    # Obesity level filter
    st.subheader("Obesity Level")
    selected_obesity = st.multiselect("Select obesity level",
                                    options=df['Obesity_Level'].unique(),
                                    default=df['Obesity_Level'].unique())
    
    # Add a reset button
    if st.button('Reset All Filters'):
        # This will cause a rerun with default values
        st.experimental_rerun()

# Apply filters
mask = (
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Sex'].isin(selected_gender)) &
    (df['Physical_Activity_Level'].isin(selected_activity)) &
    (df['Obesity_Level'].isin(selected_obesity))
)
filtered_df = df[mask]

# Data Quality Overview (moved from bottom to top)
st.markdown("<div class='section-header'>📊 Data Quality Overview</div>", unsafe_allow_html=True)
quality_cols = st.columns(4)
with quality_cols[0]:
    st.metric("Sample Size", f"{len(filtered_df):,} ")
with quality_cols[1]:
    st.metric("Age Range", f"{filtered_df['Age'].min() if not filtered_df.empty else 0}-{filtered_df['Age'].max() if not filtered_df.empty else 0} years")
with quality_cols[2]:
    gender_ratio = f"{len(filtered_df[filtered_df['Sex']=='Female'])/len(filtered_df):.1%} Female" if not filtered_df.empty else "N/A"
    st.metric("Gender Ratio", gender_ratio)
with quality_cols[3]:
    data_completeness = f"{100 - filtered_df.isnull().sum().sum()/len(filtered_df):.1f}%" if not filtered_df.empty else "N/A"
    st.metric("Data Completeness", data_completeness)

# --- BLUE METRIC CARDS ---
st.header("📈 Key Insight Cards")
card_cols = st.columns(4)

with card_cols[0]:
    total_participants = len(filtered_df)
    st.markdown(f"""
    <div class='blue-metric-card'>
        <div class='blue-metric-label'>Total Participants</div>
        <div class='blue-metric-value'>{total_participants:,}</div>
    </div>
    """, unsafe_allow_html=True)
with card_cols[1]:
    obesity_rate = (len(filtered_df[filtered_df['Obesity_Level'] == 'Obesity']) / total_participants) * 100 if total_participants else 0
    st.markdown(f"""
    <div class='red-metric-card'>
        <div class='blue-metric-label'>Obesity Rate</div>
        <div class='blue-metric-value'>{obesity_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with card_cols[2]:
    overweight_rate = (len(filtered_df[filtered_df['Obesity_Level'] == 'Overweight']) / total_participants) * 100 if total_participants else 0
    st.markdown(f"""
    <div class='purple-metric-card'>
        <div class='blue-metric-label'>Overweight Rate</div>
        <div class='blue-metric-value'>{overweight_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with card_cols[3]:
    normal_rate = (len(filtered_df[filtered_df['Obesity_Level'] == 'Normal Weight']) / total_participants) * 100 if total_participants else 0
    st.markdown(f"""
    <div class='blue-metric-card'>
        <div class='blue-metric-label'>Normal Weight Rate</div>
        <div class='blue-metric-value'>{normal_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# Second row of cards
card2_cols = st.columns(4)
with card2_cols[0]:
    female_rate = (len(filtered_df[filtered_df['Sex'] == 'Female']) / total_participants) * 100 if total_participants else 0
    st.markdown(f"""
    <div class='pink-metric-card'>
        <div class='blue-metric-label'>Female</div>
        <div class='blue-metric-value'>{female_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with card2_cols[1]:
    avg_age = filtered_df['Age'].mean() if total_participants else 0
    st.markdown(f"""
    <div class='purple-metric-card'>
        <div class='blue-metric-label'>Average Age</div>
        <div class='blue-metric-value'>{avg_age:.1f} years</div>
    </div>
    """, unsafe_allow_html=True)
with card2_cols[2]:
    fast_food_rate = (len(filtered_df[filtered_df['Fast_Food'] == 'Yes']) / total_participants) * 100 if total_participants else 0
    st.markdown(f"""
    <div class='red-metric-card'>
        <div class='blue-metric-label'>Fast Food Consumption</div>
        <div class='blue-metric-value'>{fast_food_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with card2_cols[3]:
    low_activity = (len(filtered_df[filtered_df['Physical_Activity_Level'].isin(['No Activity', '1-2 days/week'])]) / total_participants) * 100 if total_participants else 0
    st.markdown(f"""
    <div class='blue-metric-card'>
        <div class='blue-metric-label'>Low Physical Activity</div>
        <div class='blue-metric-value'>{low_activity:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# --- SINGLE PAGE DASHBOARD LAYOUT ---

# SECTION 1: DEMOGRAPHIC OVERVIEW
st.markdown("<div class='section-header'>📊 Demographic Distribution</div>", unsafe_allow_html=True)
st.write("Overview of demographic characteristics and obesity classes distribution.")

# Row 1: 2 graphs (Pie chart and Sunburst)
row1_cols = st.columns(2)

with row1_cols[0]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Obesity Class Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Pie chart
    pie_data = filtered_df['Obesity_Level'].value_counts().reset_index()
    pie_data.columns = ['Obesity Class', 'Count']
    
    # Create color mapping dictionary for consistency
    color_map = {obesity: OBESITY_COLORS[obesity] for obesity in filtered_df['Obesity_Level'].unique()}
    
    fig = px.pie(pie_data, names='Obesity Class', values='Count',
                 color='Obesity Class', color_discrete_map=color_map,
                 hole=0.4)
    fig.update_traces(textinfo='percent+label', pull=[0.05]*len(pie_data))
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📈 Chart Description"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This donut chart shows the distribution of obesity classes in the selected population. The percentages represent the proportion of each obesity class.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row1_cols[1]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Gender & Obesity Class</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Sunburst chart
    sunburst_data = filtered_df.groupby(['Sex', 'Obesity_Level']).size().reset_index(name='Count')
    
    # Create a custom color dictionary for the sunburst
    sunburst_colors = {**GENDER_COLORS, **OBESITY_COLORS}
    
    fig = px.sunburst(sunburst_data, 
                      path=['Sex', 'Obesity_Level'], 
                      values='Count',
                      color_discrete_map=sunburst_colors)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This sunburst chart displays the hierarchical relationship between gender and obesity classes. Each ring represents a level in the hierarchy, with the inner ring showing gender distribution and the outer ring showing obesity classes within each gender.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Row 2: 2 graphs (Age distribution)
row2_cols = st.columns(2)

with row2_cols[0]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Age Distribution by Obesity Class</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Violin plot
    fig = px.violin(filtered_df, x='Obesity_Level', y='Age', color='Obesity_Level',
                    box=True, points="all", color_discrete_map=OBESITY_COLORS,
                    labels={"Obesity_Level": "Obesity Class", "Age": "Age (years)"})
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This violin plot shows the age distribution for each obesity class. The width of each violin represents the density of data points at that age, while the box plot inside shows the median, quartiles, and range.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2_cols[1]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Obesity Class by Age Group</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Stacked area chart
    age_obesity = pd.crosstab(filtered_df['Age_Group'], filtered_df['Obesity_Level'], normalize='index') * 100
    fig = px.area(age_obesity.reset_index(), x='Age_Group', y=age_obesity.columns,
                 title="", color_discrete_map=OBESITY_COLORS,
                 labels={"value": "Percentage (%)", "variable": "Obesity Class"})
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This stacked area chart shows the percentage distribution of obesity classes across different age groups. The area represents the proportion of each obesity class within each age group.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# SECTION 2: LIFESTYLE FACTORS
st.markdown("<div class='lifestyle-section-header'><span class='emoji'>🏃‍♂️</span> Lifestyle & Physical Activity</div>", unsafe_allow_html=True)
st.markdown("<div class='lifestyle-section-subtitle'>Analysis of how lifestyle choices and physical activity relate to obesity classes.</div>", unsafe_allow_html=True)

# Row 3: 3 graphs (Physical Activity, Transportation, Technology)
row3_cols = st.columns(3)

with row3_cols[0]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Physical Activity Impact</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Radar chart for physical activity
    activity_data = pd.crosstab(filtered_df['Physical_Activity_Level'], 
                                filtered_df['Obesity_Level'],
                                normalize='index') * 100
    
    if not activity_data.empty:
        activities = activity_data.index.tolist()
        
        fig = go.Figure()
        
        # Add traces for each obesity level with different colors (with transparency)
        colors = {
            'Insufficient Weight': f"rgba{tuple(int(OBESITY_COLORS['Insufficient Weight'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}",
            'Normal Weight': f"rgba{tuple(int(OBESITY_COLORS['Normal Weight'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}",
            'Overweight': f"rgba{tuple(int(OBESITY_COLORS['Overweight'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}",
            'Obesity': f"rgba{tuple(int(OBESITY_COLORS['Obesity'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}"
        }
        
        for obesity_level in activity_data.columns:
            values = activity_data[obesity_level].values.tolist()
            # Close the loop for radar chart
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=activities + [activities[0]],
                fill='toself',
                name=obesity_level,
                line=dict(color=OBESITY_COLORS.get(obesity_level, PRIMARY_COLORS['BLUE'])),
                fillcolor=OBESITY_COLORS.get(obesity_level, 'rgba(33, 150, 243, 0.3)')
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This radar chart displays the distribution of all obesity classes across different physical activity levels. Each colored area represents a different obesity class, showing how physical activity frequency correlates with each weight category. Higher percentages of Normal Weight in more active categories indicates a positive relationship between exercise and healthy weight.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row3_cols[1]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Transportation & Obesity</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Horizontal stacked bar chart for transportation
    transport_data = pd.crosstab(filtered_df['Transportation'], 
                               filtered_df['Obesity_Level'],
                               normalize='index') * 100
    
    if not transport_data.empty:
        # Reshape for plotting
        transport_df = transport_data.reset_index().melt(
            id_vars=['Transportation'],
            var_name='Obesity Class',
            value_name='Percentage'
        )
        
        fig = px.bar(transport_df, 
                    x='Percentage', 
                    y='Transportation',
                    color='Obesity Class',
                    orientation='h',
                    barmode='stack',
                    color_discrete_map=OBESITY_COLORS)
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This stacked horizontal bar chart shows the distribution of all obesity classes for each transportation mode. The bars are stacked to 100%, allowing you to compare the proportions of each obesity class across different transportation methods. Active transportation methods like walking and cycling typically show higher proportions of normal weight compared to more passive modes like automobile.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row3_cols[2]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Technology Usage Effect</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Grouped bar chart for technology usage
    tech_data = pd.crosstab(filtered_df['Technology_Usage'], 
                          filtered_df['Obesity_Level'])
    
    if not tech_data.empty:
        # Convert to percentage within each technology usage category
        tech_pct = tech_data.div(tech_data.sum(axis=1), axis=0) * 100
        
        # Prepare data for plotting
        plot_data = tech_pct.reset_index().melt(
            id_vars=['Technology_Usage'],
            var_name='Obesity Class',
            value_name='Percentage'
        )
        
        # Create grouped bar chart
        fig = px.bar(plot_data, 
                   x='Technology_Usage', 
                   y='Percentage',
                   color='Obesity Class',
                   barmode='group',
                   color_discrete_map=OBESITY_COLORS,
                   labels={'Technology_Usage': 'Daily Technology Usage', 
                          'Percentage': 'Percentage (%)'})
        
        fig.update_layout(
            xaxis_title="Daily Technology Usage",
            yaxis_title="Percentage (%)",
            legend_title="Obesity Class"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This grouped bar chart compares the distribution of obesity classes across different levels of daily technology usage. Each group represents a technology usage category, with colored bars showing the percentage of each obesity class. This visualization helps identify trends in how screen time correlates with different weight categories, not just obesity.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# SECTION 3: DIETARY PATTERNS
st.markdown("<div class='section-header'>🍎 Diet & Nutrition Patterns</div>", unsafe_allow_html=True)
st.write("Analysis of dietary habits and their relationship with obesity classes.")

# Row 4: 4 graphs in a row (Fast Food, Vegetable, Liquid, Snacking)
row4_cols = st.columns(4)

with row4_cols[0]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Fast Food Impact</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Stacked bar chart for fast food
    fast_food_data = pd.crosstab(filtered_df['Fast_Food'], 
                                filtered_df['Obesity_Level'])
    
    if not fast_food_data.empty:
        # Calculate percentages
        fast_food_pct = fast_food_data.div(fast_food_data.sum(axis=1), axis=0) * 100
        
        # Reshape for plotting
        plot_data = fast_food_pct.reset_index().melt(
            id_vars=['Fast_Food'],
            var_name='Obesity Class',
            value_name='Percentage'
        )
        
        fig = px.bar(plot_data, 
                    x='Fast_Food', 
                    y='Percentage', 
                    color='Obesity Class',
                    barmode='stack',
                    color_discrete_map=OBESITY_COLORS,
                    labels={'Fast_Food': 'Fast Food Consumption', 
                           'Percentage': 'Percentage (%)'})
        
        fig.update_layout(
            xaxis_title="Fast Food Consumption",
            yaxis_title="Percentage (%)",
            legend_title="Obesity Class",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This stacked bar chart shows the distribution of obesity classes between individuals who regularly consume fast food versus those who don't. The stacked presentation allows you to see how the proportions of all weight categories differ between the two groups, not just obesity rates.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row4_cols[1]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Vegetable Consumption</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Bubble chart for vegetable consumption
    veg_data = pd.crosstab(filtered_df['Vegetable_Consumption'], 
                           filtered_df['Obesity_Level'])
    
    if not veg_data.empty:
        veg_pct = pd.crosstab(filtered_df['Vegetable_Consumption'], 
                              filtered_df['Obesity_Level'],
                              normalize='index') * 100
        
        bubble_data = []
        for veg_type in veg_data.index:
            for obesity in veg_data.columns:
                bubble_data.append({
                    'Vegetable': veg_type,
                    'Obesity': obesity,
                    'Count': veg_data.loc[veg_type, obesity],
                    'Percentage': veg_pct.loc[veg_type, obesity]
                })
        
        bubble_df = pd.DataFrame(bubble_data)
        
        fig = px.scatter(bubble_df, 
                         x='Vegetable', 
                         y='Obesity',
                         size='Count', 
                         color='Percentage',
                         color_continuous_scale=[[0, LIGHT_COLORS['INDIGO_100']], [0.5, DARK_COLORS['INDIGO_300']], [1, PRIMARY_COLORS['INDIGO']]],
                         hover_name='Obesity',
                         hover_data=['Count', 'Percentage'])
        
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This bubble chart visualizes the relationship between vegetable consumption frequency and obesity classes. Bubble size represents the count of individuals, while color intensity shows the percentage within each vegetable consumption category.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row4_cols[2]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Liquid Intake</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Grouped bar chart for liquid intake
    liquid_data = pd.crosstab(filtered_df['Liquid_Intake'], 
                             filtered_df['Obesity_Level'])
    
    if not liquid_data.empty:
        # Calculate percentages
        liquid_pct = liquid_data.div(liquid_data.sum(axis=1), axis=0) * 100
        
        # Sort by liquid intake order
        order_map = {'< 1L': 0, '1-2L': 1, '> 2L': 2}
        liquid_pct['Order'] = liquid_pct.index.map(order_map)
        liquid_pct = liquid_pct.sort_values('Order')
        liquid_pct = liquid_pct.drop('Order', axis=1)
        
        # Convert to long format for grouped bar chart
        plot_data = liquid_pct.reset_index().melt(
            id_vars=['Liquid_Intake'],
            var_name='Obesity Class',
            value_name='Percentage'
        )
        
        fig = px.bar(plot_data, 
                    x='Liquid_Intake', 
                    y='Percentage', 
                    color='Obesity Class',
                    barmode='group',
                    color_discrete_map=OBESITY_COLORS,
                    labels={'Liquid_Intake': 'Daily Liquid Intake', 
                           'Percentage': 'Percentage (%)'})
        
        fig.update_layout(
            xaxis_title="Daily Liquid Intake",
            yaxis_title="Percentage (%)",
            legend_title="Obesity Class",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This grouped bar chart displays the distribution of obesity classes across different levels of daily liquid intake. By showing all weight categories side by side, you can observe how hydration habits correlate with different weight statuses, not just obesity. This provides a more nuanced view of how liquid intake relates to weight management.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row4_cols[3]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Snacking Frequency</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Stacked area chart for snacking
    snack_data = pd.crosstab(filtered_df['Snacking'], 
                            filtered_df['Obesity_Level'])
    
    if not snack_data.empty:
        # Calculate percentages
        snack_pct = snack_data.div(snack_data.sum(axis=1), axis=0) * 100
        
        # Sort by snacking frequency order
        order_map = {'Rarely': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        snack_pct['Order'] = snack_pct.index.map(order_map)
        snack_pct = snack_pct.sort_values('Order')
        snack_pct = snack_pct.drop('Order', axis=1)
        
        # Convert to long format for area chart
        plot_data = snack_pct.reset_index().melt(
            id_vars=['Snacking'],
            var_name='Obesity Class',
            value_name='Percentage'
        )
        
        fig = px.area(plot_data, 
                     x='Snacking', 
                     y='Percentage', 
                     color='Obesity Class',
                     color_discrete_map=OBESITY_COLORS,
                     labels={'Snacking': 'Snacking Frequency', 
                            'Percentage': 'Percentage (%)'})
        
        fig.update_layout(
            xaxis_title="Snacking Frequency",
            yaxis_title="Percentage (%)",
            legend_title="Obesity Class",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This stacked area chart displays the distribution of all obesity classes across different snacking frequencies. The areas show how the proportions of each weight category change as snacking frequency increases, providing a more complete picture than just tracking obesity rates alone.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# SECTION 4: RISK FACTORS AND CORRELATIONS
st.markdown("<div class='section-header'>🔄 Risk Factors & Relationships</div>", unsafe_allow_html=True)
st.write("Analysis of risk factors, family history, and correlations between health factors.")

# Row 5: 2 graphs (Family History, Smoking)
row5_cols = st.columns(2)

with row5_cols[0]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Family History Impact</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Grouped bar chart for family history
    fam_data = pd.crosstab(filtered_df['Family_History'], filtered_df['Obesity_Level'])
    
    if not fam_data.empty:
        fig = px.bar(fam_data.reset_index(), x='Family_History', y=fam_data.columns,
                    barmode='group', color_discrete_map=OBESITY_COLORS,
                    labels={"value": "Count", "variable": "Obesity Class"})
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This grouped bar chart compares the distribution of obesity classes between individuals with and without a family history of obesity. It helps visualize the genetic component of obesity risk.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row5_cols[1]:
    st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title-container'>Smoking Status Relationship</div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-content'>", unsafe_allow_html=True)
    
    # Treemap for smoking
    smoke_data = filtered_df.groupby(['Smoker', 'Obesity_Level']).size().reset_index(name='Count')
    
    if not smoke_data.empty:
        # Create a custom color dictionary for the treemap
        treemap_colors = {**{'Yes': PRIMARY_COLORS['RED'], 'No': PRIMARY_COLORS['BLUE']}, **OBESITY_COLORS}
        
        fig = px.treemap(smoke_data, 
                        path=['Smoker', 'Obesity_Level'], 
                        values='Count',
                        color='Obesity_Level',
                        color_discrete_map=treemap_colors)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for this visualization.")
    
    with st.expander("📈 Chart Insights"):
        st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This treemap visualizes the hierarchical distribution of obesity classes within smoking status groups. The size of each rectangle represents the count of individuals in that category.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Row 6: 1 graph (Correlation Heatmap - Full width)
st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
st.markdown("<div class='graph-title-container'>Correlation Between Health Factors</div>", unsafe_allow_html=True)
st.markdown("<div class='graph-content'>", unsafe_allow_html=True)

# Correlation heatmap
corr_df = filtered_df.copy()
order_mapping = {
    'Obesity_Level': {'Insufficient Weight': 1, 'Normal Weight': 2, 'Overweight': 3, 'Obesity': 4},
    'Physical_Activity_Level': {'No Activity': 1, '1-2 days/week': 2, '3-4 days/week': 3, '5-6 days/week': 4, '6+ days/week': 5},
    'Technology_Usage': {'0-2 hours': 1, '3-5 hours': 2, '5+ hours': 3},
    'Vegetable_Consumption': {'Rarely': 1, 'Sometimes': 2, 'Always': 3},
    'Liquid_Intake': {'< 1L': 1, '1-2L': 2, '> 2L': 3},
    'Meal_Count': {'1-2 meals': 1, '3 meals': 2, 'More than 3': 3},
    'Snacking': {'Rarely': 1, 'Sometimes': 2, 'Frequently': 3, 'Always': 4}
}
binary_mapping = {
    'Sex': {'Male': 0, 'Female': 1},
    'Fast_Food': {'No': 0, 'Yes': 1},
    'Family_History': {'No': 0, 'Yes': 1},
    'Tracks_Calories': {'No': 0, 'Yes': 1},
    'Smoker': {'No': 0, 'Yes': 1}
}

for col, mapping in order_mapping.items():
    corr_df[col] = corr_df[col].map(mapping)
for col, mapping in binary_mapping.items():
    corr_df[col] = corr_df[col].map(mapping)

corr_df = pd.get_dummies(corr_df, columns=['Transportation'], prefix=['Transport'])
numeric_cols = corr_df.select_dtypes(include=['int64', 'float64']).columns

if not corr_df[numeric_cols].empty:
    correlation = corr_df[numeric_cols].corr()
    
    # Create better labels for the heatmap
    better_labels = {
        'Obesity_Level': 'Obesity Level',
        'Age': 'Age',
        'Sex': 'Gender (1=Female)',
        'Fast_Food': 'Fast Food (1=Yes)',
        'Family_History': 'Family History (1=Yes)',
        'Tracks_Calories': 'Tracks Calories (1=Yes)',
        'Smoker': 'Smoker (1=Yes)',
        'Physical_Activity_Level': 'Physical Activity',
        'Technology_Usage': 'Tech Usage Hours',
        'Vegetable_Consumption': 'Vegetable Frequency',
        'Liquid_Intake': 'Liquid Intake',
        'Meal_Count': 'Meals per Day',
        'Snacking': 'Snacking Frequency'
    }
    
    # Filter for key variables only
    key_vars = ['Obesity_Level', 'Age', 'Sex', 'Fast_Food', 'Family_History', 
                'Physical_Activity_Level', 'Technology_Usage', 'Vegetable_Consumption',
                'Liquid_Intake', 'Snacking']
    
    # Only keep correlation with obesity level and selected key variables
    correlation_filtered = correlation.loc[key_vars, key_vars]
    
    # Rename for better labels
    correlation_filtered = correlation_filtered.rename(index=better_labels, columns=better_labels)
    
    fig = px.imshow(correlation_filtered,
                    labels=dict(color="Correlation Coefficient"),
                    color_continuous_scale=[[0, LIGHT_COLORS['INDIGO_50']], 
                                           [0.25, LIGHT_COLORS['INDIGO_100']], 
                                           [0.5, DARK_COLORS['INDIGO_300']],
                                           [0.75, DARK_COLORS['INDIGO_400']], 
                                           [1, PRIMARY_COLORS['INDIGO']]],
                    text_auto='.2f')
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data for correlation analysis.")

with st.expander("📈 Chart Insights"):
    st.markdown("<div class='insights-box'><div class='insights-title'>What This Shows</div>This heatmap displays the correlation coefficients between key health factors. Values close to 1 (dark blue) indicate strong positive correlations, values close to -1 (light blue) indicate strong negative correlations, and values close to 0 indicate little correlation. This visualization helps identify which factors have the strongest relationships with obesity levels.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Dashboard created for healthcare analysis and decision support in obesity prevention and intervention.</i><br>
    <small>Data updated: June 2025</small>
</div>
""", unsafe_allow_html=True)
