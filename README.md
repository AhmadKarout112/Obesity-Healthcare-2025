# 🏥 Obesity Healthcare Analytics Dashboard 2025

An interactive Streamlit dashboard exploring the relationships between demographic characteristics, lifestyle factors, and obesity classes.

## Dashboard Overview

This dashboard provides comprehensive insights into how various demographic and lifestyle factors influence obesity classes. It enables healthcare professionals, researchers, and policymakers to understand patterns and make data-driven decisions for prevention and intervention strategies.

## Key Features

- **Interactive Filtering**: Filter data by age, gender, physical activity level, and obesity class.
- **Key Insight Cards**: At-a-glance summary statistics about the dataset.
- **Comprehensive Visualizations**: Detailed analysis through multiple tabs:
  - Main Analysis: Distribution of obesity classes and demographic patterns
  - Lifestyle Factors: Impact of fast food, family history, smoking, and transportation
  - Dietary Patterns: Effects of vegetable consumption, meal frequency, snacking, and hydration
  - Technology & Activity: Relationship between technology usage and physical activity
  - Correlation Analysis: Heatmap showing relationships between variables
- **Dedicated Legends Panel**: Clear reference for all color schemes used in visualizations
- **Key Insights & Recommendations**: Evidence-based suggestions for interventions

## Dataset

The dataset contains information about individuals including:
- Demographic details (Sex, Age)
- Lifestyle factors (Fast Food Consumption, Physical Activity, Technology Usage, etc.)
- Dietary habits (Vegetable Consumption, Meal Frequency, Snacking, etc.)
- Obesity classification (Insufficient Weight, Normal Weight, Overweight, Obesity)

## How to Run the Dashboard

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

## Technologies Used

- Streamlit
- Pandas
- Plotly
- NumPy
- Seaborn
- Matplotlib

## Data Source

The dataset used in this dashboard is sourced from obesity research conducted in 2025.

---

*Dashboard created for healthcare analysis and decision support in obesity prevention and intervention.*
