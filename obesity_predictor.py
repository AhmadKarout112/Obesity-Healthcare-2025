import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st
import plotly.graph_objects as go

def load_and_preprocess_data():
    """Load and preprocess the data for model training"""
    df = pd.read_excel('data/Obesity_Dataset.xlsx')
    
    # Create target variable (simplified to a binary classification for demonstration)
    # 1: Insufficient Weight, 2: Normal Weight, 3: Overweight, 4: Obesity
    df['Target'] = df['Class'].map({
        1: 'Not Obese',  # Insufficient Weight
        2: 'Not Obese',  # Normal Weight
        3: 'Not Obese',  # Overweight
        4: 'Obese'       # Obesity
    })
    
    # Features for the model
    features = [
        'Age', 'Sex', 'Physical_Excercise', 'Schedule_Dedicated_to_Technology',
        'Type_of_Transportation_Used', 'Frequency_of_Consuming_Vegetables',
        'Liquid_Intake_Daily', 'Consumption_of_Fast_Food', 'Overweight_Obese_Family',
        'Calculation_of_Calorie_Intake', 'Smoking', 'Number_of_Main_Meals_Daily',
        'Food_Intake_Between_Meals'
    ]
    
    X = df[features]
    y = df['Target']
    
    return X, y

def build_model():
    """Build the machine learning pipeline"""
    # Define categorical and numerical features
    categorical_features = [
        'Sex', 'Physical_Excercise', 'Schedule_Dedicated_to_Technology',
        'Type_of_Transportation_Used', 'Frequency_of_Consuming_Vegetables',
        'Liquid_Intake_Daily', 'Consumption_of_Fast_Food', 'Overweight_Obese_Family',
        'Calculation_of_Calorie_Intake', 'Smoking', 'Number_of_Main_Meals_Daily',
        'Food_Intake_Between_Meals'
    ]
    
    numerical_features = ['Age']
    
    # Create preprocessing steps for categorical and numerical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and configure the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return model

def train_model():
    """Train the obesity prediction model and save it"""
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save the model
    joblib.dump(model, 'obesity_prediction_model.pkl')
    
    return model, accuracy, report, conf_matrix

def get_user_input():
    """Collect user input for prediction"""
    st.subheader("Enter Your Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 70, 30)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        physical_activity = st.selectbox("Physical Activity", 
                                        options=[1, 2, 3, 4, 5],
                                        format_func=lambda x: {
                                            1: "No Activity", 
                                            2: "1-2 days/week",
                                            3: "3-4 days/week", 
                                            4: "5-6 days/week",
                                            5: "6+ days/week"
                                        }[x])
        tech_usage = st.selectbox("Technology Usage (hours)", 
                                options=[1, 2, 3],
                                format_func=lambda x: {
                                    1: "0-2 hours", 
                                    2: "3-5 hours", 
                                    3: "5+ hours"
                                }[x])
        transportation = st.selectbox("Primary Transportation", 
                                    options=[1, 2, 3, 4, 5],
                                    format_func=lambda x: {
                                        1: "Automobile", 
                                        2: "Motorbike", 
                                        3: "Bike",
                                        4: "Public Transport", 
                                        5: "Walking"
                                    }[x])
        vegetable_consumption = st.selectbox("Vegetable Consumption", 
                                           options=[1, 2, 3],
                                           format_func=lambda x: {
                                               1: "Rarely", 
                                               2: "Sometimes", 
                                               3: "Always"
                                           }[x])
    
    with col2:
        liquid_intake = st.selectbox("Daily Liquid Intake", 
                                   options=[1, 2, 3],
                                   format_func=lambda x: {
                                       1: "< 1L", 
                                       2: "1-2L", 
                                       3: "> 2L"
                                   }[x])
        fast_food = st.selectbox("Consume Fast Food Regularly", 
                               options=[1, 2],
                               format_func=lambda x: "Yes" if x == 1 else "No")
        family_history = st.selectbox("Family History of Obesity", 
                                    options=[1, 2],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
        tracks_calories = st.selectbox("Track Calorie Intake", 
                                     options=[1, 2],
                                     format_func=lambda x: "Yes" if x == 1 else "No")
        smoker = st.selectbox("Do You Smoke", 
                            options=[1, 2],
                            format_func=lambda x: "Yes" if x == 1 else "No")
        meal_count = st.selectbox("Main Meals per Day", 
                                options=[1, 2, 3],
                                format_func=lambda x: {
                                    1: "1-2 meals", 
                                    2: "3 meals", 
                                    3: "More than 3"
                                }[x])
        snacking = st.selectbox("Snacking Between Meals", 
                              options=[1, 2, 3, 4],
                              format_func=lambda x: {
                                  1: "Rarely", 
                                  2: "Sometimes", 
                                  3: "Frequently", 
                                  4: "Always"
                              }[x])
    
    # Create a dictionary with the input values
    user_data = {
        'Age': age,
        'Sex': gender,
        'Physical_Excercise': physical_activity,
        'Schedule_Dedicated_to_Technology': tech_usage,
        'Type_of_Transportation_Used': transportation,
        'Frequency_of_Consuming_Vegetables': vegetable_consumption,
        'Liquid_Intake_Daily': liquid_intake,
        'Consumption_of_Fast_Food': fast_food,
        'Overweight_Obese_Family': family_history,
        'Calculation_of_Calorie_Intake': tracks_calories,
        'Smoking': smoker,
        'Number_of_Main_Meals_Daily': meal_count,
        'Food_Intake_Between_Meals': snacking
    }
    
    return pd.DataFrame([user_data])

def predict_obesity(input_data):
    """Make a prediction based on user input"""
    try:
        # Try to load the pre-trained model
        model = joblib.load('obesity_prediction_model.pkl')
    except:
        # If model doesn't exist, train it
        model, _, _, _ = train_model()
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    return prediction[0], probability[0]

def display_model_metrics():
    """Display model performance metrics"""
    try:
        # Try to load the pre-trained model and evaluate
        model = joblib.load('obesity_prediction_model.pkl')
        X, y = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
    except:
        # If model doesn't exist, train it and get metrics
        _, accuracy, report, conf_matrix = train_model()
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        # Feature importance (if available)
        try:
            feature_importances = pd.Series(
                model.named_steps['classifier'].feature_importances_,
                index=model.named_steps['preprocessor'].get_feature_names_out()
            ).sort_values(ascending=False)
            
            st.subheader("Top 10 Most Important Features")
            st.bar_chart(feature_importances.head(10))
        except:
            st.write("Feature importance not available for this model.")
    
    with col2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    confusion_df = pd.DataFrame(
        conf_matrix, 
        index=["Actual: Not Obese", "Actual: Obese"],
        columns=["Predicted: Not Obese", "Predicted: Obese"]
    )
    st.dataframe(confusion_df)

def render_prediction_tab():
    st.title("🔮 Obesity Risk Prediction")
    
    st.markdown("""
    This tool uses machine learning to predict your risk of obesity based on lifestyle factors and personal characteristics.
    Fill in the form below to get your personalized risk assessment.
    """)
    
    # Show model information in an expander
    with st.expander("About the Prediction Model"):
        st.markdown("""
        ### Model Information
        
        This tool uses a **Random Forest Classifier** trained on obesity data to predict the risk of obesity.
        
        The model considers factors such as:
        - Age and gender
        - Physical activity levels
        - Technology usage habits
        - Transportation methods
        - Dietary patterns (vegetable intake, liquid consumption, fast food)
        - Family history
        - Meal frequency and snacking habits
        
        The prediction is based on statistical patterns identified in the data and should be used as a general guideline, not as medical advice.
        """)
        
        # Show model metrics
        if st.checkbox("Show Model Performance Metrics"):
            display_model_metrics()
    
    # Get user input
    user_data = get_user_input()
    
    # Make prediction when the button is clicked
    if st.button("Predict My Obesity Risk"):
        with st.spinner("Analyzing your data..."):
            prediction, probability = predict_obesity(user_data)
            
            # Display the prediction
            st.subheader("Prediction Results")
            
            # Create columns for the result display
            col1, col2 = st.columns(2)
            
            # Display prediction result with risk categories
            with col1:
                # Get the obesity risk probability
                obesity_prob = probability[1] * 100 if prediction == "Obese" else probability[0] * 100
                
                # Determine risk category based on probability percentage
                if obesity_prob < 10:
                    risk_category = "Minimal"
                    bg_color = "rgba(200, 200, 200, 0.1)"
                    border_color = "#9e9e9e"
                    text_color = "#616161"
                elif obesity_prob >= 10 and obesity_prob < 50:
                    risk_category = "Low"
                    bg_color = "rgba(76, 175, 80, 0.1)"
                    border_color = "#4caf50"
                    text_color = "#388e3c"
                elif obesity_prob >= 50 and obesity_prob < 70:
                    risk_category = "Medium"
                    bg_color = "rgba(255, 193, 7, 0.1)"
                    border_color = "#ffc107"
                    text_color = "#ff8f00"
                elif obesity_prob >= 70 and obesity_prob < 90:
                    risk_category = "High"
                    bg_color = "rgba(255, 87, 34, 0.1)"
                    border_color = "#ff5722"
                    text_color = "#e64a19"
                else:
                    risk_category = "Very High"
                    bg_color = "rgba(244, 67, 54, 0.1)"
                    border_color = "#f44336"
                    text_color = "#d32f2f"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border_color};">
                    <h3 style="color: {text_color};">{risk_category} Risk of Obesity</h3>
                    <p>Based on the information provided, our model predicts you have a <b>{risk_category.lower()}</b> risk of obesity.</p>
                    <p>Risk Score: <b>{obesity_prob:.1f}%</b></p>
                    <p>Risk Category:
                        <span style="display: inline-block; margin-left: 10px; padding: 5px 10px; background-color: {border_color}; color: white; border-radius: 15px; font-weight: bold;">
                            {risk_category}
                        </span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a visual risk gauge
            st.markdown("<h4>Risk Level Visualization</h4>", unsafe_allow_html=True)
            
            # Create a gauge chart to visualize the risk level
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=obesity_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Obesity Risk Level", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                    'bar': {'color': border_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 10], 'color': 'rgba(200, 200, 200, 0.3)'},
                        {'range': [10, 50], 'color': 'rgba(76, 175, 80, 0.3)'},
                        {'range': [50, 70], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [70, 90], 'color': 'rgba(255, 87, 34, 0.3)'},
                        {'range': [90, 100], 'color': 'rgba(244, 67, 54, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': obesity_prob
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=50, b=10),
                font=dict(family="Segoe UI, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a risk level scale legend
            st.markdown("""
            <div style="display: flex; justify-content: space-between; margin-top: -20px; margin-bottom: 20px;">
                <div style="text-align: center; flex: 1;">
                    <div style="background-color: rgba(200, 200, 200, 0.3); height: 10px; border-radius: 5px;"></div>
                    <div style="font-size: 12px; margin-top: 5px;">Minimal (0-10%)</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="background-color: rgba(76, 175, 80, 0.3); height: 10px; border-radius: 5px;"></div>
                    <div style="font-size: 12px; margin-top: 5px;">Low (10-50%)</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="background-color: rgba(255, 193, 7, 0.3); height: 10px; border-radius: 5px;"></div>
                    <div style="font-size: 12px; margin-top: 5px;">Medium (50-70%)</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="background-color: rgba(255, 87, 34, 0.3); height: 10px; border-radius: 5px;"></div>
                    <div style="font-size: 12px; margin-top: 5px;">High (70-90%)</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="background-color: rgba(244, 67, 54, 0.3); height: 10px; border-radius: 5px;"></div>
                    <div style="font-size: 12px; margin-top: 5px;">Very High (90%+)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations based on the risk category
            with col2:
                st.subheader("Personalized Recommendations")
                
                # Get the top risk factors from the user input
                high_risk_factors = []
                
                if user_data['Physical_Excercise'].values[0] <= 2:
                    high_risk_factors.append("Low physical activity")
                    
                if user_data['Consumption_of_Fast_Food'].values[0] == 1:
                    high_risk_factors.append("Regular fast food consumption")
                    
                if user_data['Frequency_of_Consuming_Vegetables'].values[0] == 1:
                    high_risk_factors.append("Low vegetable intake")
                    
                if user_data['Liquid_Intake_Daily'].values[0] == 1:
                    high_risk_factors.append("Low liquid intake")
                    
                if user_data['Food_Intake_Between_Meals'].values[0] >= 3:
                    high_risk_factors.append("Frequent snacking")
                
                if user_data['Type_of_Transportation_Used'].values[0] <= 2:
                    high_risk_factors.append("Sedentary transportation")
                
                # Display header based on risk category
                if risk_category == "Minimal":
                    st.markdown(f"""
                    <div style="background-color: rgba(200, 200, 200, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="color: #616161; margin: 0;">✨ Minimal Risk Recommendations</h4>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_category == "Low":
                    st.markdown(f"""
                    <div style="background-color: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="color: #388e3c; margin: 0;">✅ Low Risk Recommendations</h4>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_category == "Medium":
                    st.markdown(f"""
                    <div style="background-color: rgba(255, 193, 7, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="color: #ff8f00; margin: 0;">⚠️ Medium Risk Recommendations</h4>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_category == "High":
                    st.markdown(f"""
                    <div style="background-color: rgba(255, 87, 34, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="color: #e64a19; margin: 0;">🚨 High Risk Recommendations</h4>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Very High
                    st.markdown(f"""
                    <div style="background-color: rgba(244, 67, 54, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="color: #d32f2f; margin: 0;">⛔ Very High Risk Recommendations</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display the risk factors and recommendations
                if high_risk_factors:
                    st.markdown("#### Key Risk Factors:")
                    for factor in high_risk_factors:
                        st.markdown(f"- {factor}")
                    
                    st.markdown("#### Recommendations:")
                    if "Low physical activity" in high_risk_factors:
                        st.markdown("- Aim for at least 150 minutes of moderate exercise per week")
                        st.markdown("- Start with short walks and gradually increase intensity")
                    
                    if "Regular fast food consumption" in high_risk_factors:
                        st.markdown("- Limit fast food to once per week")
                        st.markdown("- Prepare home-cooked meals with fresh ingredients")
                    
                    if "Low vegetable intake" in high_risk_factors:
                        st.markdown("- Add vegetables to at least two meals per day")
                        st.markdown("- Try a variety of vegetables to find ones you enjoy")
                    
                    if "Low liquid intake" in high_risk_factors:
                        st.markdown("- Aim for at least 2 liters of water daily")
                        st.markdown("- Carry a water bottle as a reminder to stay hydrated")
                    
                    if "Frequent snacking" in high_risk_factors:
                        st.markdown("- Replace unhealthy snacks with fruits, nuts, or yogurt")
                        st.markdown("- Establish regular meal times to reduce snacking")
                    
                    if "Sedentary transportation" in high_risk_factors:
                        st.markdown("- Consider walking or cycling for short trips")
                        st.markdown("- Park farther away to increase daily steps")
                else:
                    st.markdown("#### You're on the right track!")
                    st.markdown("- Continue maintaining your current healthy habits")
                    st.markdown("- Regular check-ups with healthcare providers are still recommended")
        
        # Disclaimer
        st.markdown("""
        ---
        **Disclaimer**: This prediction is for informational purposes only and should not replace professional medical advice. 
        Please consult with a healthcare provider for proper diagnosis and treatment plans.
        """)
