import streamlit as st
from xgb_pred import getApplication 

# Title and description of the project
st.title("Materials Application Predictor using ML methods")

st.header("About the Project")
st.subheader("Description")
st.text("""
The primary objective of this project is to develop a machine learning model
that can predict the potential applications of materials based on their properties.
The project involves generating synthetic data, preprocessing the dataset,
feature engineering, and comparing different ML models.
""")

# Tasks to enhance model efficiency
st.subheader("Tasks to Enhance Model Efficiency")
st.text("""
1. **Synthetic Data Generation**: 
   This can be useful for increasing the diversity of the dataset and handling imbalanced datasets. By augmenting the data, we can improve the robustness of the model and make it resilient to real-world use cases.

2. **Data Pre-Processing**: 
   Data analytics and removing unwanted data points (cleaning data) help the model prevent overfitting and remove noise. It also helps in predicting the variance of the input parameters with the output.

3. **Experimenting with Different Supervised and Unsupervised Models**: 
   Models like Support Vector Machines (SVM) and Random Forest Regression can be used to compare the efficiency of the model.

4. **Researching Real-World Use-Cases**: 
   Investigating the applications of materials in today’s industries to understand practical challenges and refine the model's capabilities.
""")

# Possible Models Section
st.subheader("Possible Models to be Used")
st.text("""
- Decision Tree (Supervised)
- XGBoost (Supervised)
- K-means Clustering (Unsupervised)
- Support Vector Machines (Supervised)
""")

# Material property input boxes for user
st.header("Enter Material Properties to Predict Application")

# Input fields for material properties
float_Su = st.number_input('Su (Ultimate Tensile Strength)', value=0.0, step=0.1)
float_Sy = st.number_input('Sy (Yield Strength)', value=0.0, step=0.1)
float_E = st.number_input('E (Elastic Modulus)', value=0.0, step=0.1)
float_G = st.number_input('G (Shear Modulus)', value=0.0, step=0.1)
float_mu = st.number_input('mu (Poisson\'s Ratio)', value=0.0, step=0.01)
float_Ro = st.number_input('Ro (Density)', value=0.0, step=0.01)

# Button to predict the application
if st.button('Predict Application'):
    # Call the getApplication function and make prediction
    predicted_application = getApplication(float_Su, float_Sy, float_E, float_G, float_mu, float_Ro)
    
    # Display the predicted application in a well-designed box
    st.markdown(f"""
    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px;">
        <h3 style="color: #000080;">Predicted Material Application</h3>
        <p style="font-size: 18px; color: #333333;">The material is predicted to be suitable for: <strong>{predicted_application}</strong></p>
    </div>
    """, unsafe_allow_html=True)
