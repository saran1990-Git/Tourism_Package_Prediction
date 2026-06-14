import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="saranappu1990/Tourism-Package-Prediction", filename="best_Tourism_Prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a purchasing tourisum package based on its parameters.
Please enter the customer details below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=80, value=24)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=300, value=12)
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=0)
NumberOfTrips = st.number_input("Number of Trips", min_value=0, max_value=10, value=0)
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Standard", "Basic","King", "Super Deluxe"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
Passport = st.selectbox("Passport", ["0", "1"])
OwnCar = st.selectbox("Own Car", ["0", "1"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=100000, value=50000)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfTrips': NumberOfTrips,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'Passport': int(Passport),
    'OwnCar': int(OwnCar),
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'PitchSatisfactionScore': PitchSatisfactionScore
}])

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The customer is likely to purchase the tourism package.")
    else:
        st.warning("The customer is unlikely to purchase the tourism package.")
