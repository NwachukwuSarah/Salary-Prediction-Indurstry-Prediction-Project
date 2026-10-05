import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Load the saved model bundle
# (contains the best regression model + every preprocessing
# step it needs: the one-hot encoder, the ordinal encoder
# and the scaler)
# ---------------------------------------------------------
bundle = joblib.load("best_salary_regression_model.pkl")

model = bundle["model"]
model_name = bundle["model_name"]
ohe = bundle["ohe"]
ohe_cols = bundle["ohe_cols"]
ordinal_encoder = bundle["ordinal_encoder"]
degree_order = bundle["degree_order"]
preprocessor = bundle["preprocessor"]
feature_columns = bundle["feature_columns"]
raw_input_columns = bundle["raw_input_columns"]

# Dropdown options come straight from what the encoder saw during training
ohe_categories = dict(zip(ohe_cols, ohe.categories_))

# ---------------------------------------------------------
# Page
# ---------------------------------------------------------
st.title("Nigerian Graduate Salary Predictor")
st.write(
    "Fill in a graduate's details below and click **Predict Salary** "
    f"to estimate their monthly salary. (Model used: {model_name})"
)

geopolitical_zone = st.selectbox("Geopolitical Zone", ohe_categories["Geopolitical_Zone"])
gender = st.selectbox("Gender", ohe_categories["Gender"])
age = st.number_input("Age", min_value=18, max_value=70, value=25)
ethnicity = st.selectbox("Ethnicity", ohe_categories["Ethnicity"])
disability_status = st.selectbox("Disability Status", ohe_categories["Disability_Status"])
residential_setting = st.selectbox("Residential Setting", ohe_categories["Residential_Setting"])
university_attended = st.selectbox("University Attended", ohe_categories["University_Attended"])
degree = st.selectbox("Field of Study (Degree)", ohe_categories["Degree"])
class_of_degree = st.selectbox("Class of Degree", degree_order)
nysc_status = st.selectbox("NYSC Status", ohe_categories["NYSC_Status"])
additional_qualification = st.selectbox("Additional Qualification", ohe_categories["Additional_Qualification"])
employment_status = st.selectbox("Employment Status", ohe_categories["Employment_Status"])
industry = st.selectbox("Industry", ohe_categories["Industry"])
prof_certs_count = st.number_input("Number of Professional Certificates", min_value=0, max_value=20, value=0)
years_work_experience = st.number_input("Years of Work Experience", min_value=0, max_value=50, value=1)

if st.button("Predict Salary"):

    # Put all the raw inputs into a single-row DataFrame,
    # in the same column order used during training.
    raw_input = pd.DataFrame([{
        "Geopolitical_Zone": geopolitical_zone,
        "Gender": gender,
        "Age": age,
        "Ethnicity": ethnicity,
        "Disability_Status": disability_status,
        "Residential_Setting": residential_setting,
        "University_Attended": university_attended,
        "Degree": degree,
        "Class_of_Degree": class_of_degree,
        "NYSC_Status": nysc_status,
        "Additional_Qualification": additional_qualification,
        "Employment_Status": employment_status,
        "Industry": industry,
        "Prof_Certs_Count": prof_certs_count,
        "Years_Work_Experience": years_work_experience
    }])[raw_input_columns]

    # Step 1: one-hot encode the same columns that were one-hot encoded during training
    encoded = ohe.transform(raw_input[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(ohe_cols),
        index=raw_input.index
    )

    processed = pd.concat([raw_input.drop(columns=ohe_cols), encoded_df], axis=1)

    # Step 2: ordinal encode Class_of_Degree the same way as training
    processed["Class_of_Degree"] = ordinal_encoder.transform(processed[["Class_of_Degree"]])

    # Step 3: match the exact column order the scaler was fit on
    processed = processed[feature_columns]

    # Step 4: scale the numeric columns the same way as training
    processed_scaled = preprocessor.transform(processed)

    # Step 5: predict
    predicted_salary = model.predict(processed_scaled)[0]

    st.success(f"Predicted Monthly Salary: ₦{predicted_salary:,.2f}")
