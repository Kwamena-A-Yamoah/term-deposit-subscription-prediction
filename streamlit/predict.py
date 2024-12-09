import streamlit as st
import joblib
import pandas as pd 
import os
import datetime


# =========================================================================================================== Page Setup

st.set_page_config(
    page_title= 'Predict',
    layout= 'wide'
)

# =========================================================================================================== Load Pipelines

# Create function to load models from the project
@st.cache_resource(show_spinner='Model loading')    # A decorator for loading
def load_rf_pipeline():
    pipeline = joblib.load('./models/random_forest_model.pk1')
    return pipeline

@st.cache_resource(show_spinner='Model loading')
def load_svc_pipeline():
    pipeline = joblib.load('./models\svc_model.pkl')
    return pipeline

# =========================================================================================================== Function

# A function to define a 'Selexbox' for the models, load the select model and encoder
def select_model():
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox('Select a Model', options= ['Random Forest', 'SVC'], key= 'selected_model')
    with col2:
        pass
    
    if st.session_state['selected_model'] == 'Random Forest':
        pipeline = load_rf_pipeline()
    else:
        pipeline = load_svc_pipeline()
    
    # This is the encoder used to transform "y_train" and "y_test"
    encoder = joblib.load('./models/encoder.joblib')   
    return pipeline, encoder

# =========================================================================================================== For Session State

# This is for the prediction and probabiliy in the session state when nothing has been run
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'probability' not in st.session_state:
    st.session_state.probability = None
    
# =========================================================================================================== Function

# A function to process the input data and make prediction
def make_prediction(pipeline, encoder):
    gender = st.session_state['gender']
    SeniorCitizen = st.session_state['senior_citizen']
    Partner = st.session_state['partner']
    Dependents = st.session_state['dependants']
    tenure = st.session_state['tenure']
    PhoneService = st.session_state['phone_service']
    MultipleLines = st.session_state['multiple_lines']
    InternetService = st.session_state['internet_service']
    OnlineSecurity = st.session_state['online_security']
    OnlineBackup = st.session_state['online_backup']
    DeviceProtection = st.session_state['device_protection']
    TechSupport = st.session_state['tech_support']
    StreamingTV = st.session_state['tv_stream']
    StreamingMovies = st.session_state['stream_movies']
    Contract = st.session_state['contract']
    PaperlessBilling = st.session_state['paperless_billing']
    PaymentMethod = st.session_state['payment_method']
    MonthlyCharges = st.session_state['monthly_charges']
    TotalCharges = st.session_state['total_charges']
    
    column = ['gender', 'SeniorCitizen', 'Partner','Dependents','tenure',
              'PhoneService', 'MultipleLines', 'InternetService',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
              'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
    
    data = [[gender, SeniorCitizen, Partner, Dependents,
             tenure, PhoneService, MultipleLines, InternetService,
             OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
             StreamingTV, StreamingMovies, Contract, PaperlessBilling,
             PaymentMethod, MonthlyCharges, TotalCharges]]
    
    # create a dataframe
    df = pd.DataFrame(data, columns= column)
    
    # Create the 'Prediction time' and 'Model used' columns
    df['Prediction time'] = datetime.date.today()
    df['Model used'] = st.session_state['selected_model']
    
    # Save prediction info into a history, history.csv
    df.to_csv('./data/history.csv', mode='a', 
              header=not os.path.exists('./data/history.csv'), index=False)
    
    # Make prediction
    pred = pipeline.predict(df)
    pred = int(pred[0])
    prediction = encoder.inverse_transform([pred])
    
    # Get probabilities
    probability = pipeline.predict_proba(df)
    probability = probability[0]
    
    st.write("Debug: Final prediction:", prediction[0])
    st.write("Debug: Final probability:", probability)
  
    st.session_state['predictions'] = prediction[0]
    st.session_state['probability'] = probability
    
    return prediction[0], probability
    print(prediction, probability)

# =========================================================================================================== Function (display)

# Write the Display form function
def display_form():
    
    # Display the model options
    pipeline, encoder = select_model()
    
    # The display form
    with st.form('input-feature'):
        col1, col2, col3 =st.columns(3)
    
        with col1:
            st.write("#####")
            st.selectbox('Gender', ['Male', 'Female'], key='gender')
            st.selectbox('Type of Internet service', ['DSL', 'Fiber optic', 'No'], key='internet_service')
            st.selectbox('Type of Contract', ['Month-to-month', 'One year', 'Two year'], key='contract')
            st.selectbox('Payment method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                            'Credit card (automatic)'], key='payment_method')
            st.slider('Tenure', min_value=0, max_value=72, key='tenure')
            st.number_input('Monthly charges', min_value=0, max_value=120, key='monthly_charges')
            st.number_input('Total charges', min_value=0.00, max_value=8684.80, key='total_charges')
        with col2:
            st.write("Please input the customer's details")
            st.selectbox('Senior Citizen', ['Yes','No'], key='senior_citizen')
            st.selectbox('Partner', ['Yes','No'], key='partner')
            st.selectbox('Dependants', ['Yes','No'], key='dependants')
            st.selectbox('Phone service', ['Yes','No'], key='phone_service')
            st.selectbox('Multiple lines', ['Yes','No'], key='multiple_lines')
            st.selectbox('Online security', ['Yes','No'], key='online_security')
            st.selectbox('Online backup', ['Yes','No'], key='online_backup') 
        with col3:
            st.write("#####")
            st.selectbox('Device protection', ['Yes','No'], key='device_protection')
            st.selectbox('Tech support', ['Yes','No'], key='tech_support')
            st.selectbox('TV Stream', ['Yes','No'], key='tv_stream')
            st.selectbox('Stream Movies', ['Yes','No'], key='stream_movies')
            st.selectbox('Paperless billing', ['Yes','No'], key='paperless_billing')
            
        st.form_submit_button('Make Prediction', on_click=make_prediction,
                              kwargs= dict(pipeline=pipeline, encoder=encoder))

# ========================================================================= Main Page

# if __name__== "__main__":
#     st.title('Make Prediction')
#     display_form()
    
#     prediction = st.session_state['predictions']
#     probability = st.session_state['probability']
    
#     if not prediction:
#         st.markdown('### Predictions will show here')
#     elif prediction == 'true':
#         probability_of_stay = probability[1] * 100
#         st.markdown(f'### The employee will stay with a probability of {round(probability_of_stay, 1)}%')
#     elif prediction == 'false':
#         probability_of_leave = probability[0] * 100
#         st.markdown(f'### The employee will leave with a probability of {round(probability_of_leave, 1)}%')
#     else:
#         st.markdown(f'### Unexpected prediction value: {prediction}')
    
# ========================================================================= Main Page

if __name__ == "__main__":
    st.title('Make Prediction')
    display_form()
    
    prediction = st.session_state['predictions']
    probability = st.session_state['probability']
    
    if prediction is None:  # Check if prediction is None
        st.markdown('### Predictions will show here')
    elif prediction == 'True':  # Assuming the encoder returns 'Yes' for a positive prediction
        probability_of_stay = probability[1] * 100
        st.markdown(f'### The customer is likely to leave with a probability of {round(probability_of_stay, 1)}%')
    else:
        probability_of_leave = probability[0] * 100
        st.markdown(f'### The customer is likely to stay with a probability of {round(probability_of_leave, 1)}%')
