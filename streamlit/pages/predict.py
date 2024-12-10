import streamlit as st
import joblib
import pandas as pd
import os
import datetime

# =========================================================================================================== Page Setup
st.set_page_config(
    page_title='Predict',
    layout='wide'
)

# =========================================================================================================== Load Pipelines
@st.cache_resource(show_spinner='Model loading')  # A decorator for caching
def load_rf_pipeline():
    pipeline = joblib.load(r'C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\streamlit\models\random_forest_model.pk1')
    return pipeline

@st.cache_resource(show_spinner='Model loading')
def load_svc_pipeline():
    pipeline = joblib.load(r'C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\streamlit\models\svc_model.pkl')
    return pipeline

# =========================================================================================================== Function
def select_model():
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox('Select a Model', options=['Random Forest', 'SVC'], key='selected_model')
    with col2:
        pass
    
    if st.session_state['selected_model'] == 'Random Forest':
        pipeline = load_rf_pipeline()
    else:
        pipeline = load_svc_pipeline()
    
    encoder = joblib.load(r'C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\streamlit\models\encoder.joblib')   
    return pipeline, encoder

# =========================================================================================================== For Session State
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'probability' not in st.session_state:
    st.session_state.probability = None

# =========================================================================================================== Function
def make_prediction(pipeline, encoder):
    age = st.session_state['age']
    job = st.session_state['job']
    marital = st.session_state['marital']
    education = st.session_state['education']
    default = st.session_state['default']
    balance = st.session_state['balance']
    housing = st.session_state['housing']
    loan = st.session_state['loan']
    contact = st.session_state['contact']
    day = st.session_state['day']
    month = st.session_state['month']
    duration = st.session_state['duration']
    campaign = st.session_state['campaign']
    pdays = st.session_state['pdays']
    previous = st.session_state['previous']
    poutcome = st.session_state['poutcome']
    
    column = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
              'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
              'previous', 'poutcome']
    
    data = [[age, job, marital, education,
             default, balance, housing, loan,
             contact, day, month, duration,
             campaign, pdays, previous, poutcome]]
    
    df = pd.DataFrame(data, columns=column)
    df['Prediction time'] = datetime.date.today()
    df['Model used'] = st.session_state['selected_model']
    df.to_csv('history_data\history.csv', mode='a', header=not os.path.exists('history_data\history.csv'), index=False)
    
    pred = pipeline.predict(df)
    pred = int(pred[0])
    prediction = encoder.inverse_transform([pred])
    
    probability = pipeline.predict_proba(df)
    probability = probability[0]
    
    st.session_state['predictions'] = prediction[0]
    st.session_state['probability'] = probability
    
    return prediction[0], probability

# =========================================================================================================== Function (display)
def display_form():
    pipeline, encoder = select_model()
    with st.form('input-feature'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.number_input('Age', min_value=0, max_value=120, key='age')
            st.selectbox('Job', ['Admin.', 'Technician', 'Entrepreneur', 'Management', 'Blue-collar',
                                 'Unknown', 'Retired', 'Services', 'Self-employed', 'Unemployed', 'Student'], key='job')
            st.selectbox('Marital Status', ['Married', 'Single', 'Divorced'], key='marital')
            st.selectbox('Education Level', ['Primary', 'Secondary', 'Tertiary', 'Unknown'], key='education')
            st.selectbox('Default Credit', ['Yes', 'No'], key='default')
            st.number_input('Balance', value=0.0, key='balance')
        
        with col2:
            st.selectbox('Housing Loan', ['Yes', 'No'], key='housing')
            st.selectbox('Personal Loan', ['Yes', 'No'], key='loan')
            st.selectbox('Contact Type', ['Telephone', 'Cellular', 'Unknown'], key='contact')
            st.number_input('Day of Month Contacted', min_value=1, max_value=31, key='day')
            st.selectbox('Month', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], key='month')
            st.number_input('Call Duration (seconds)', value=0, key='duration')
        
        with col3:
            st.number_input('Campaign Contacts', min_value=0, key='campaign')
            st.number_input('Days since Previous Campaign Contact', min_value=-1, key='pdays')
            st.number_input('Number of Previous Contacts', min_value=0, key='previous')
            st.selectbox('Previous Outcome', ['Success', 'Failure', 'Other', 'Unknown'], key='poutcome')
        
        st.form_submit_button('Make Prediction', on_click=make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder))

# =========================================================================================================== Main Page
if __name__ == "__main__":
    st.title('Make Prediction')
    display_form()
    
    prediction = st.session_state['predictions']
    probability = st.session_state['probability']
    
    if prediction is None:
        st.markdown('### Predictions will show here')
    elif prediction == 'Yes':
        probability_of_yes = probability[1] * 100
        st.markdown(f'### The customer is likely to subscribe with a probability of {round(probability_of_yes, 1)}%')
    else:
        probability_of_no = probability[0] * 100
        st.markdown(f'### The customer is unlikely to subscribe with a probability of {round(probability_of_no, 1)}%')
