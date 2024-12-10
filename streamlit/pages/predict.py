import streamlit as st
import joblib
import pandas as pd
import os
import datetime

# ===========================================================================================================
# Page Setup
st.set_page_config(
    page_title='Predict',
    layout='wide'
)

# ===========================================================================================================
# Load Pipelines
@st.cache_resource(show_spinner='Model loading')
def load_rf_pipeline():
    return joblib.load(r'C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\streamlit\models\random_forest_model.pk1')


@st.cache_resource(show_spinner='Model loading')
def load_svc_pipeline():
    return joblib.load(r'C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\streamlit\models\svc_model.pkl')


# ===========================================================================================================
# Load Thresholds
@st.cache_resource(show_spinner='Threshold loading')
def load_thresholds():
    rf_threshold = joblib.load(r'C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\streamlit\models\rf_threshold.pkl')
    svc_threshold = joblib.load(r"C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\streamlit\models\svc_threshold.pkl")
    return rf_threshold, svc_threshold


# ===========================================================================================================
# Model Selection Function
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


# ===========================================================================================================
# Initialize Session State
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'probability' not in st.session_state:
    st.session_state.probability = None


# ===========================================================================================================
# Prediction Function
def make_prediction(pipeline, encoder, threshold):
    # Collect user inputs from session state
    features = ['age', 'job', 'marital', 'education', 'default', 'balance',
                'housing', 'loan', 'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome']
    data = [[st.session_state[feature] for feature in features]]
    df = pd.DataFrame(data, columns=features)

    # Add metadata for history
    df['Prediction time'] = datetime.date.today()
    df['Model used'] = st.session_state['selected_model']
    history_path = 'history_data/history.csv'
    df.to_csv(history_path, mode='a', header=not os.path.exists(history_path), index=False)

    # Make prediction
    probability = pipeline.predict_proba(df)[0]
    predicted_class = int(probability[1] >= threshold)
    prediction = encoder.inverse_transform([predicted_class])
    st.session_state['predictions'] = prediction[0]
    st.session_state['probability'] = probability
    return prediction[0], probability


# ===========================================================================================================
# Display Form Function
def display_form():
    pipeline, encoder = select_model()
    rf_threshold, svc_threshold = load_thresholds()
    threshold = rf_threshold if st.session_state['selected_model'] == 'Random Forest' else svc_threshold

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

        st.form_submit_button('Make Prediction', on_click=make_prediction, kwargs=dict(pipeline=pipeline, encoder=encoder, threshold=threshold))


# ===========================================================================================================
# Main Page
if __name__ == "__main__":
    st.title('Make Prediction')
    display_form()

    prediction = st.session_state['predictions']
    probability = st.session_state['probability']

    if prediction is None:
        st.markdown('### Predictions will show here')
    elif prediction == 'yes':
        probability_of_yes = probability[1] * 100
        st.markdown(f'### The customer is likely to subscribe with a probability of {round(probability_of_yes, 1)}%')
    else:
        probability_of_no = probability[0] * 100
        st.markdown(f'### The customer is unlikely to subscribe with a probability of {round(probability_of_no, 1)}%')
