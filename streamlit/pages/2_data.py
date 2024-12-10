import streamlit as st
import pandas as pd

st.title("View Data")

# Load data

@st.cache_data(persist= True)
def load_data():
    data = pd.read_csv(r'C:\Users\Pc\Desktop\Data analysis\Azubi Africa\term-deposit-subscription-prediction\data\bank-full.csv', sep=";")
    data = data[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                'previous', 'poutcome', 'y']]
    return data

st.dataframe(load_data())

# st.write('Summary Statistics')
# st.write(load_data().describe())