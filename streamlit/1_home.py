import streamlit as st

# Page configuration
st.set_page_config(page_title="Bank Client Subscription Prediction App")

# Title and introduction
st.title("Bank Client Subscription Prediction")
st.markdown("""
This app uses machine learning to predict the likelihood of a client subscribing to a bank's term deposit product.  
By inputting relevant customer details, the app generates accurate predictions to support your decision-making.
""")

# Features
st.subheader("Key Features")
st.markdown("""
- **Prediction**: Input customer details to predict subscription likelihood.
""")

# Steps for prediction
st.subheader("How It Works")
st.markdown("""
1. **Input Data**: Provide the required customer details (features).  
2. **Get Prediction**: The app processes your input and predicts the likelihood of subscription.  
""")

# Benefits of the app
st.subheader("Why Use This App?")
st.markdown("""
- **Precise Predictions**: Focus your efforts on customers most likely to subscribe.  
- **Data-Driven Decision Making**: Enhance strategies with reliable insights.  
- **Improved Efficiency**: Prioritize high-potential leads effectively.  
""")

# About the model
st.subheader("Model Information")
st.markdown("""
The app leverages a robust machine learning model trained on historical customer data. It considers multiple factors to provide accurate predictions.  
While predictions are reliable, they should be used as part of a broader strategy and not as the sole basis for decisions.
""")

# Disclaimer
st.subheader("Disclaimer")
st.markdown("""
This app is designed solely for making predictions about client subscriptions.  
The accuracy of the predictions depends on the quality and completeness of the input data.  
We recommend using these predictions as a supplementary tool alongside your own analysis and expertise, not as the sole basis for critical business decisions.
""")

# Contact information
st.subheader("Contact Us")
customer_support_email = "kay.yamoah10@gmail.com"
st.markdown(f"""
For inquiries or assistance, please reach out to our support team at [this email address]({customer_support_email}).
""")
