import streamlit as st

st.set_page_config(
    page_title= "Bank Client Subscribtion App")

st.title("Bank Client Subscribtion Prediction")
st.markdown("""
            This app leverages advanced data analytics and machine learning algorithms to predict the likelihood of client subscribtion to bank product (term deposit). 
            By analyzing historical data, the app identifies patterns and key factors contributing to customer attrition.
""")

st.subheader("Key features")
st.markdown("""
            - __Predictions__ : View subscribtion of bank customers.
            - __View Data__ : Access proprietary data
            - __Dashboard__ : Explore interactive data visualization for insights.
           """)


st.subheader("Steps")
st.markdown("""
            - __Upload Your Data__ : Upload your customer dataset, ensure it all includes relevant features.
            - __Data Processing__ : The app cleans and processes the data to prepare it for modeling.
            - __Prediction__ : The machine learning model predicts the probability of each customer leaving the organization.
            - __Insights__ : Explore the insights generated to understand the drivers of churn and identify at-risk customers.
            """)

st.subheader("Benefits of Predicting Client Subscriptions")
st.markdown("""
            - **Targeted Marketing**: Enables precise identification of potential subscribers, optimizing marketing efforts and costs.  
            - **Improved Conversion Rates**: Focuses on customers most likely to subscribe, boosting campaign success rates.  
            - **Efficient Resource Utilization**: Allocates time and resources effectively by prioritizing high-potential leads.  
            - **Enhanced Decision-Making**: Provides actionable insights into customer behavior and preferences for strategic planning.  
            - **Business Growth**: Strengthens customer relationships and maximizes ROI through data-driven marketing strategies.  
            """)

st.subheader("About The Model")
st.markdown("""
            Our app utilizes a robust machine learning model, fine-tuned for accuracy and interpretability.  
            The model considers multiple factors, including customer demographics, engagement patterns, and previous campaign outcomes,  
            providing a comprehensive prediction of subscription likelihood. While the model offers high accuracy,  
            predictions should be interpreted within the context of your bank's unique marketing strategies and objectives.  
            """)


st.subheader("Disclaimer")
st.markdown("""Please note that while our predictions are highly accurate, they depend on the quality and relevance of the input data. 
            We recommend using these insights in conjunction with other business strategies and not as the sole basis for critical decisions.
            """)

st.subheader("Contact Us")
customer_support_email = "kay.yamoah10@gmail.com"
st.markdown(f"""
For any questions or concerns, please contact our customer support team at [this email address]({customer_support_email}).
""")
# st.button("Repository on GitHub", help= "Visit the Github repository")
