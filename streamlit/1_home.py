import streamlit as st

st.set_page_config(
    page_title= "Customer Churn App")

st.title("Customer Churn Prediction")
st.markdown("""
            This app leverages advanced data analytics and machine learning algorithms to predict the likelihood of customer churn. 
            By analyzing historical data, the app identifies patterns and key factors contributing to customer attrition.
""")

st.subheader("Key features")
st.markdown("""
            - __Predictions__ : View churn prediction of customers.
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

st.subheader("Benefits of Predicting Churn")
st.markdown("""
            - __Cost-Effectiveness__ : Reduces customer churn by identifying and addressing issues early.
            - __Resource Allocation__ : Allocates resources effectively to prevent churn.
            - __Strategic Planning__ : Improves business strategies by anticipating customer churn and taking proactive measures.
            - __Customer Insights__ : Dashboard Displays a comprehensive analysis of your Customers for deeper understanding of what drives customer behavior a 
            """)

st.subheader("About The Model")
st.markdown("""Our app utilizes a robust machine learning model, fine-tuned for accuracy and interpretability. 
            The model considers multiple factors, from transaction frequency to customer feedback, 
            providing a comprehensive churn prediction. While the model offers high accuracy, 
            predictions should be interpreted within the context of your business's unique circumstances.
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
