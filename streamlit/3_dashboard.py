import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 

# Caching the data loading process
@st.cache_data(persist=True)
def load_data(filepath):
    df1= pd.read_csv(filepath)
     # Preprocess the data
    df2 = df1.drop(['customerID'], axis=1, errors='ignore')  # drop irrelevant columns
    return df2

# Caching the data conversion process
@st.cache_data(persist=True)
def convert_categorical_to_continuous(df):
    # Convert all categorical columns to numeric using pandas' factorize
    for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        df[col], _ = pd.factorize(df[col])
    return df

def dynamic_bar_chart(data):
    
    # Allow user to select columns
    columns = data.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        col_y = st.selectbox("Select for Y-axis", columns)
    with col2:
        col_x = st.selectbox("Select for X-axis", columns)
    
    # Create bar chart if both selections are valid
    if col_x and col_y:
        fig = px.bar(data, x=col_x, y=col_y, title=f"{col_x} vs {col_y}",
                     labels={col_x: col_x, col_y: col_y})
        st.plotly_chart(fig)
        
def bivariate_analysis(df, target_column):
    st.write("## Bivariate Analysis for Churn")
    
    # Convert boolean target_column to string for proper labeling
    df[target_column] = df[target_column].astype(str)
    
    # Categorical Columns - Use Bar Chart
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in categorical_cols:
        if col != target_column:  # To separate Churn
            fig = px.bar(df, x=col, color=target_column,
                        title=f"{col} vs {target_column}",
                        barmode='group')  # Label for True/False
            st.plotly_chart(fig)
    
    # Numerical Columns - Use Box Plot
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        fig = px.box(df, x=target_column, y=col, 
                    title=f"{col} vs {target_column}",
                    labels={target_column: 'Churn'},
                    color=target_column)
        st.plotly_chart(fig)

@st.cache_data(persist=True)
def compute_churn_distribution(data):
    churn_count = data['Churn'].value_counts().reset_index()
    churn_count.columns = ['Churn', 'Count']
    return churn_count

def plot_churn_distribution(churn_count):
    fig = px.pie(churn_count, names='Churn', values='Count', title='Distribution of Churn',color_discrete_sequence=['red', 'blue'])
    return fig 

@st.cache_data(persist=True)
def compute_correlation(data):
    data_continuous = convert_categorical_to_continuous(data)
    return data_continuous.corr() 

def show_select_scatterplot(data):
    # Get numeric columns
    num_cols = data.select_dtypes(include=['float64', 'int64', 'int', 'float']).columns.tolist()
    
    # Allow the user to select X and Y axes
    col1, col2 = st.columns(2)
    with col1:
        y_axis = st.selectbox('Select for Y-axis', num_cols)
    with col2:
        x_axis = st.selectbox('Select for X-axis', num_cols)
   
    fig = px.scatter(data, x=x_axis, y=y_axis, color='Churn', 
                     title=f"Scatterplot: {x_axis} vs {y_axis}")
    st.plotly_chart(fig)

def show_dashboard():
    
    # Load the dataset using the cached function
    data = load_data(r'data\cleaned_data.csv')
    
    # Display the dataset
    st.title('Telecom Customer Churn EDA')
    st.write('## Dataset Overview')
    st.write(data.head())  # Display the first 5 rows of the dataset
    
    # Descriptive statistics
    st.write('## Descriptive Statistics')
    st.write(data.describe())  # Summary statistics
    
    # Distribution of churn
    st.write('## Churn Distribution')
    churn_count = compute_churn_distribution(data)
    fig1 = plot_churn_distribution(churn_count)
    st.plotly_chart(fig1)
    
    # Bar Chart comparison
    st.write('## Bar Chart Comparisons')
    dynamic_bar_chart(data)
    
    # Distribution of  features
    st.write('## Distribution of Features')
    columns = data.columns.tolist()
    selected_dnf = st.selectbox("Select a Features", columns)
    if selected_dnf:
        fig2 = px.histogram(data, x=selected_dnf, title=f'Distribution of {selected_dnf}')
        st.plotly_chart(fig2)
    
    # Correlation heatmap
    st.write('## Correlation Heatmap')
    data2 = convert_categorical_to_continuous(data)
    corr = data2.corr()
    fig3 = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Heatmap')
    st.plotly_chart(fig3)
    
    if st.button("Show More Analysis"):
        # Pairplot (Optional for highly correlated features)
        st.write('## Scatterplot of Numerical Pairs')
        show_select_scatterplot(data)

        # Perform bivariate analysis
        bivariate_analysis(data, 'Churn')  # 'Churn' is the target variable
            
# Run the dashboard
if __name__ == '__main__':
    show_dashboard()

