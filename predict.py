import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Tech Career Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stSelectbox label {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        margin-top: 2rem;
        padding: 0.5rem;
        font-size: 1.1rem;
    }
    h1 {
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
with open('predict_model.pkl', 'rb') as file:
    data = pickle.load(file)

# Main title with emoji
st.title('💰 Tech Career Salary Predictor')

# Subtitle
st.markdown("""
    <p style='text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;'>
        Estimate your potential salary based on job details and market data
    </p>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Job Titles (kept the same list)
    Job = ('Data Engineer', 'Data Scientist', 'BI Developer',
           'Research Analyst', 'Business Intelligence Developer',
           'Data Analyst', 'Director of Data Science', 'MLOps Engineer',
           'Machine Learning Scientist', 'Machine Learning Engineer',
           'Data Science Manager', 'Applied Scientist',
           'Business Intelligence Analyst', 'Analytics Engineer',
           'Business Intelligence Engineer', 'Data Science',
           'Research Scientist', 'Research Engineer',
           'Managing Director Data Science', 'AI Engineer', 'Data Specialist',
           'Data Architect', 'Data Visualization Specialist', 'ETL Developer',
           'Data Science Practitioner', 'Computer Vision Engineer',
           'Data Lead', 'ML Engineer', 'Data Developer', 'Data Modeler',
           'Data Science Consultant', 'AI Architect',
           'Data Analytics Manager', 'Data Science Engineer',
           'Data Product Manager', 'Data Quality Analyst', 'Data Strategist',
           'Prompt Engineer', 'Data Science Lead',
           'Business Intelligence Manager', 'Data Manager',
           'Data Analytics Lead', 'Machine Learning Infrastructure Engineer',
           'Data Integration Engineer', 'Data Management Analyst',
           'BI Analyst', 'Business Data Analyst',
           'Machine Learning Operations Engineer', 'NLP Engineer',
           'Marketing Data Scientist', 'AI Scientist',
           'Machine Learning Researcher', 'Data Science Director',
           'Head of Data', 'Machine Learning Modeler',
           'Data Integration Specialist', 'Data Management Specialist',
           'AI Developer', 'Business Intelligence Specialist',
           'Data Quality Engineer', 'Decision Scientist',
           'Financial Data Analyst', 'Data Strategy Manager',
           'Data Visualization Engineer', 'Principal Data Scientist',
           'Data Infrastructure Engineer',
           'Machine Learning Software Engineer', 'Head of Machine Learning',
           'Data Operations Analyst', 'Data Operations Engineer',
           'Machine Learning Manager', 'BI Data Analyst',
           'AI Research Engineer', 'Deep Learning Engineer',
           'Data Operations Manager', 'Head of Data Science',
           'Software Data Engineer', 'Compliance Data Analyst',
           'Data Operations Specialist', 'Business Intelligence Data Analyst',
           'AWS Data Architect', 'Product Data Analyst',
           'Data Visualization Analyst', 'Cloud Data Engineer',
           'Finance Data Analyst', 'Applied Machine Learning Engineer',
           'Lead Data Analyst', 'BI Data Engineer', 'Cloud Database Engineer',
           'Applied Machine Learning Scientist', 'Data Analytics Specialist',
           'Big Data Engineer', 'Machine Learning Research Engineer',
           'Staff Data Analyst', 'Lead Data Scientist', 'Insight Analyst',
           'Azure Data Engineer', 'Data Analyst Lead', 'AI Programmer',
           'Applied Data Scientist', 'AI Product Manager',
           'Principal Machine Learning Engineer',
           'Lead Machine Learning Engineer', 'Data Quality Manager',
           'Data Product Owner', 'Data Modeller',
           'Autonomous Vehicle Technician', 'ETL Engineer',
           'Big Data Architect', 'Machine Learning Specialist',
           'Data DevOps Engineer', 'Principal Data Engineer',
           'Power BI Developer', 'Deep Learning Researcher',
           'Consultant Data Engineer', 'Computer Vision Software Engineer',
           'Manager Data Management', 'Data Analytics Consultant',
           'Data Analytics Engineer', 'Data Scientist Lead',
           'Machine Learning Developer', 'Principal Data Architect',
           'Marketing Data Analyst', 'Lead Data Engineer',
           'Principal Data Analyst', 'Staff Machine Learning Engineer',
           'Cloud Data Architect', 'Staff Data Scientist',
           'Marketing Data Engineer', 'Sales Data Analyst')
    
    job_title = st.selectbox(
        'Role',
        Job,
        help='Select your current or desired job title'
    )

    # Experience Level with descriptions
    Level = {
        'Entry-level': 'Entry Level (0-2 years)',
        'Mid-level': 'Mid Level (2-5 years)',
        'Senior-level': 'Senior Level (5+ years)',
        'Executive-level': 'Executive Level (Director/Head)'
    }
    experience_level = st.selectbox(
        'Experience Level',
        list(Level.keys()),
        format_func=lambda x: Level[x],
        help='Select your years of professional experience'
    )

    # Employment Type with descriptions
    Type = {
        'Full-time': 'Full Time',
        'Part-time': 'Part Time',
        'Contract': 'Contract Based',
        'Freelance': 'Freelance/Independent'
    }
    employment_type = st.selectbox(
        'Employment Type',
        list(Type.keys()),
        format_func=lambda x: Type[x]
    )

with col2:
    # Work Model with descriptions
    Model = {
        'Remote': 'Remote (100% Work from Home)',
        'Hybrid': 'Hybrid (Mix of Office & Remote)',
        'On-site': 'On-site (Office Based)'
    }
    work_models = st.selectbox(
        'Work Model',
        list(Model.keys()),
        format_func=lambda x: Model[x]
    )

    # Location with descriptions
    Location = {
        'United States': 'United States',
        'Canada': 'Canada',
        'Others': 'Other Countries'
    }
    company_location = st.selectbox(
        'Location',
        list(Location.keys()),
        format_func=lambda x: Location[x],
        help='Select the country where the job is based'
    )

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Centered predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('Calculate Estimated Salary', use_container_width=True)

# Prediction logic
if predict_button and job_title and experience_level and employment_type and work_models and company_location:
    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'job_title': [job_title],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'work_models': [work_models],
        'company_location': [company_location]
    })

    # Preprocess and predict
    new_data_transformed = data.named_steps['preprocessor'].transform(new_data)
    prediction = data.named_steps['model'].predict(new_data_transformed)
    
    # Display the prediction in a nicely formatted box
    st.markdown("""
        <div class="prediction-box">
            <h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Estimated Annual Salary</h3>
            <p style='font-size: 2rem; font-weight: bold; color: #0066cc; margin-bottom: 1rem;'>
                ${:,.2f}
            </p>
            <p style='font-size: 0.9rem; color: #666;'>
                Based on a {}-level {} position<br>
                {} • {} • {}
            </p>
        </div>
    """.format(
        prediction[0],
        experience_level.lower(),
        job_title,
        work_models.lower(),
        company_location
    ), unsafe_allow_html=True)
    
    # Add a disclaimer
    st.markdown("""
        <p style='text-align: center; font-size: 0.8rem; color: #666; margin-top: 1rem;'>
            This estimate is based on market data and may vary based on additional factors
            such as company size, specific skills, and market conditions.
        </p>
    """, unsafe_allow_html=True)
