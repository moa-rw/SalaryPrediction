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

# Custom CSS
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Load model
with open('predict_model.pkl', 'rb') as file:
    data = pickle.load(file)

# Title and description
st.title('💰 Tech Career Salary Predictor')
st.markdown('##### Estimate your potential salary based on job details and market data')
st.markdown('---')

# Job titles
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

# Create two columns
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox('Role', Job)
    
    experience_level = st.selectbox(
        'Experience Level',
        ('Entry-level (0-2 years)', 
         'Mid-level (2-5 years)', 
         'Senior-level (5+ years)', 
         'Executive-level')
    )
    # Convert to model format
    experience_level = experience_level.split(' ')[0]
    
    employment_type = st.selectbox(
        'Employment Type',
        ('Full-time', 'Part-time', 'Contract', 'Freelance')
    )

with col2:
    work_models = st.selectbox(
        'Work Model',
        ('Remote (Work from Home)', 
         'Hybrid (Office & Remote)', 
         'On-site (Office Based)')
    )
    # Convert to model format
    work_models = work_models.split(' ')[0]
    
    company_location = st.selectbox(
        'Location',
        ('United States', 'Canada', 'Others')
    )

st.markdown('---')

# Predict button
predict_button = st.button('Calculate Estimated Salary')

# Prediction logic
if predict_button:
    # Create DataFrame for prediction
    new_data = pd.DataFrame({
        'job_title': [job_title],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'work_models': [work_models],
        'company_location': [company_location]
    })

    # Make prediction
    new_data_transformed = data.named_steps['preprocessor'].transform(new_data)
    prediction = data.named_steps['model'].predict(new_data_transformed)
    
    # Display prediction
    st.success(f"### Estimated Annual Salary: ${prediction[0]:,.2f}")
    
    # Display job details
    st.info(f"""
    #### Position Details:
    - Role: {job_title}
    - Experience Level: {experience_level}
    - Work Model: {work_models}
    - Location: {company_location}
    """)
    
    st.caption("""
    Note: This estimate is based on market data and may vary based on factors such as 
    company size, specific skills, and market conditions.
    """)
