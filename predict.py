import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open('predict_model.pkl', 'rb') as file:
    data = pickle.load(file)


st.title ('Tech Salary Prediction')

st.sidebar.header('Select Features For Prediction')


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

Level = ('Mid-level', 'Senior-level', 'Entry-level', 'Executive-level')

Type = ('Full-time', 'Part-time', 'Contract', 'Freelance')

Model = ('Remote', 'On-site', 'Hybrid')

Location = ('United States','Canada', 'Others')

job_title = st.sidebar.selectbox('Job Title', Job)
experience_level = st.sidebar.selectbox('Job Level', Level)
employment_type = st.sidebar.selectbox('Job Type', Type)
work_models = st.sidebar.selectbox('Job Model', Model)
company_location = st.sidebar.selectbox('Location', Location)

predict_button = st.sidebar.button('Predict')

if predict_button and job_title and experience_level and employment_type and work_models and company_location:
    # Create a DataFrame for the new data
    new_data = pd.DataFrame({
        'job_title': [job_title],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'work_models': [work_models],
        'company_location': [company_location]
    })


# Preprocess the input data using the same preprocessor from the pipeline
    new_data_transformed = data.named_steps['preprocessor'].transform(new_data)
    
    # Make prediction using the loaded model
    prediction = data.named_steps['model'].predict(new_data_transformed)

    # Display the prediction
    st.write(f"Predicted Salary of a {experience_level} {job_title} working {work_models} is ${prediction[0]:,.2f}")

    st.write(f"Country: {company_location}")
 