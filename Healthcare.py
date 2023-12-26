#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_excel(r"C:\Users\USER\Documents\Chat GPT Projects\RAW DATA\Healthcare\healthcare_dataset.xlsx")


# In[3]:


df.head()


# In[6]:


# Checking for missing values
missing_values = df.isnull().sum()
missing_values


# In[7]:


# Checking data types and unique values for potential inconsistencies
data_types = df.dtypes
unique_values = df.nunique()

missing_values, data_types, unique_values


# In[8]:


# Converting 'Date of Admission' and 'Discharge Date' to datetime
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])


# In[9]:


# For the gender inconsistency, we'll simply check a few potentially incorrect entries
# Ideally, a more sophisticated name-gender matching could be used, but this is out of scope for this analysis
suspicious_gender_entries = df[df['Name'].str.contains('Mrs.') & (df['Gender'] == 'Male')]
suspicious_gender_entries.head()


# In[10]:


# Correcting gender inconsistencies
df.loc[df['Name'].str.contains('Mrs.'), 'Gender'] = 'Female'


# In[11]:


# Re-checking the entries to confirm the correction
corrected_entries = df[df['Name'].str.contains('Mrs.') & (df['Gender'] == 'Female')]
corrected_entries.head()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


# Analysis of treatment effectiveness
effectiveness_data = df.groupby('Medication')['Test Results'].value_counts(normalize=True).unstack()


# In[14]:


# Focusing on the percentage of 'Normal' results for each medication
normal_results = effectiveness_data['Normal'].sort_values(ascending=False)


# In[15]:


# Plotting the effectiveness
plt.figure(figsize=(10, 6))
sns.barplot(x=normal_results.index, y=normal_results.values)
plt.title('Effectiveness of Medications (Percentage of Normal Test Results)')
plt.ylabel('Percentage of Normal Test Results')
plt.xlabel('Medication')
plt.show()


# In[16]:


# Calculating readmission rates:

# First, we need to sort the data by patient name and admission date
sorted_data = df.sort_values(by=['Name', 'Date of Admission'])


# In[17]:


# Calculating the time difference between consecutive admissions for the same patient
sorted_data['Previous Discharge Date'] = sorted_data.groupby('Name')['Discharge Date'].shift()
sorted_data['Days Since Last Admission'] = (sorted_data['Date of Admission'] - sorted_data['Previous Discharge Date']).dt.days


# In[18]:


# Identifying readmissions within 30 days
readmissions = sorted_data[sorted_data['Days Since Last Admission'] <= 30]


# In[19]:


# Calculating readmission rates
total_patients = df['Name'].nunique()
readmission_rate = len(readmissions['Name'].unique()) / total_patients

readmission_rate


# In[20]:


# Distribution of medical conditions
condition_distribution = df['Medical Condition'].value_counts(normalize=True)


# In[21]:


# Plotting the distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=condition_distribution.index, y=condition_distribution.values)
plt.title('Distribution of Medical Conditions')
plt.ylabel('Percentage of Patients')
plt.xlabel('Medical Condition')
plt.xticks(rotation=45)
plt.show()


# In[24]:


# Assuming df is your DataFrame
plt.figure(figsize=(12, 8))
sns.boxplot(x='Medical Condition', y='Age', data=df)  # 'data=df' is the correct way
plt.title('Age Distribution by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# Gender distribution for each medical condition
gender_distribution = pd.crosstab(df['Medical Condition'], df['Gender'])


# In[26]:


# Plotting the gender distribution
gender_distribution.plot(kind='bar', figsize=(12, 8))
plt.title('Gender Distribution for Each Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.show()


# In[27]:


# Billing amounts across different medical conditions
billing_medical_condition = df.groupby('Medical Condition')['Billing Amount'].mean().sort_values(ascending=False)


# In[28]:


# Plotting the billing amounts
plt.figure(figsize=(12, 8))
sns.barplot(x=billing_medical_condition.index, y=billing_medical_condition.values)
plt.title('Average Billing Amount by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Average Billing Amount')
plt.xticks(rotation=45)
plt.show()


# In[29]:


import numpy as np


# In[30]:


df.info()


# In[31]:


# Calculate the length of stay
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days



# In[33]:


# Analyzing the relationship between length of stay and test results

# Assuming df is the DataFrame
plt.figure(figsize=(10, 6))
sns.boxplot(x='Test Results', y='Length of Stay', data=df)  # Corrected here
plt.title('Length of Stay by Test Results')
plt.xlabel('Test Results')
plt.ylabel('Length of Stay (Days)')
plt.show()


# In[34]:


# Checking the mean length of stay for each test result category
mean_length_of_stay = df.groupby('Test Results')['Length of Stay'].mean()
mean_length_of_stay


# In[35]:


# Identifying patients who have been admitted more than once
readmissions = df['Name'].value_counts()
readmitted_patients = readmissions[readmissions > 1]


# In[36]:


# Total number of unique patients
total_unique_patients = df['Name'].nunique()


# In[37]:


# Number of readmitted patients
num_readmitted_patients = len(readmitted_patients)


# In[38]:


# Calculating the readmission rate
readmission_rate = num_readmitted_patients / total_unique_patients


# In[39]:


# Displaying the readmission rate
readmission_rate


# In[40]:


# Readmission rates by medical condition
readmissions_by_condition = df[df['Name'].isin(readmitted_patients.index)].groupby('Medical Condition')['Name'].nunique()
readmission_rate_by_condition = readmissions_by_condition / df.groupby('Medical Condition')['Name'].nunique()


# In[41]:


# Plotting readmission rates by medical condition
plt.figure(figsize=(10, 6))
readmission_rate_by_condition.plot(kind='bar')
plt.title('Readmission Rate by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Readmission Rate')
plt.show()


# In[42]:


# Displaying the readmission rates by medical condition
readmission_rate_by_condition


# In[43]:


# Analyzing the frequency of different medical conditions
condition_frequency = df['Medical Condition'].value_counts()


# In[44]:


# Plotting the frequency of medical conditions
plt.figure(figsize=(10, 6))
condition_frequency.plot(kind='bar')
plt.title('Frequency of Medical Conditions')
plt.xlabel('Medical Condition')
plt.ylabel('Frequency')
plt.show()


# In[45]:


# Exploring correlations between medical conditions and age
plt.figure(figsize=(10, 6))
sns.boxplot(x='Medical Condition', y='Age', data=df)
plt.title('Age Distribution by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Age')
plt.show()


# In[46]:


# Exploring correlations between medical conditions and gender
gender_distribution = df.groupby(['Medical Condition', 'Gender']).size().unstack()
gender_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Gender Distribution by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.show()


# In[47]:


# Checking if certain blood types are more prevalent in certain conditions
blood_type_distribution = df.groupby(['Medical Condition', 'Blood Type']).size().unstack()
blood_type_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Blood Type Distribution by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.show()


# In[48]:


# Exploring the relationship between billing amounts and medical conditions
plt.figure(figsize=(10, 6))
sns.barplot(x='Medical Condition', y='Billing Amount', data=df)
plt.title('Average Billing Amount by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Average Billing Amount')
plt.xticks(rotation=45)
plt.show()


# In[49]:


# Exploring the relationship between billing amounts and age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Billing Amount', data=df)
plt.title('Billing Amount vs. Age')
plt.xlabel('Age')
plt.ylabel('Billing Amount')
plt.show()


# In[50]:


# Exploring the relationship between billing amounts and length of stay
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Length of Stay', y='Billing Amount', data=df)
plt.title('Billing Amount vs. Length of Stay')
plt.xlabel('Length of Stay (Days)')
plt.ylabel('Billing Amount')
plt.show()


# In[ ]:




