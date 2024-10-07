#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer


# In[25]:


df=pd.read_csv("/home/ubuntu/Downloads/owid-covid-data.csv")
df.head()


# In[26]:


df.shape 


# In[27]:


df.drop(["new_cases_smoothed"],axis=1,inplace=True)


# In[28]:


df.drop(["new_deaths_smoothed","new_cases_per_million","new_cases_smoothed_per_million"],axis=1,inplace=True)


# In[29]:


df.drop(["reproduction_rate","icu_patients","icu_patients_per_million"],axis=1,inplace=True)


# In[30]:


df.drop(["hosp_patients","hosp_patients_per_million","weekly_icu_admissions","weekly_icu_admissions_per_million"],axis=1,inplace=True)


# In[31]:


df.drop(["new_tests_smoothed","new_tests_smoothed_per_thousand","new_vaccinations_smoothed"],axis= 1,inplace=True)


# In[32]:


df.drop(["total_vaccinations_per_hundred","people_vaccinated_per_hundred"],axis= 1,inplace=True)


# In[33]:


df.drop(["people_fully_vaccinated_per_hundred","total_boosters_per_hundred"],axis= 1,inplace=True)


# In[34]:


df.drop(["new_vaccinations_smoothed_per_million","new_people_vaccinated_smoothed"],axis= 1,inplace=True)


# In[35]:


df.drop(["new_people_vaccinated_smoothed_per_hundred"],axis= 1,inplace=True)


# In[36]:


df.rename(columns={'date': 'Date','location':'Country','continent': 'Continent','iso_code':'ISO_code'},inplace=True)


# In[37]:


imputer=SimpleImputer(strategy='constant')
df2=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)


# In[38]:


df3 = df2.groupby(['Date','Country',])[['total_cases','total_deaths','total_vaccinations']].sum().reset_index()


# In[39]:


df3.total_cases.replace({'missing_value':0},inplace=True)


# In[40]:


df3.total_deaths.replace({'missing_value':0},inplace=True)


# In[41]:


df3.total_vaccinations.replace({'missing_value':0},inplace=True)


# In[42]:


print(df3)


# In[43]:


df4=df[df3.total_deaths>1000000]


# In[44]:


cont=df4.Continent.unique() 


# In[45]:


len(cont)


# In[46]:


continent_deaths_greater=list(df4.Continent.unique())


# In[47]:


print(continent_deaths_greater)


# In[52]:


for i in range(0,len(cont)):
    c=df[df.Continent==cont[i]].reset_index()
    plt.scatter(np.arange(0,len(c)),c['total_cases'],color='blue' ,label='Total Cases')
    plt.scatter(np.arange(0,len(c)),c['total_deaths'],color='red' ,label='Total Deaths')
    plt.scatter(np.arange(0,len(c)),c['total_vaccinations'],color='green' ,label='Total Vaaccinations')
    plt.title(cont[i])
    plt.xlabel("No.of days since first suspect")
    plt.ylabel("No.of cases")
    plt.legend()
    plt.show()
    


# In[ ]:





# In[ ]:




