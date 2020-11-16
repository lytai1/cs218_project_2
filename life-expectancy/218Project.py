#!/usr/bin/env python
# coding: utf-8

# In[33]:


## need to do: some null value can be replaced with mean, some population can be obtained from internet,do not need
## to delete all with null value, need to do it later if have time.
## normalized data


# In[1]:


import pandas as pd
import numpy as np 
##import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[2]:


lifeExpec = pd.read_csv('LifeExpectancyData.csv')


# In[3]:


lifeExpec.head()


# In[4]:


lifeExpec.shape


# In[5]:


lifeExpec.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Expectancy","Adult Mortality":"Adult_Mortality",
                   "infant deaths":"Infant_Deaths","percentage expenditure":"Percentage_Exp","Hepatitis B":"HepatitisB",
                  "Measles ":"Measles"," BMI ":"BMI","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",
                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_1to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",
                   "Total expenditure":"Tot_Exp"},inplace=True)


# In[6]:


lifeExpec.columns


# In[7]:


lifeExpec.isnull().sum()


# In[8]:


## replace developed country to 1,developing country to 2
##lifeExpec['Status'].replace({'Developing': 2,'Developed': 1 }, inplace = True)


# In[9]:


country_list = lifeExpec.Country.unique()
fill_list = ['Life_Expectancy','Adult_Mortality','Alcohol','HepatitisB','BMI','Polio','Tot_Exp','Diphtheria','GDP','Population','thinness_1to19_years','thinness_5to9_years','Income_Comp_Of_Resources','Schooling']


# In[10]:


for country in country_list:
    lifeExpec.loc[lifeExpec['Country'] == country,fill_list] = lifeExpec.loc[lifeExpec['Country'] == country,fill_list].interpolate()


# In[11]:


# Drop remaining null values after interpolation.
lifeExpec.dropna(inplace=True)
lifeExpec.shape


# In[12]:


countryNames = lifeExpec.Country.unique()
countryNames.size


# In[13]:


lifeExpec.describe()


# In[14]:


round(lifeExpec[['Status','Life_Expectancy']].groupby(['Status']).mean(),2)


# In[15]:


l =(round(lifeExpec.groupby('Status')['Life_Expectancy'].mean(), 2).to_numpy())


# In[16]:


## life Expectancy vs status 
l =(round(lifeExpec.groupby('Status')['Life_Expectancy'].mean(), 2).to_numpy())
plt.figure(figsize=(6,6))
plt.bar(lifeExpec.groupby('Status')['Status'].count().index,lifeExpec.groupby('Status')['Life_Expectancy'].mean())
plt.xlabel("Status",fontsize=12,fontweight='bold')
plt.ylabel("Avg Life Expectancy",fontsize=12,fontweight='bold')
plt.ylim(0, 85)
plt.title("Life Expectancy vs Status",fontweight='bold')
for i, v in enumerate(l):
    plt.text(i +0.05,v +2,str(v), color='blue', fontweight='bold' )
plt.show()


# In[17]:


import scipy.stats as stats
stats.ttest_ind(lifeExpec.loc[lifeExpec['Status']=='Developed','Life_Expectancy'],lifeExpec.loc[lifeExpec['Status']=='Developing','Life_Expectancy'])


# In[18]:


## detect outliers using boxplot
col_dict = {'Life_Expectancy':1,'Adult_Mortality':2,'Infant_Deaths':3,'Alcohol':4,'Percentage_Exp':5,'HepatitisB':6,'Measles':7,'BMI':8,'Under_Five_Deaths':9,'Polio':10,'Tot_Exp':11,'Diphtheria':12,'HIV/AIDS':13,'GDP':14,'Population':15,'thinness_1to19_years':16,
            'thinness_5to9_years':17,'Income_Comp_Of_Resources':18,'Schooling':19}
plt.figure(figsize=(20,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(lifeExpec[variable],whis=1.5)
                     plt.title(variable)

plt.show()


# The boxplot results shows there are outliers exist in each variables. Next, we will take a look at each indivual variable, to see if we need to normalize the variables.

# In[20]:


## check life Expecancy
lifeExpec[lifeExpec["Life_Expectancy"] < 45]


# In[27]:


lifeExpec[lifeExpec["Country"] == 'Sierra Leone']['Life_Expectancy']


# In[28]:


lifeExpec[lifeExpec["Country"] == 'Malawi']['Life_Expectancy']


# In[29]:


lifeExpec[lifeExpec["Country"] == 'Lesotho']['Life_Expectancy']


# In[30]:


lifeExpec[lifeExpec["Country"] == 'Zambia']['Life_Expectancy']


# In[31]:


lifeExpec[lifeExpec["Country"] == 'Zimbabwe']['Life_Expectancy']


# Based on the analysis above, the life Expectancy low is mainly because these data are from developing country. Here we will keep the data. 

# In[32]:


## check  Adult_Mortality
lifeExpec[lifeExpec["Adult_Mortality"] > 500]


# The boxplot shows the data the mortality in some countries, such as Zimbabwe, Zambia, Lesotho, Botswana,Malawi are all high. That may be caused by some factors. We will do specific analysis on these countries to see if we can dig out some important information. Here, we mainly investigate Central African Republic, Eritrea,Haiti, Sierra Leone to see if the mortality is abnormal.

# In[35]:


lifeExpec[lifeExpec["Country"] == 'Central African Republic']


# Adult Mortality in Central African Republic looks very abnormal, the Adult Mortality between 2000 and 2003 are very low, arount 50, however, after 2004, it increased dramatically. Based on data from world bank, https://data.worldbank.org/indicator/SP.DYN.AMRT.MA?locations=CF, the Adult Mortality is always above 390. So the Adult Mortality in 2000, 2001, 2002, 2003 and 2006 are not correct. Here, the Adult Mortality of those years will be modified based on world bank data. 

# In[47]:


## make a copy
lifeExpecCopy = lifeExpec


# In[72]:


## 2000 
lifeExpec.loc[527,'Adult_Mortality'] = 540.27   
## 2001
lifeExpec.loc[526,'Adult_Mortality'] = 548.42
## 2002
lifeExpec.loc[525,'Adult_Mortality'] = 556.57
##2003
lifeExpec.loc[524,'Adult_Mortality'] = 548.02
## 2006
lifeExpec.loc[521,'Adult_Mortality'] = 522.39


# In[74]:


lifeExpec[lifeExpec["Country"] == 'Eritrea']


# The Adult_Mortality of Eritrea in 2005 is abnormal. Based on data from world bank, https://data.worldbank.org/indicator/SP.DYN.AMRT.MA?locations=CF, the Adult Mortality is 202.96. So we change the value of the Adult_Mortality of Eritrea in 2005 into 202.96.

# In[75]:


##2005 Eritrea
lifeExpec.loc[860,'Adult_Mortality'] = 202.96


# In[ ]:


## check each country for "Adult_Mortality"


# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


lifeExpecOutput = lifeExpec["Life_Expectancy"]


# In[19]:


lifeDropYearCountry = lifeExpec.drop(["Country", "Year","Status"], axis = 1)


# In[21]:


lifeDropYearCountry.columns


# In[24]:


## normalize data
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(lifeDropYearCountry )


# In[30]:


len(data_scaled)


# In[ ]:





# In[ ]:




