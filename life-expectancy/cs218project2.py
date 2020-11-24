#!/usr/bin/env python
# coding: utf-8

#  
#   <h1><center> Analyze life expectancy</center></h1>

# The life expectancy data is from kaggle, https://www.kaggle.com/kumarajarshi/life-expectancy-who. It contains 193 countries data. There are 22 columns in the dataset. All predicting variables was then divided into several broad categories:Immunization related factors, Mortality factors, Economical factors and Social factors.Through the analysis, we want to answer the following questions:
# - The life expectancy between developed countries and developing countries are siginificant different?
# - Which factors affect the life expectancy?
# - Will Immunization factor play a role in the average age of life expectancy?
# - How adult mortality and infant mortality affect life expectancy?
# - How economic factors affect life expectancy?
# - Do social factors have same effect on life expectancy?
# - How about the countries with very short life expectancy? Are the factors the same with developing countries?
# 

# In[1]:


from google.cloud import storage
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler


# In[2]:


lifeExpec = pd.read_csv('gs://life2/LifeExpectancyData.csv', sep=",")


# In[3]:


lifeExpec.head()


# In[4]:


lifeExpec.shape


# In[5]:


lifeExpec.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Expectancy","Adult Mortality":"Adult_Mortality",
                   "infant deaths":"Infant_Deaths","percentage expenditure":"Percentage_Exp","Hepatitis B":"HepatitisB",
                  "Measles ":"Measles"," BMI ":"BMI","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",
                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_10to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",
                   "Total expenditure":"Tot_Exp"},inplace=True)


# In[6]:


lifeExpec.columns


# In[7]:


lifeExpec.isnull().sum()


# In[8]:


country_list = lifeExpec.Country.unique()
fill_list = ['Life_Expectancy','Adult_Mortality','Alcohol','HepatitisB','BMI','Polio','Tot_Exp','Diphtheria','GDP','Population','thinness_10to19_years','thinness_5to9_years','Income_Comp_Of_Resources','Schooling']


# In[9]:


for country in country_list:
    lifeExpec.loc[lifeExpec['Country'] == country,fill_list] = lifeExpec.loc[lifeExpec['Country'] == country,fill_list].interpolate()


# In[10]:


# Drop remaining null values after interpolation.
lifeExpec.dropna(inplace=True)
lifeExpec.shape


# In[11]:


countryNames = lifeExpec.Country.unique()
countryNames.size


# In[12]:


life = lifeExpec.drop('Year', axis = 1)


# In[13]:


decribe = life.describe()


# In[14]:


decribe.round(2)


# In[15]:


round(lifeExpec[['Status','Life_Expectancy']].groupby(['Status']).mean(),2)


# In[16]:


lifeExpec[lifeExpec['Status'] == "Developing"].shape[0]/lifeExpec.shape[0]


# In[17]:


lifeExpec[lifeExpec['Status'] == "Developed"].shape[0]/lifeExpec.shape[0]


# The percentage of developing countries among all countries is 86.31%, while developed countries only account of 13.69% of total data. This is an unbalanced dataset, the analysis result will be dominated by the majority class, developing countries. So, we will analayze developing countries and developed countries separately. 

# In[18]:


## life Expectancy vs status 
l =(round(lifeExpec.groupby('Status')['Life_Expectancy'].mean(), 2).to_numpy())
plt.figure(figsize=(6,6))
plt.bar(lifeExpec.groupby('Status')['Status'].count().index,lifeExpec.groupby('Status')['Life_Expectancy'].mean(), color = "lightsteelblue")
plt.xlabel("Status",fontsize=12,fontweight='bold')
plt.ylabel("Avg Life Expectancy",fontsize=12,fontweight='bold')
plt.ylim(0, 85)
plt.title("Life Expectancy vs Status", fontsize=15,fontweight='bold')
for i, v in enumerate(l):
    plt.text(i +0.05,v +2,str(v), color='blue', fontweight='bold' )
plt.show()


# In[19]:


stats.ttest_ind(lifeExpec.loc[lifeExpec['Status']=='Developed','Life_Expectancy'],lifeExpec.loc[lifeExpec['Status']=='Developing','Life_Expectancy'])


# In[20]:


## check outliers using boxplot
col_dict = {'Life_Expectancy':1,'Adult_Mortality':2,'Infant_Deaths':3,'Alcohol':4,'Percentage_Exp':5,'HepatitisB':6,'Measles':7,'BMI':8,'Under_Five_Deaths':9,'Polio':10,'Tot_Exp':11,'Diphtheria':12,'HIV/AIDS':13,'GDP':14,'Population':15,'thinness_10to19_years':16,
            'thinness_5to9_years':17,'Income_Comp_Of_Resources':18,'Schooling':19}
plt.figure(figsize=(20,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(lifeExpec[variable],whis=1.5)
                     plt.title(variable)

plt.show()


# The boxplot results shows there are outliers exist in each variables. Next, we will take a look at each indivual variable, 
# to see if we need to normalize the variables.

# In[21]:


## check life Expecancy
lifeExpec[lifeExpec["Life_Expectancy"] < 45]


# In[22]:


lifeExpec[lifeExpec["Country"] == 'Sierra Leone']['Life_Expectancy']


# In[23]:


lifeExpec[lifeExpec["Country"] == 'Malawi']['Life_Expectancy']


# In[24]:


lifeExpec[lifeExpec["Country"] == 'Lesotho']['Life_Expectancy']


# In[25]:


lifeExpec[lifeExpec["Country"] == 'Zambia']['Life_Expectancy']


# In[26]:


lifeExpec[lifeExpec["Country"] == 'Zimbabwe']['Life_Expectancy']


# Based on the analysis above, the life Expectancy low is mainly because these data are from developing country. Here we will 
# keep the data.
# 

# In[27]:


## check  Adult_Mortality
lifeExpec[lifeExpec["Adult_Mortality"] > 500]


# The boxplot shows the data the mortality in some countries, such as Zimbabwe, Zambia, Lesotho, Botswana,Malawi are all high. 
# That may be caused by some factors. We will do specific analysis on these countries to see if we can dig out some important 
# information. Here, we mainly investigate Central African Republic, Eritrea,Haiti, Sierra Leone to see if the mortality is 
# abnormal.

# In[28]:


lifeExpec[lifeExpec["Country"] == 'Central African Republic']


# Adult Mortality in Central African Republic looks very abnormal, the Adult Mortality between 2000 and 2003 are very low, 
# arount 50, however, after 2004, it increased dramatically. Based on data from world bank, 
# https://data.worldbank.org/indicator/SP.DYN.AMRT.MA?locations=CF, the Adult Mortality is always above 390. So the Adult 
# Mortality in 2000, 2001, 2002, 2003 and 2006 are not correct. Here, the Adult Mortality of those years will be modified 
# based on world bank data. 

# In[29]:


## make a copy
lifeExpecCopy = lifeExpec


# In[30]:


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


# In[31]:


lifeExpec[lifeExpec["Country"] == 'Eritrea']


# The Adult_Mortality of Eritrea in 2005 is abnormal. Based on data from world bank, 
# https://data.worldbank.org/indicator/SP.DYN.AMRT.MA, the Adult Mortality is 202.96. So we change the value of the Adult_Mortality of Eritrea in 2005 into 202.96.

# In[32]:


##2005 Eritrea
lifeExpec.loc[860,'Adult_Mortality'] = 202.96


# In[33]:


lifeExpec[lifeExpec["Country"] == 'Haiti']


# From above chart, we could see, the Adult_Mortality in Haiti in 2010,and from 2000 to 2006 are abnormal. Based on data from 
# world bank, https://data.worldbank.org/indicator/SP.DYN.AMRT.MA?locations=HT, the Adult Mortality is decreasing, but the the 
# Adult_Mortality is still above 250. So we need to modify the data based on world bank.
# 

# In[34]:


## 2000 
lifeExpec.loc[1137,'Adult_Mortality'] = 337.64  
## 2001
lifeExpec.loc[1136,'Adult_Mortality'] = 336.45
## 2002
lifeExpec.loc[1135,'Adult_Mortality'] = 335.25
##2003
lifeExpec.loc[1134,'Adult_Mortality'] = 329.85
## 2004
lifeExpec.loc[1133,'Adult_Mortality'] = 324.45
##2005
lifeExpec.loc[1132,'Adult_Mortality'] = 319.05
## 2006
lifeExpec.loc[1131,'Adult_Mortality'] = 313.65
## 2010
lifeExpec.loc[1127,'Adult_Mortality'] = 293.38


# In[37]:


lifeExpec[lifeExpec["Country"] == 'Sierra Leone']


# From above chart, we could see, the Adult_Mortality in Sierra Leone in 2003, 2005, 2007, 2013 are abnormal. Based on data from
# world bank, https://data.worldbank.org/indicator/SP.DYN.AMRT.MA, the Adult_Mortality is still above 400. So we need to modify 
# the data based on world bank.
# 

# In[38]:


## modify the Adult_Mortality in Sierra Leone
## 2003
lifeExpec.loc[2309,'Adult_Mortality'] = 511.26
## 2005
lifeExpec.loc[2307,'Adult_Mortality'] = 483.17
## 2007
lifeExpec.loc[2305,'Adult_Mortality'] = 455.08
##2013
lifeExpec.loc[2299,'Adult_Mortality'] = 409.65


# In[39]:


lifeExpec[lifeExpec["Country"] == 'Sierra Leone']


# Infant death are the number of Infant Deaths per 1000 population. Based on the boxplot, we can see there is some value greater than 1000. These values must be not correct. The details shows below.
# 

# In[40]:


lifeExpec[lifeExpec["Infant_Deaths"] > 750]


# The above shows that the infant death in India is greater than 1000. Based on data from world bank, https://data.worldbank.org/indicator/SP.DYN.IMRT.IN?locations=IN, the Adult_Mortality is below 400. So, we need to modify the infant death rate in India.

# In[41]:


## 2000
lifeExpec.loc[1201,'Infant_Deaths'] = 66.7
## 2001
lifeExpec.loc[1200,'Infant_Deaths'] = 64.4
## 2002
lifeExpec.loc[1199,'Infant_Deaths'] = 62.2
##2003
lifeExpec.loc[1198,'Infant_Deaths'] = 60
## 2004
lifeExpec.loc[1197,'Infant_Deaths'] = 57.8
## 2005
lifeExpec.loc[1196,'Infant_Deaths'] = 55.7
## 2006
lifeExpec.loc[1195,'Infant_Deaths'] = 53.7
##2007
lifeExpec.loc[1194,'Infant_Deaths'] = 51.6
## 2008
lifeExpec.loc[1193,'Infant_Deaths'] = 49.4
## 2009
lifeExpec.loc[1192,'Infant_Deaths'] = 47.3
## 2010
lifeExpec.loc[1191,'Infant_Deaths'] = 45.1
##2011
lifeExpec.loc[1190,'Infant_Deaths'] = 43
##2012
lifeExpec.loc[1189,'Infant_Deaths'] = 40.9
## 2013
lifeExpec.loc[1188,'Infant_Deaths'] = 38.8
## 2014
lifeExpec.loc[1187,'Infant_Deaths'] = 36.9


# In[42]:


lifeExpec[lifeExpec["Alcohol"] > 16]


# It seems that there is only two countries whose alcohol values is higher,so a investigatation will be done below.

# In[43]:


lifeExpec[lifeExpec["Country"] == 'Belarus']["Alcohol"]


# In[44]:


lifeExpec[lifeExpec["Country"] == 'Estonia']["Alcohol"]


# The data here looks a ittle bit sketptical, but without concrete reason, no further step is taken at this moment.

# The lifeExpec data looks skeptical based on the boxplot and analysis below. As discribe on the website, Expenditure on health as a percentage of Gross Domestic Product per capita(%). For now, just keep the data here, further investigation will be done.

# In[45]:


lifeExpec["Percentage_Exp"].max()


# In[46]:


lifeExpec["Percentage_Exp"].min()


# In[47]:


lifeExpec[lifeExpec["HepatitisB"] < 5].head()


# HepatitisB: is Hepatitis B (HepB) immunization coverage among 1-year-olds, the HepatitisB rate lower than 5% are from developed country, so we will leave the data as original.

# In[48]:


lifeExpec[lifeExpec["Measles"] > 30000]


# In[49]:


lifeExpec["Measles"].max()


# As the more Measles cases are in developing country, without further evidence that Measles is not correct, we will keep the data as original. 

# In the beggining, we analyze whether developed country or developing country, it make different with the Life_Expectancy. Next we will analyze other factors that could affect the life expendency. Let's see the heat map.

# In[50]:


corr = lifeExpec.corr()
ax = plt.axes()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
ax.set_title("Heat Map",fontsize = 20,fontweight='bold')


# From the heat map above, we could see life Expectancy has negative correlation with Adult_Mortality, Infant_Deaths, Measles, HIV/AIDS, under-five-year death, thinness 1-19 years, thinness 5-9 years. Life Expectancy has positive correlation with alcohol, percentage_Exp, BMI,Diphtheria,GDP,income_comp_Of_Resource, and Schoolng. For poio, population, the correlation is so small, we could not say it's negative or positive. It worth to mention that life expectancy has high correlation with adult_mortality, BMI, Diphtheria, HIV/AIDS,thinness_1-19_years, thiness_5_to_9_years, income_of_resource and schooling. Further analysis is needed to gain more information.

# In[51]:


## divide country into developed and developing counties
lifeExpecDevelping = lifeExpec[lifeExpec["Status"] == "Developing"]
lifeExpecDevelped = lifeExpec[lifeExpec["Status"] == "Developed"]
lifeExpecDevelping.shape


# In[52]:


developing2000 = lifeExpecDevelping[lifeExpecDevelping['Year'] == 2000]['Life_Expectancy'].mean()
developing2015 = lifeExpecDevelping[lifeExpecDevelping['Year'] == 2015]['Life_Expectancy'].mean()
developing2015 - developing2000


# In[53]:


developed2014 = lifeExpecDevelped[lifeExpecDevelped['Year'] == 2014]['Life_Expectancy'].mean()
developed2000 = lifeExpecDevelped[lifeExpecDevelped['Year'] == 2000]['Life_Expectancy'].mean()
developed2014 - developed2000


# The lifespan increases by 6.36 years for developing countries, while for developed countries, it increases by 4.34 years. Note, for developed countries, because the data in 2015 is missing, we are calculating the life expectancy from 2000 to 2014.

# In[54]:


X = np.arange(2)
d1 = [round(developing2000,2),round(developed2000,2)]
d2 = [round(developing2015,2),round(developed2014,2)]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, d1, color = 'lightsteelblue', width = 0.25)
ax.bar(X + 0.25, d2, color = 'orange', width = 0.25)
ax.legend(labels=['2000', '2015'])
ax.set_ylabel("Life Expectancy",fontweight = 'bold')
ax.set_xlabel("Developing Countries and Developed Countries",fontweight = 'bold')
plt.text( 0.0005, d1[0]+2,str(d1[0]), color='blue', fontweight='bold' )
plt.text( 0.2005, d2[0]+2,str(d2[0]), color='blue', fontweight='bold' )
plt.text( 1.0, d1[1]+2,str(d1[1]), color='blue', fontweight='bold' )
plt.text( 1.2005, d2[1]+2,str(d2[1]), color='blue', fontweight='bold' )
plt.title("Life Expectancy Difference in 16 Years", fontweight = 'bold')
plt.show()


# The life expectancy in both developing countries and developed countries are both increased. From the figure above we can see that lifespan in developed countries increased by 4.34 years, while those in developing countries increased by 6.37 years.

# Next we will investigate factors one by one, to see which factor will affect life expectancy and how it will affect. To dive more, the factors will be divided into immunization factors, mortality factors, economic factors, social factors and other health related factors.
# 

# ### Mortality Factor
# - Adult mortality
# - Infant death
# - Under_Five_Deaths

# In[55]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Adult_Mortality'])


# In[56]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Adult_Mortality'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Adult Mortality', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Adult Mortality",  fontsize = 20,fontweight='bold')


# In[57]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Adult_Mortality'])


# In[58]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelped, y = lifeExpecDevelped['Life_Expectancy'], x = lifeExpecDevelped['Adult_Mortality'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Adult_Mortality', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Adult_Mortality in Developed Countries",  fontsize = 12,fontweight='bold')

plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[59]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Adult_Mortality'])


# In[60]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['Adult_Mortality'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Adult_Mortality', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Adult_Mortality",  fontsize = 15,fontweight='bold')

plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# Above, the pearson correlation is used to test the correlation between Life Expectancy and Adult Mortality. The correlation is -0.7028. That means, the higher the mortality in a country, the shorter life expectancy. This applies to both developed countries and developing countries.

# In[61]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Infant_Deaths'])


# In[62]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Infant_Deaths'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Infant Death', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Infant Death",  fontsize = 20,fontweight='bold')


# From above analysis, the correlation between life Expectancy and Infant Death is relatively low, -0.3007. That means, the higher life expectancy, the relatively lower infant death rate. But the correlation is low. There is an interesting phenomenia observed from correlation graph. For developed country, there is almost no relationship between life expectancy and infant death. Only for developing country the infant death may have low relationship with life expectancy. Further analysis are done below.

# In[63]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['Infant_Deaths'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Infant Death', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Infant Death",  fontsize = 20,fontweight='bold')

plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[64]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Infant_Deaths'])


# In[65]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Infant_Deaths'])


# Indeed, some countries, the life expectancy has high correlation with infant death. For these country, lower the infant death could be one factor increase the whole country life expectancy.

# In[66]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Under_Five_Deaths'])


# In[67]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Under_Five_Deaths'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Under Five Deaths', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Under Five Deaths",  fontsize = 20,fontweight='bold')


# In[68]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Under_Five_Deaths'])


# In[69]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Under_Five_Deaths'])


# There is very low relationship with life expectancy and under five death. For developed countries, there is no correlation between life expectancy and under five death.

# In[70]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['Under_Five_Deaths'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Under Five Death', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Under Five Death",  fontsize = 20,fontweight='bold')

plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# Only for a few counties, life expectancy and under five death have higher correlations. For these countries, lower the under five death could improve the life expectancy. We can do further analysis on these countries.
# 

# ### Immunization Factors
# - HepatitisB
# - Polio
# - Diphtheria

# In[71]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['HepatitisB'])


# In[72]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['HepatitisB'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('HepatitisB', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs HepatitisB",  fontsize = 20,fontweight='bold')


# In[73]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['HepatitisB'])


# In[74]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['HepatitisB'])


# In[75]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Polio'])


# In[76]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Polio'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Polio', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Polio",  fontsize = 20,fontweight='bold')


# In[77]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Polio'])


# In[78]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Polio'])


# In[79]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['Polio'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Polio', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Polio",  fontsize = 20,fontweight='bold')

plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[80]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Diphtheria'])


# In[81]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Diphtheria'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Diphtheria', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Diphtheria",  fontsize = 20,fontweight='bold')


# In[82]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Diphtheria'])


# In[83]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Diphtheria'])


# In[84]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['Diphtheria'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Polio', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Diphtheria",  fontsize = 20,fontweight='bold')

plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# ### Economic Factors
#  - Percentage_Exp
#  - Tot_Exp
#  - GDP
#  - Income_Comp_Of_Resources

# In[87]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Percentage_Exp'])


# In[88]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Percentage_Exp'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Percentage Expenditure', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Percentage Expenditure",  fontsize = 15,fontweight='bold')


# In[89]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Percentage_Exp'])


# In[90]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Tot_Exp'])


# In[91]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Tot_Exp'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Total Expediture', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Total Expenditure",  fontsize = 15,fontweight='bold')


# In[92]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Tot_Exp'])


# In[93]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Tot_Exp'])


# In[94]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelped, y = lifeExpecDevelped['Life_Expectancy'], x = lifeExpecDevelped['Tot_Exp'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Total Expediture', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Total Expediture in Developed Countries",  fontsize = 20,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[95]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['GDP'])


# In[96]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['GDP'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('GDP', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs GDP",  fontsize = 20,fontweight='bold')


# In[97]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['GDP'])


# In[98]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['GDP'])


# In[99]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelped, y = lifeExpecDevelped['Life_Expectancy'], x = lifeExpecDevelped['GDP'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('GDP', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs GDP in Developed Countries",  fontsize = 20,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[100]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x= lifeExpecDevelping['GDP'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('GDP', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs GDP in Developing Countries",  fontsize = 20,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[101]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Income_Comp_Of_Resources'])


# In[102]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y= lifeExpec['Life_Expectancy'],x = lifeExpec['Income_Comp_Of_Resources'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 10,fontweight='bold')
plt.xlabel('Income Composition of Resources', fontsize = 10,fontweight='bold')
ax.set_title("Life Expectancy vs Income Composition of Resources",  fontsize = 12,fontweight='bold')


# In[103]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelped, y = lifeExpecDevelped['Life_Expectancy'], x = lifeExpecDevelped['Income_Comp_Of_Resources'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Income Composition of Resources', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Income Composition of Resources in Developed Countries",  fontsize = 12,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[104]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['Income_Comp_Of_Resources'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Income Composition of Resources', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Income Composition of Resources in Developed Countries",  fontsize = 12,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[105]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Income_Comp_Of_Resources'])


# In[106]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Income_Comp_Of_Resources'])


# ### Social factors
# - schooling
# - Population

# In[107]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Schooling'])


# In[108]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y= lifeExpec['Life_Expectancy'],x = lifeExpec['Schooling'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 10,fontweight='bold')
plt.xlabel('Schooling', fontsize = 10,fontweight='bold')
ax.set_title("Life Expectancy vs Schooling",  fontsize = 15,fontweight='bold')


# In[109]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Schooling'])


# In[110]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Schooling'])


# In[111]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['Schooling'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Schooling', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Schooling",  fontsize = 12,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[112]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Population'])


# In[113]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y= lifeExpec['Life_Expectancy'],x = lifeExpec['Population'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 10,fontweight='bold')
plt.xlabel('Population', fontsize = 10,fontweight='bold')
ax.set_title("Life Expectancy vs Population",  fontsize = 15,fontweight='bold')


# In[114]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Population'])


# In[115]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Population'])


# ### Other Health Related Factors
# - Alcohol
# - BMI
# - thinness_1to19_years
# - thinness_5to9_years
# - HIV/AIDS
# - Measles

# In[138]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['Measles'])


# In[139]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Measles'])


# In[140]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Measles'])


# In[141]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['Alcohol'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Alcohol', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Alcohol",  fontsize = 20,fontweight='bold')


# In[142]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['Alcohol'])


# In[143]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['Alcohol'])


# In[144]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelped, y = lifeExpecDevelped['Life_Expectancy'], x = lifeExpecDevelped['Schooling'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Schooling', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Schooling in Developed Countries",  fontsize = 12,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[145]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['BMI'])


# In[146]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['BMI'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('BMI', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs BMI",  fontsize = 20,fontweight='bold')


# In[147]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['BMI'])


# In[148]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['BMI'])


# In[149]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelping, y = lifeExpecDevelping['Life_Expectancy'], x = lifeExpecDevelping['BMI'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('BMI', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs BMI in Developing Countries",  fontsize = 12,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[150]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['thinness_10to19_years'])


# In[151]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['thinness_10to19_years'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('thinness 10 to 19 years', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs thinness 10 to 19 years",  fontsize = 15,fontweight='bold')


# In[152]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['thinness_10to19_years'])


# In[153]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['thinness_10to19_years'])


# In[154]:


ax = plt.axes()
sns.scatterplot(data = lifeExpecDevelped, y = lifeExpecDevelped['Life_Expectancy'], x = lifeExpecDevelped['Schooling'],hue = "Country")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('Schooling', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs Schooling in Developed Countries",  fontsize = 12,fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.06),fancybox=True, shadow=True, ncol=5)


# In[155]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['thinness_5to9_years'])


# In[156]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['thinness_5to9_years'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('thinness 5 to 9 years', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs thinness 5 to 9 years",  fontsize = 15,fontweight='bold')


# In[157]:


lifeExpec['Life_Expectancy'].corr(lifeExpec['HIV/AIDS'])


# In[158]:


ax = plt.axes()
sns.scatterplot(data = lifeExpec, y = lifeExpec['Life_Expectancy'], x = lifeExpec['HIV/AIDS'],hue = "Status")
plt.ylabel('Life Expectancy', fontsize = 15,fontweight='bold')
plt.xlabel('HIV/AIDS', fontsize = 15,fontweight='bold')
ax.set_title("Life Expectancy vs HIV/AIDS",  fontsize = 20,fontweight='bold')


# In[159]:


lifeExpecDevelped['Life_Expectancy'].corr(lifeExpecDevelped['HIV/AIDS'])


# In[160]:


lifeExpecDevelping['Life_Expectancy'].corr(lifeExpecDevelping['HIV/AIDS'])


# In the next session, we will check if the life expectancy is increased from 2000 to 2015.

# In[161]:


# Life_Expectancy through years
plt.figure(figsize=(6,6))
plt.bar(lifeExpec.groupby('Year')['Year'].count().index,lifeExpec.groupby('Year')['Life_Expectancy'].mean(),color='cornflowerblue',alpha=0.65)
plt.xlabel("Year",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy vs Year",fontweight='bold')
plt.show()


# In[ ]:





# In[162]:


# Life_Expectancy through years
plt.figure(figsize=(6,6))
plt.bar(lifeExpecDevelped.groupby('Year')['Year'].count().index,lifeExpecDevelped.groupby('Year')['Life_Expectancy'].mean(),color='cornflowerblue',alpha=0.65)
plt.xlabel("Year",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy in developed countries vs Year",fontweight='bold')
plt.show()


# In[ ]:





# In[163]:


# Life_Expectancy through years
plt.figure(figsize=(6,6))
plt.bar(lifeExpecDevelping.groupby('Year')['Year'].count().index,lifeExpecDevelping.groupby('Year')['Life_Expectancy'].mean(),color='cornflowerblue',alpha=0.65)
plt.xlabel("Year",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy in developing countries vs Year",fontweight='bold')
plt.show()


# For both developed countries and developing countries, the lifespan are increased every year. And from analysis before, we also know that the increase of lifespan in developing countres increased more than those in developed countries. It worth to dig deeper to understand which area has improved for these two kind of countries.

# In[164]:


## check correlation within features
lifeExpec['thinness_5to9_years'].corr(lifeExpec['thinness_10to19_years'])


# In[147]:


lifeExpec['Income_Comp_Of_Resources'].corr(lifeExpec['Schooling'])


# In[148]:


lifeExpec['Infant_Deaths'].corr(lifeExpec['Under_Five_Deaths'])


# In[149]:


lifeExpec['GDP'].corr(lifeExpec['Percentage_Exp'])


# From above analysis and also the heatmap, we could see that the correlation between thinness_10to19_years and thinness_5to9_years is  very high, close to 1. To avoid multicolinearity, we will only keep one of them.

# ## models
# - linear regression
# - random forest 

# In[165]:


## fit linear regression model
response = lifeExpecDevelped["Life_Expectancy"]
predictors = lifeExpecDevelped[['Adult_Mortality','Percentage_Exp', 'GDP','Income_Comp_Of_Resources','Schooling','Population','thinness_10to19_years']]


# In[166]:


model =sm.OLS(response,predictors).fit()
##print (model.params)
print (model.summary())


# In[167]:


response = lifeExpecDevelped["Life_Expectancy"]
predictors1 = lifeExpecDevelped[['Adult_Mortality', 'GDP','Population','thinness_10to19_years']]


# In[168]:


model1 =sm.OLS(response,predictors1).fit()
print (model1.summary())


# In[169]:


from sklearn.model_selection import train_test_split
trainDataDeveloped = lifeExpecDevelped.drop(["Country", "Year",'Status'],axis=1)
##XDeveloped = trainDataDeveloped.drop(['Life_Expectancy'],axis=1)
XDeveloped = lifeExpecDevelped[['Adult_Mortality','Percentage_Exp', 'GDP','Income_Comp_Of_Resources','Schooling','Population','thinness_10to19_years']]
yDeveloped = trainDataDeveloped["Life_Expectancy"]
X_train, X_test, y_train, y_test = train_test_split(XDeveloped, yDeveloped, test_size=0.30, random_state=101)


# In[170]:


from sklearn.linear_model import LinearRegression


# In[171]:


Linear_model= LinearRegression()


# In[172]:


Linear_model.fit(X_train,y_train)


# In[173]:


predictions1=Linear_model.predict(X_test)


# In[174]:


Linear_model.coef_


# In[175]:


lifeExpec.columns


# In[176]:


trainData = lifeExpec.drop(["Country", "Year",'Status'],axis=1)
X = trainData.drop(['Life_Expectancy'],axis=1)
y = trainData["Life_Expectancy"]


# In[177]:


trainData2 = lifeExpec.drop(["Country", "Year",'Status','Schooling','thinness_10to19_years'],axis=1)
X = trainData2.drop(['Life_Expectancy'],axis=1)
y = trainData2["Life_Expectancy"]


# In[178]:


trainData2.columns


# In[179]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[180]:


## normalize data
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(trainData2 )


# In[181]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[182]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[183]:


# Get numerical feature importances
importances = list(regressor.feature_importances_)
feature_list = list(X.columns)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[184]:


trainDataDeveloped = lifeExpecDevelped.drop(["Country", "Year",'Status'],axis=1)
XDeveloped = trainDataDeveloped.drop(['Life_Expectancy'],axis=1)
yDeveloped = trainDataDeveloped["Life_Expectancy"]


# In[185]:


## normalize data
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(trainDataDeveloped )


# In[186]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(XDeveloped, yDeveloped, test_size=0.2, random_state=0)


# In[187]:


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[188]:


# Get numerical feature importances
importances = list(regressor.feature_importances_)
feature_list = list(X.columns)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[189]:


plt.figure(figsize = (4,4))
feat_importances = pd.Series(regressor.feature_importances_, index=X_train.columns)
feat_importances.nlargest(5).plot(kind='barh', color = "lightsteelblue")
plt.title("Feature Importance in Developed Countries",fontsize = 10, fontWeight = 'bold')


# In[190]:


trainDataDeveloping = lifeExpecDevelping.drop(["Country", "Year",'Status'],axis=1)
XDeveloping = trainDataDeveloping.drop(['Life_Expectancy'],axis=1)
yDeveloping = trainDataDeveloping["Life_Expectancy"]


# In[191]:


## normalize data
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(trainDataDeveloping )

X_train, X_test, y_train, y_test = train_test_split(XDeveloping, yDeveloping, test_size=0.2, random_state=0)
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Get numerical feature importances
importances = list(regressor.feature_importances_)
#feature_list = list(X.columns)
# List of tuples with variable and importance
#feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
#feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[192]:


plt.figure(figsize = (4,4))
feat_importances = pd.Series(regressor.feature_importances_, index=X_train.columns)
feat_importances.nlargest(5).plot(kind='barh',color = "lightsteelblue")
plt.title("Feature Importance in Developing Countries",fontsize = 10, fontWeight = 'bold')


# In[193]:


## get the lifeexpectancy is lower than 50
lifeExpectLow = lifeExpec[lifeExpec["Life_Expectancy"] < 50]


# In[194]:


len(lifeExpectLow["Country"].unique())


# In[195]:


trainDataDevelopingLow = lifeExpectLow.drop(["Country", "Year",'Status'],axis=1)
XDevelopingLow = trainDataDevelopingLow.drop(['Life_Expectancy'],axis=1)
yDevelopingLow = trainDataDevelopingLow["Life_Expectancy"]


# In[196]:


## normalize data
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(trainDataDevelopingLow )

X_train, X_test, y_train, y_test = train_test_split(XDevelopingLow, yDevelopingLow, test_size=0.2, random_state=0)
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Get numerical feature importances
importances = list(regressor.feature_importances_)
feature_list = list(X.columns)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[197]:


lifeExpectLow.describe()


# In[198]:


lifeExpectLow.Country.unique()


# In[ ]:





# In[199]:


plt.figure(figsize = (4,4))
feat_importances = pd.Series(regressor.feature_importances_, index=X_train.columns)
feat_importances.nlargest(5).plot(kind='barh',color = "lightsteelblue")
plt.title("Feature Importance in low Life Expetancy Countries", fontSize = 10,fontWeight = 'bold')


# In[200]:


lowlife = round(lifeExpectLow["Life_Expectancy"].mean(),2)
devlife = round(lifeExpecDevelping["Life_Expectancy"].mean())
lowGDP = round(lifeExpectLow["GDP"].mean(),2)
devGDP = round(lifeExpecDevelping["GDP"].mean(),2)
lowMortality = round(lifeExpectLow["Adult_Mortality"].mean(),2)
devMortality = round(lifeExpecDevelping["Adult_Mortality"].mean(),2)


# In[201]:


low = [lowlife,lowGDP,lowMortality]
dev = [devlife,devGDP,devMortality ]


# In[202]:


X = np.arange(1)
fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0,0,1,1])
ax.bar( 0.00, low[0], color = 'lightsteelblue', width = 0.025)
ax.bar( 0.1, dev[0], color = 'orange', width = 0.025)
ax.set_ylabel("life Expectancy",fontWeight = "bold")
ax.legend(loc ="upper left",labels=['Low Life Expectancy Countries', 'Developing Countries'])
ax.set_title("Life Expectancy Comparison",fontWeight = "bold")
plt.text( 0, low[0]+2,str(low[0]), color='blue', fontweight='bold' )
plt.text( 0.1, dev[0]+1,str(dev[0]), color='blue', fontweight='bold' )


# In[203]:


X = np.arange(1)
fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0,0,1,1])
ax.bar( 0.00, low[1], color = 'lightsteelblue', width = 0.025)
ax.bar( 0.1, dev[1], color = 'orange', width = 0.025)
ax.legend(loc = "upper left", labels=['Low Life Expectancy Countries', 'Developing Countries'])
ax.set_ylabel("GDP", fontWeight = "bold")
ax.set_title("GDP Comparison", fontWeight = "bold")
plt.text( 0, low[1]+2,str(low[1]), color='blue', fontweight='bold' )
plt.text( 0.1, dev[1]+1,str(dev[1]), color='blue', fontweight='bold' )


# In[204]:


X = np.arange(1)
fig = plt.figure(figsize=(4,4))
ax = fig.add_axes([0,0,1,1])
ax.bar( 0.00, low[2], color = 'lightsteelblue', width = 0.025)
ax.bar( 0.1, dev[2], color = 'orange', width = 0.025)
ax.legend(labels=['Low Life Expectancy Countries', 'Developing Countries'])
ax.set_title("Adult Mortality Comparison",fontWeight = "bold")
ax.set_ylabel("Adult Mortality",fontWeight = "bold")
plt.text( 0, low[2]+2,str(low[2]), color='blue', fontweight='bold' )
plt.text( 0.1, dev[2]+1,str(dev[2]), color='blue', fontweight='bold' )


# - From the analysis above, we observed that there is a significant difference in lifespan between developed countries and developing countries. And there is also an obvious phenomenon in our dataset.  Developing countries account for 86.3% of all the dataset, while developed countries only account for  13.7% of the total data. The two classes, developed countries and developing countries, are not equally represented in our dataset. Our learning algorithm may not generalize the behavior of minority classes, which are developed countries. In addition, based on the result of correlation and scatter plot shown in figure above, we could conclude that the factors that influence the life expectancy between developed countries and developed countries are different.
# - For developed countries, the most important factors that influence life expectancy are GDP,Income_Comp_Of_Resources, and adult matility. For developing countries, HIV, adult mortality are very important. If want to improve the average life expectancy in these two categories of countries, we shoould work on specific area accordingly.
# - Developing countries are a broad concept, so we want to dig deep to know if we could do something for the every short life expecancy countries. The results are surprise and make sense. These countries are too poor, they are far behind the average level of developing countries, see the comparison in above figures.
# - In conclusion, we could gather more information from this dataset, such as why some countries increase more in life expectancy? why some are not? The more information we have, the more conclusion we could draw. Therefore, we ccould help these countries to improve their life expectancy.

# In[ ]:




