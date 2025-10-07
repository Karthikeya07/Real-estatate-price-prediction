#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#test
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import math
from matplotlib.ticker import MaxNLocator
plt.style.use('fivethirtyeight')
from IPython.display import HTML
from subprocess import check_output
import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)
import warnings 
warnings.simplefilter('ignore')


# In[ ]:


#To read data uploaded int0 collab
data=pd.read_csv("dataset_pric.csv")
Rdata=data.copy()


# In[ ]:


#To shoe first 5 rows of data
data.head()


# In[ ]:


#To show last 5 rows of data
data.tail()


# In[ ]:


#To get no of rows and columns of data
data.shape


# In[ ]:


#To get list of columns in data
cols=list(data.columns)
cols


# In[ ]:


#Drop the columns not required
newdata = data.drop(['area_type','society','balcony','availability'],axis='columns')
Rdata=Rdata.dropna(subset=['size','total_sqft'])


# In[ ]:


#Checking if there are null values
newdata.isnull().sum()


# In[ ]:


#shape of new data
newdata.shape


# In[ ]:


#Dropping rows with null values
NNdata= newdata.dropna()


# In[ ]:


#Checking if there are null values
NNdata.isnull().sum()


# In[ ]:


#shape of new data
NNdata.shape


# In[ ]:


#To check data types of all elements
NNdata.dtypes


# In[ ]:


#Unique values in Non NUll data size column
NNdata['size'].unique()


# In[ ]:


#Adding new bhk column of int data type to convert string data type of size column
NNdata['bhk'] = NNdata['size'].apply(lambda x: int(x.split(' ')[0]))
Rdata['bhk'] = Rdata['size'].apply(lambda x: int(x.split(' ')[0]))
NNdata.head()


# In[ ]:


#checking Retrieval data
Rdata.head()


# In[ ]:


#Unique values in Non NUll data Total_sqft column

NNdata['total_sqft'].unique()


# In[ ]:


#All are string values and some even have a range which is to be corrected
#lets define a function so that it can be reused for every row
def convert_sqft_to_float(x):

  tokens = x.split('-')
  if len(tokens) == 2:
    return(float(tokens[0])+float(tokens[1]))/2
  try:
    return float(x)
  except:
    return None
            


# In[ ]:


#Applying the function to every row
NNdata['total_sqft'] = NNdata['total_sqft'].apply(convert_sqft_to_float)
Rdata['total_sqft'] = Rdata['total_sqft'].apply(convert_sqft_to_float)


# In[ ]:


#A function to check if there are any floats
def is_float(x):
   try:
     float(x)
   except:
     return False
   return True    


# In[ ]:


NNdata[~NNdata['total_sqft'].apply(is_float)].head(12)


# In[ ]:


# Creating a new column to find the price per sq-ft
#We multiply the price here with 100000 since its in lakhs
NNdata['Price_per_sqft'] = NNdata['price']*100000/NNdata['total_sqft']
NNdata.head()


# In[ ]:


#Refining location column so that all the locations with less no of properties can be categeorized to 1 so to avoid imporper training
NNdata['location'].unique()


# In[ ]:


#to remove any left spaces and right spaces in locations
NNdata['location'] = NNdata['location'].apply(lambda x: x.strip())


# In[ ]:


#To check count of no of rows for each location
locations = NNdata.groupby('location')['location'].agg('count').sort_values(ascending=False)
locations


# In[ ]:


#lets group all the locations with very low numbers into others categeory
NNdata['location'] = NNdata.location.apply(lambda x: 'other' if x in locations[locations<10] else x)


# In[ ]:


#Removing exceptional houses with abnormally high bedroom size OUTLIERS
NNdata=NNdata[~(NNdata['total_sqft']/NNdata['bhk']<300)]


# In[ ]:


NNdata.Price_per_sqft.describe()


# In[ ]:


#Since max price per sqft is abnormally high
#we need to remove those outliers also
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced_df = subdf[(subdf.Price_per_sqft>(m-st)) & (subdf.Price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df],ignore_index=True)
    return df_out 

NNdata= remove_pps_outliers(NNdata)  


# In[ ]:


NNdata.head()


# In[ ]:


#Drop size column and price per sft column since sizer per sft can be calculated by total price /sft
#and size since we already added bhk column
NNdata =NNdata.drop(['size','Price_per_sqft'],axis='columns')
NNdata.head()


# In[ ]:


#To find correlation of features
Cdata=NNdata.copy()
#To conver location values to numeric
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
Corelationlist=Cdata.corr()['price'].abs().tolist()
Cdata.corr()['price'].abs().sort_values(ascending=False)


# In[ ]:


#Use one hot encoding since it will be better while taking user inputs than standard scaler
dummies = pd.get_dummies(NNdata.location)
dummies.head()


# In[ ]:


NNdata= pd.concat([NNdata, dummies],axis="columns")
NNdata.head()


# In[ ]:


#since other is having NUll values lets remove it also along with location column
NNdata=NNdata.drop(['location'],axis='columns')
NNdata.head()


# In[ ]:


NNdata.shape


# In[ ]:


#Seperaring independent variables into X
X =NNdata.drop(['price'], axis='columns')
X


# In[ ]:


#dependent variables into Y
Y =NNdata['price']
Y


# In[ ]:


#splitting into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=10)


# In[ ]:


#Testing with different algorithms
All_Algos=[]


# In[ ]:


#1 linear regression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
LR= LinearRegression()
LR.fit(X_train, y_train)
LRA=LR.score(X_test,y_test)
LRA=LRA*100
CvLR=cross_val_score(LinearRegression(),X,Y, cv=cv)
LRAvg=(sum(CvLR)/len(CvLR))*100
print("Linear regression accuracy is",LRA,"\n and the avg accuracy for linear regression after cross val is ",LRAvg )
All_Algos.append([LRAvg,'LinearRegression'])
print("added for comparison ")


# In[ ]:


#2 Decision Tree
from sklearn.tree import DecisionTreeRegressor
DT= DecisionTreeRegressor(random_state=0)
DT.fit(X_train, y_train)
DTA=DT.score(X_test,y_test)
DTA=DTA*100
CvDT=cross_val_score(DecisionTreeRegressor(),X,Y, cv=cv)
DTAvg=(sum(CvDT)/len(CvDT))*100
print("Decision Tress Accuracy is",DTA,"\n and the avg accuracy for Decision Tree after cross val is ",DTAvg )
All_Algos.append([DTAvg,"DecisionTree"])
print("added for comparison ")


# In[ ]:


#3 MLP regressor
from sklearn.neural_network import MLPRegressor
MLPR= MLPRegressor(random_state=1, max_iter=500)
MLPR.fit(X_train, y_train)
MLPRA=MLPR.score(X_test,y_test)
MLPRA=MLPRA*100
CvMLPR=cross_val_score(MLPRegressor(random_state=1, max_iter=500),X,Y, cv=cv)
MLPRAvg=(sum(CvMLPR)/len(CvMLPR))*100
print("MLPRegressor Accuracy is",MLPRA,"\n and the avg accuracy for MLPRegressor after cross val is ",MLPRAvg )
All_Algos.append([MLPRAvg,"MLPRegressor"])
print("added for comparison ")


# In[ ]:


#4 Ridge regression
from sklearn.linear_model import Ridge
RR= Ridge(alpha=1.0)
RR.fit(X_train, y_train)
RRA=RR.score(X_test,y_test)
RRA=RRA*100
CvRR=cross_val_score(Ridge(alpha=1.0),X,Y, cv=cv)
RRAvg=(sum(CvRR)/len(CvRR))*100
print("MLPRegressor Accuracy is",RRA,"\n and the avg accuracy for MLPRegressor after cross val is ",RRAvg )
All_Algos.append([RRAvg,"RRegressor"])
print("added for comparison ")


# In[ ]:


#5 Linear model Lasso Regression
from sklearn import linear_model
LMLR=linear_model.Lasso(alpha=0.1)
LMLR.fit(X_train, y_train)
LMLRA=LMLR.score(X_test,y_test)
LMLRA=LMLRA*100
CvLMLR=cross_val_score(linear_model.Lasso(alpha=0.1),X,Y, cv=cv)
LMLRAvg=(sum(CvLMLR)/len(CvLMLR))*100
print("LassoRegressor Accuracy is",LMLRA,"\n and the avg accuracy for LassoRegressor after cross val is ",LMLRAvg )
All_Algos.append([LMLRAvg,"LassoRegressor"])
print("added for comparison ")


# In[ ]:


#6 Random Forest
from sklearn.ensemble import RandomForestRegressor
RF= RandomForestRegressor(max_depth=10, random_state=0)
RF.fit(X_train, y_train)
RFA=RF.score(X_test,y_test)
RFA=RFA*100
CvRF=cross_val_score(RandomForestRegressor(max_depth=10, random_state=0),X,Y, cv=cv)
RFAvg=(sum(CvRF)/len(CvRF))*100
print("RandomForest Accuracy is",RFA,"\n and the avg accuracy for RandomForest after cross val is ",RFAvg )
All_Algos.append([RFAvg,"RandomForest"])
print("added for comparison ")


# In[ ]:


#7KNNRegressor
from sklearn.neighbors import KNeighborsRegressor
KNN= KNeighborsRegressor(n_neighbors=4)
KNN.fit(X_train, y_train)
KNNA=KNN.score(X_test,y_test)
KNNA=KNNA*100
CvKNN=cross_val_score(KNeighborsRegressor(n_neighbors=4),X,Y, cv=cv)
KNNAvg=(sum(CvKNN)/len(CvKNN))*100
print("KNNRegressor Accuracy is",KNNA,"\n and the avg accuracy for KNNRegressor after cross val is ",KNNAvg )
All_Algos.append([KNNAvg,"KNNRegressor"])
print("added for comparison ")


# In[ ]:


#To find out the best Accuracy algorithm
Best=sorted(All_Algos)[::-1][0]
All_Names=['LinearRegression',"DecisionTree","MLPRegressor","RRegressor",'LassoRegressor',"RandomForest","KNNRegressor"]
All_models=[LR,DT,MLPR,RR,LMLR,RF,KNN]
BA=All_models[All_Names.index(Best[1])]
print(Best[1],"is best algorithm with vg accuracy",Best[0])
print("The selected model is ",BA)


# In[ ]:


#Function to predict values
def predict_price(location, sqft,bath, bhk):
  
  try:
    loc_index=np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0 :
      x[loc_index] = 1 
    return BA.predict([x])[0]
  except:
    print("please check inputs")
    return 0  


# In[ ]:


1#To take inputs
pred=0
while(pred==0):
  try:
    l=str(input("Enter the location"))
    s=int(input("Enter required sq ft"))
    b=int(input("enter no of bathrooms"))
    bh=int(input("enter the no of bedrooms"))
    pred=predict_price(l,s,b,bh)
  except:
    print("Please check input data type")  
print("The predicted price for this loaction is ",pred*100000,"₹")  


# **Prediction done, Now to retrieve and show data to the user based on the given data set**
# 
# 
# 

# In[ ]:


#To check retrieval Data
Rdata['price']=Rdata["price"]*100000
Rdata.head()


# In[ ]:


#The list of properties close to given size and in specified  place, with required  bedrooms and bathrooms specifications
Fa=int(input("enter range of change in area thats ok for user"))
List_of_all_properties_with_specs=Rdata[(Rdata['location']==l)&(Rdata['bhk']==bh)&(Rdata['bath']==b)&(Rdata['total_sqft']<=(s+Fa))&(Rdata['total_sqft']>=(s-Fa))]
List_of_all_properties_with_specs


# In[ ]:


#Returning all the ready to move into properties in specified area with price less than or equal to predicted price
Predicted_Price_per_sqft=(pred*100000)/s
List_of_all_properties_ready_to_move_with_predicted_ppsft=List_of_all_properties_with_specs[(List_of_all_properties_with_specs['availability']=="Ready To Move")&((List_of_all_properties_with_specs['price'])<=(List_of_all_properties_with_specs['total_sqft']*Predicted_Price_per_sqft))]
List_of_all_properties_ready_to_move_with_predicted_ppsft


# In[ ]:


#To return all properties which are over priced
List_of_all_properties_over_priced=List_of_all_properties_with_specs[((List_of_all_properties_with_specs['price'])>(List_of_all_properties_with_specs['total_sqft']*Predicted_Price_per_sqft))]
List_of_all_properties_over_priced


# In[ ]:


#Check dummies columns for all location names
Locs=list(dummies.columns)


# In[ ]:


#user inputs the area,bedrooms,bathrooms and his budget and areas he can afford are shown
budget=int(input("Enter the budget"))
Locations_in_budget=[]
for i in Locs:
  pred1=predict_price(i,s,b,bh)
  if((pred1*100000)<=budget):
    Locations_in_budget.append(i)
    print(i)
print("No of areas in budget =",len(Locations_in_budget))    


# In[ ]:


#returning all the properties with specified parameters with price less than or equal to budget
List_of_all_properties_with_specs_in_budget_locations=Rdata[(Rdata['location'].isin(Locations_in_budget))&(Rdata['bhk']==bh)&(Rdata['bath']==b)&(Rdata['total_sqft']<=(s+Fa))&(Rdata['total_sqft']>=(s-Fa))&(Rdata['price']<=budget)]
List_of_all_properties_with_specs_in_budget_locations


# In[ ]:




