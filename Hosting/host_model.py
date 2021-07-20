# coding: utf-8

# # TCS ION Internship - HR-Salary Prediction Dashboard
# ## By Hussain
#
# ### Objective : The objective of this project is to build a salary prediction dashboard for HRs

# In[44]:


# importing libraries
import pandas as pd

# In[45]:

data = pd.read_csv('/home/salaryprediction/data.csv')
print("data :", data.shape)


# In[46]:


data.info()


# In[47]:


# Import necessary libraries
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

# In[48]:


# Proceed to model building
X=data.drop(['Salary'],axis=1)
y=data['Salary']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=32)


# In[49]:

# Prediction using DecisionTreeClassifier
dt_model = DecisionTreeClassifier()

#Fitting the model
m=dt_model.fit(X_train, y_train)
y_pred = m.predict(X_test)

#Save model using pickle
import pickle
pickle.dump(m,open('/home/salaryprediction/salary_model.pkl','wb'))
print('model saved locally using pickle')


