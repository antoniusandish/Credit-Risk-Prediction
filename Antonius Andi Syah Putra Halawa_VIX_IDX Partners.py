#!/usr/bin/env python
# coding: utf-8

# # FINAL TASK_DATA SCIENTIST ID/X PARTNERS

# Antonius Andi Syah Putra Halawa

# ## Credit Risk Prediction

# Data Science Workﬂow
# * Business Understanding
# * Analytic Approach
# * Data Requirements
# * Data Collection
# * Data Understanding
# * Data Preparation
# * Exploratory Data Analysis
# * Model Building
# * Model Evaluation
# * Model Deployment

# ## Business Understanding

# Credit Risk adalah kemungkinan bahwa peminjam tidak akan membayar kembali pinjaman mereka kepada pemberi pinjaman. Maka dilakukan manajemen Credit Risk dan membangun model yang dapat memprediksi credit risk data pinjaman yang diterima dan yang ditolak, sehingga menyediakan solusi teknologi bagi lending company tersebut.
# 
# * Bagaimana membangun model yang dapat memprediksi credit risk data pinjaman yang diterima dan yang ditolak yang memiliki credit  risk tinggi atau rendah.
# * Bagaimana ukuran risiko kredit peminjam.
# * Rumus yang menggunakan elemen data atau variabel untuk menentukan skor kredit peminjam.

# ## Analytic Approach

# * Predictive Model; Supervised Learning

# ## Data Requirements

# * Lending Company

# ## Data Understanding

# In[260]:


# Import the Libraries and Packages

import pandas as pd             # To work with dataset
import numpy as np              # math library
import matplotlib.pyplot as plt # to plot some parameters in seaborn
import seaborn as sns           # Graph library that use matplot in background
import xgboost as xgb
import sklearn
import warnings # ignores any warning
warnings.filterwarnings("ignore")

# import Scikit-learn library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import iqr



# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# XGBoost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score


# In[261]:


# Load the data 

df = pd.read_csv('loan_data_2007_2014.csv') 


# In[91]:


# Get an overview of the data
df.head()


# In[92]:


# Viewing the last n lines
df.tail()


# In[93]:


# Check the shape of the data
df.shape


# In[94]:


# Check the data types
df.info()


# In[95]:


# Looking the Describe of Data
df.describe().T


# In[96]:


# missing values (NaNs)
df.isnull().sum()


# In[97]:


# percentage of missing values
df.isnull().sum()*100/len(df)


# ## Data Preparation

# ### Clean the data

# In[262]:


# Remove unnecessary columns
df = df.drop(['Unnamed: 0','policy_code','id'],axis=1)
df.head()


# In[263]:


# Drop Columns with NaN Values
df = df.dropna(how='all',axis=1)
df


# In[264]:


# Fill null values (Nan and None)
df.fillna(0,inplace=True)
df


# In[265]:


# remove missing values
df.dropna(inplace=True)
df


# In[102]:


# Check the nans are replaced 
df.isnull().sum()*100/len(df)


# ### Explore the data

# In[103]:


# summary statistics
df.describe().T


# In[104]:


# Data types and non-null counts
df.info()


# In[266]:


# Looking distribution of a target label "loan_status"

loan_status_count = df[['member_id','loan_status']].groupby(['loan_status']).size().reset_index(name = 'count')
total_loans = loan_status_count['count'].sum()
loan_status_count['percentage(%)'] = ((loan_status_count['count'] / total_loans)*100).astype('float')

print(loan_status_count)


# In[267]:


# visualization of a target label "loan_status"
plt.figure(figsize=(15, 8))
sns.countplot(data=df, y='loan_status')
plt.xlabel('count')
plt.ylabel('loan_status')
plt.title("Target variable distribution")
plt.show()


# In[268]:


# Changing category column names of "loan_status"

df['loan_status'] = df['loan_status'].replace(
    {'Fully Paid':'excellent',
    'Current':'good',
    'Charged Off': 'bad',
    'Default':'bad',
    'In Grace Period':'poor',
    'Late (16-30 days)':'poor',
    'Late (31-120 days)':'poor',
    'Does not meet the credit policy. Status:Charged Off':'bad',
    'Does not meet the credit policy. Status:Fully Paid':'bad'})

plt.figure(figsize=(10, 8))
sns.countplot(x='loan_status',data=df, palette='Greens_r',order=['excellent','good','poor','bad'])
plt.title('Target variable distribution')
plt.show()


# In[235]:


# loan_status vs grade
plt.figure(figsize=(15, 8))
sns.countplot(data=df, x='loan_status',order=['excellent','good','poor','bad'],palette='Greens_r', 
            hue='grade',hue_order = df['grade'].value_counts().index.values)
plt.title('loan_status vs grade')


# In[109]:


# loan_status vs verified_status_joint
plt.figure(figsize=(15, 8))
sns.countplot(data=df, x='loan_status',order=['excellent','good','poor','bad'],palette='Greens_r', 
            hue='verification_status', 
            hue_order = df['verification_status'].value_counts().index.values)
plt.title('loan_status vs verified_status_joint')


# In[110]:


# loan_status vs purpose
plt.figure(figsize=(15, 8))
sns.countplot(data=df, x='loan_status',order=['excellent','good','poor','bad'],palette='Greens_r', 
            hue='purpose', 
            hue_order = df['purpose'].value_counts().index.values)
plt.title("loan_status vs purpose")


# In[111]:


# loan_status vs home_ownership
plt.figure(figsize=(15, 8))
sns.countplot(data=df, x='loan_status',order=['excellent','good','poor','bad'],palette='Greens_r', 
            hue='home_ownership', 
            hue_order = df['home_ownership'].value_counts().index.values)
plt.title("loan_status vs home_ownership")


# In[112]:


# loan_status vs emp_length
plt.figure(figsize=(15, 8))
sns.countplot(data=df, x='loan_status',order=['excellent','good','poor','bad'],palette='Greens_r', 
            hue='emp_length', 
            hue_order = df['emp_length'].value_counts().index.values)
plt.title("loan_status vs emp_length")


# In[113]:


# purpose
plt.figure(figsize=(15, 10))
sns.countplot(data=df, y='purpose')
plt.xlabel('count')
plt.ylabel('purpose')
plt.title("purpose")


# ### Correlation of data

# In[236]:


# Correlation Analysis
corr_matrix = df.corr()
corr_matrix


# In[115]:


# create a map showing the correlation
fig = plt.figure(figsize=(30,30))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# ## Model Building

# ####  feature as the target feature.
# ('issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d')

# #### Encoding

# In[269]:


# List of Column Object target feature
df.select_dtypes(include='object').columns.tolist()


# In[271]:


df.drop(['emp_title',
 'url',
 'desc',
 'title',
 'zip_code',
 'addr_state',
 'application_type'], inplace=True, axis=1)


# In[272]:


df.select_dtypes(include='object').columns.tolist()


# In[273]:


# categories data
cat_1 = df[['home_ownership', 'verification_status', 'pymnt_plan', 'purpose', 'initial_list_status']]

# Create OneHotEncoder object
oho = OneHotEncoder(sparse=False)

# Fit and transform the data
df_encoded = pd.DataFrame(oho.fit_transform(cat_1))

# Get the names of the encoded features
df_encoded.columns = oho.get_feature_names_out(['home_ownership', 'verification_status', 'pymnt_plan', 'purpose', 'initial_list_status'])
concatenated_data = pd.concat([df_1, df_encoded], axis=1)
concatenated_data.sample()


# In[310]:


# Encoding variabel of column emp_length
concatenated_data['emp_length'].replace({'< 1 year':0, '1 year':1, '2 years':2,
                                           '3 years':3, '4 years':4, '5 years':5,
                                           '6 years':6, '7 years':7, '8 years':8, 
                                           '9 years':9, '10+ years':10},inplace=True)
concatenated_data


# In[311]:


# Label Encoding
# convert all non-numeric variables (ordinal) to numeric type
for column in concatenated_data.columns:
    if concatenated_data[column].dtype == np.number: continue
# perform encoding for each non-numeric variables
    concatenated_data[column] = LabelEncoder().fit_transform(concatenated_data[column])
concatenated_data.head()


# ### Feature Selection

# In[312]:


corr = concatenated_data.corrwith(concatenated_data["loan_status"])
corr.reset_index(name='corr value').sort_values('corr value', ascending=False)


# In[324]:


# let's only include the features mentioned above
fig = plt.figure(figsize = (40,30))
corr_data = concatenated_data[['loan_status', 'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 
                               'total_pymnt_inv', 'total_pymnt', 'last_pymnt_amnt', 'out_prncp', 'out_prncp_inv', 
                               'total_rec_late_fee', 'grade', 'sub_grade', 'int_rate']]
sns.heatmap(corr_data.corr(),cmap='Reds', annot = True);


#  ### Logistic Regression

# In[341]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create a logistic regression object
clf = LogisticRegression()

# train the model
log_model = LogisticRegression().fit(X_train, y_train)
print(log_model)


# In[342]:


# Define the target variable and features
X = corr_data.drop(['loan_status'], axis=1) #features
y = corr_data['loan_status'] #target


# In[343]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')


# In[344]:


# Fit the model using the training data
clf.fit(X_train, y_train)


# #### Performance of Testing Model

# In[345]:


# Predict the class labels for the test data
y_pred = clf.predict(X_test)


# In[352]:


# Print the confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

acc_clf_train=round(log_model.score(X_train,y_train)*100,2)
acc_clf_test=round(log_model.score(X_test,y_test)*100,2)
print("Training Accuracy: {} %".format(acc_clf_train))
print("Test Accuracy: {} %".format(acc_clf_test))


# #### Performance of Training Model

# In[355]:


# Predict the class labels for the test data
y_train = clf.predict(X_train)


# In[361]:


# Print the confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ### Random Forest

# In[380]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# predict data train
y_train_pred_rf = rf_model.predict(X_train)

# print classification report
print('Classification Report Training Model (Random Forest):')
print(classification_report(y_train, y_train_pred_rf))


# ### XGBoost Classifier

# In[383]:


# train the model
xgb_model = XGBClassifier().fit(X_train, y_train)
# predict data train
y_train_pred_xgb = xgb_model.predict(X_train)

# print classification report
print('Classification Report Training Model (XGBoost Classifier):')
print(classification_report(y_train, y_train_pred_xgb))


# In[384]:


# predict data test
y_test_pred_xgb = xgb_model.predict(X_test)

# print classification report
print('Classification Report Testing Model (XGBoost Classifier):')
print(classification_report(y_test, y_test_pred_xgb))


# ### DecisionTreeClassifier

# In[385]:


# train the model
dt_model = DecisionTreeClassifier().fit(X_train,y_train)
print(dt_model)


# In[386]:


# predict data train
y_train_pred_dt = dt_model.predict(X_train)

# print classification report
print('Classification Report Training Model (Decision Tree):')
print(classification_report(y_train, y_train_pred_dt))


# ### K-Nearest Neighbors

# In[389]:


from sklearn.neighbors import KNeighborsClassifier #k-nearest neighbor

# train the model
knn_model = KNeighborsClassifier().fit(X_train,y_train)
print(knn_model)


# In[390]:


# predict data train
y_train_pred_knn = knn_model.predict(X_train)

# print classification report
print('Classification Report Training Model (K-Nearest Neighbors):')
print(classification_report(y_train, y_train_pred_knn))


# In[391]:


# predict data test
y_test_pred_knn = knn_model.predict(X_test)

# print classification report
print('Classification Report Testing Model (K-Nearest Neighbors):')
print(classification_report(y_test, y_test_pred_knn))

