
# coding: utf-8

# In[17]:


#Imputing missing data in a ML Pipeline 

'''
There are many steps to building a model, from creating training and test sets, to fitting a classifier or regressor, to tuning its parameters, to evaluating its performance on new data. 
Imputation can be seen as the first step of this machine learning process, the entirety of which can be viewed within the context of a pipeline. 
Scikit-learn provides a pipeline constructor that allows you to piece together these steps into one process and thereby simplify your workflow.
What makes pipelines so incredibly useful is the simple interface that they provide. You can use the .fit() and .predict() methods on pipelines 
 
'''

# Import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC





# In[18]:


df = pd.read_csv ('house-votes-84.data', delimiter = ',', header=None, names = ['party', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] )
for i in range (1, 17):
    df[i].replace ( to_replace = ['?'], value = df[i].mode().iloc[0], inplace = True)
    df[i] = LabelEncoder().fit_transform (df[i])
    
# Create arrays for the features and the response variable
y = df['party']
X = df.drop('party', axis=1)


# In[19]:


# Setup the Imputation transformer to impute missing data (represented as 'NaN') with the 'most_frequent' value in the column (axis=0).
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]


# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))

