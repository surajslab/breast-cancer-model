# by suraj
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# creating the dataset
cancer = load_breast_cancer()
df_cancer=pd.DataFrame(cancer.data,columns=cancer.feature_names)
df_cancer=df_cancer.assign(target= pd.Series(cancer.target))

#training and modelling
predict='target'
x=df_cancer.drop([predict],axis=1)
y=df_cancer.target
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=42)
svm_model=SVC()
svm_model.fit(X_train,y_train)
y_predict=svm_model.predict(X_test)

#scoring the model
scr=accuracy_score(y_predict,y_test)
print("done")
print(scr)

#plotting the performance of the model
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)
plt.show()







