import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
cell_df=pd.read_csv("cell_samples.csv")
print(cell_df.tail())
print(cell_df.shape)#rows and columns
print(cell_df.size)#size in memory
print(cell_df.count())#total no. of values in each colmns
print(cell_df['Class'].value_counts())

benin_df=cell_df[cell_df['Class']==2][:200]
malign_df=cell_df[cell_df['Class']==4][:200]

axes=benin_df.plot(kind='scatter',x='Clump',y='UnifShape',color='blue',label='cancer damn')
malign_df.plot(kind='scatter',x='Clump',y='UnifShape',color='red',label='cancer damn',ax=axes)

plt.show()
print(cell_df.dtypes)
cell_df=cell_df[pd.to_numeric(cell_df['BareNuc'],errors='coerce').notnull()]#error correction of non int data i.e. not null
cell_df['BareNuc']=cell_df['BareNuc'].astype('int')#assure type
print(cell_df.columns)
feature_df=cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize','BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X=np.asarray(feature_df)#independent variables
y=np.asarray(cell_df['Class'])#dependent variables
print(X[:5])
print(y[:5])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
classifier=svm.SVC(kernel='linear',gamma='auto',C=2)#those points distance from hyperplane sum of distance is max such data points are support vector classifier
#kernel views data in different dimensions
#gamma kernel coefficients
#C is penalty of each point due to incorrect placement according to the hyper plane
classifier.fit(X_train,y_train )
y_predict=classifier.predict(X_test)

print(classification_report(y_test,y_predict))#actual,predict
