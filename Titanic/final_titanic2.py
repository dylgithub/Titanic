#encoding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
'''
相比1这里用GridSearchCV进行超参数调优准确率为0.84和0.83差不多
'''
data=pd.read_csv('train.csv')
label=list(data['Survived'])
# print(data['Survived'].value_counts())
data['Embarked']=data['Embarked'].fillna('S')
data['Embarked']=data['Embarked'].map({'S':0,'C':1,'Q':2})
data['Sex']=data['Sex'].map({'male':0,'female':1})
ca_is=data.ix[data['Cabin'].isnull()].index
ca_not=data.ix[data['Cabin'].notnull()].index
data.ix[ca_is,'Cabin']=0
data.ix[ca_not,'Cabin']=1
age=data[['Pclass','Sex','SibSp','Parch','Fare','Embarked','Cabin']]
#选出Age为空和不为空的列
age_null_index=data.ix[data['Age'].isnull()].index
age_notnull_index=data.ix[data['Age'].notnull()].index
train_age_data=age.ix[age_notnull_index]
train_age_label=data['Age'][age_notnull_index]
pre_age_data=age.ix[age_null_index]
rf=RandomForestRegressor(n_estimators=1000)
rf.fit(train_age_data,train_age_label)
pre_age=rf.predict(pre_age_data)
data.ix[age_null_index]['Age']=pre_age
# Mapping Age
data.loc[ data['Age'] <= 16, 'Age']= 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[ data['Age'] > 64, 'Age'] = 4
data.loc[ data['Fare'] <= 7.91, 'Fare']=0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare']=1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']=2
data.loc[ data['Fare'] > 31, 'Fare']=3
data['Fare'] = data['Fare'].astype(int)
data.drop(['PassengerId','Survived','Name','Ticket'],axis=1,inplace=True)
X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.25,random_state=10)
#训练模型
#n_estimators=900,subsample=0.8,learning_rate=0.007,max_depth=6,min_child_weight=5
model=xgb.XGBClassifier(n_estimators=900,subsample=0.8,learning_rate=0.006,max_depth=3,min_child_weight=3)
# params=dict(min_child_weight=[2,3,4,5,6],n_estimators=[600,700,800,900,1000,1100],learning_rate=[0.003,0.004,0.005,0.006,0.007,0.008],
#             max_depth=[2,3,4,5,6,7])
# grid_search=GridSearchCV(model,param_grid=params)
# grid_search.fit(data,label)
# print(grid_search.best_params_)
model.fit(X_train,y_train)
#对测试集进行预测
yHat=model.predict(X_test)
# print(yHat)
acc=accuracy_score(yHat,y_test)
print("准确率为:",acc)