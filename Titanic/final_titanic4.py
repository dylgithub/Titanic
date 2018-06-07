#encoding=utf-8
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
'''
相比于准确率最高的xg_titanic的0.86这里进行了特征的合并与新特征的生成
把SibSp和Parch两列进行了合并同时根据此判断是不是IsAlone列
准确率提高到了0.88
'''
train_data=pd.read_csv('train.csv')
train_data.drop('Cabin',axis=1,inplace=True)
train_data['Embarked']=train_data['Embarked'].fillna('C')
train_data.ix[train_data.Embarked=='C','Embarked']=0
train_data.ix[train_data.Embarked=='S','Embarked']=1
train_data.ix[train_data.Embarked=='Q','Embarked']=2
train_data.ix[train_data.Sex=='male','Sex']=0
train_data.ix[train_data.Sex=='female','Sex']=1
age=train_data[['Age','Survived','Pclass','Sex','SibSp','Parch','Embarked','Fare']]
age_notnull=age.ix[age.Age.notnull()]
age_isnull=age.ix[age.Age.isnull()]
rfr=RandomForestRegressor(n_estimators=1000,n_jobs=-1)
X=age_notnull.values[:,1:]
Y=age_notnull.values[:,0]
rfr.fit(X,Y)
pre_age=rfr.predict(age_isnull.values[:,1:])
train_data.ix[train_data.Age.isnull(),'Age']=pre_age
train_data['FamilySize']=train_data['SibSp']+train_data['Parch']+1
train_data['IsAlone']=0
train_data.ix[train_data['FamilySize']==1,'IsAlone']=1
data=train_data[['Sex','Age','Pclass','IsAlone','Fare','Embarked']]
label=train_data.Survived
label=label.values
# print(label)
#划分数据集
X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.25,random_state=10)
#训练模型
model=xgb.XGBClassifier(n_estimators=700,n_jobs=-1,subsample=0.8,learning_rate=0.007,max_depth=10,min_child_weight=3)
model.fit(X_train,y_train)
#对测试集进行预测
yHat=model.predict(X_test)
# print(yHat)
acc=metrics.accuracy_score(yHat,y_test)
print("准确率为:",acc)
#读测试数据
# test_data = pd.read_csv('test.csv')
# # 数据清洗, 数据预处理
# test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
# test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1
# age = test_data[['Age', 'Sex', 'Parch', 'SibSp', 'Pclass']]
# age_notnull = age.loc[(test_data.Age.notnull())]
# age_isnull = age.loc[(test_data.Age.isnull())]
# X = age_notnull.values[:, 1:]
# Y = age_notnull.values[:, 0]
# rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
# rfr.fit(X, Y)
# predictAges = rfr.predict(age_isnull.values[:, 1:])
# test_data.loc[(test_data.Age.isnull()), 'Age'] = predictAges
# test_data['Embarked'] = test_data['Embarked'].fillna('S')
# test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
# test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
# test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2
# test_data.drop(['Cabin'], axis=1, inplace=True)
# # 特征选择
# X_test2 = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare','Embarked']]
# # 评估模型
# predictions = model.predict(X_test2)
# # 保存结果
# submission = pd.DataFrame({
#     "PassengerId": test_data["PassengerId"],
#     "Survived": predictions
# })
# submission.to_csv("xg-titanic-submission3.csv", index=False)