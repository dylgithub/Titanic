#encoding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier
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
data=train_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Embarked']]
label=train_data.Survived
label=label.values
clf1 = AdaBoostClassifier(DecisionTreeClassifier())
clf2 = RandomForestClassifier(random_state=1)
clf3 = GradientBoostingClassifier()
clf4=xgb.XGBClassifier(max_depth=5,n_estimators=160,n_jobs=-1,learning_rate=0.1)
lr = lgb.LGBMClassifier(n_estimators=160,max_depth=5,n_jobs=-1)
clf5 = lgb.LGBMClassifier(n_estimators=160,max_depth=5,n_jobs=-1)
#classifiers参数接收的是一个列表，里面每一项是一个基学习器
#meta_classifier接收的参数是第二层的最终模型
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=clf4)
print('5-fold cross validation:\n')
for clf, label2 in zip([clf1, clf2, clf3,sclf],
                      ['AdaBoost',
                       'Random Forest',
                       'GBDT',
                       'StackingClassifier']):
    scores = model_selection.cross_val_score(clf, data, label,
                                              cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label2))
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