#encoding=utf-8
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
'''
这个准确率较高
数据计算说明，全连接相当于矩阵相乘，
前一个矩阵的每一行数据代表一个样本是输入层的一个结点
前一矩阵的每一行数据和后一矩阵的每一列数据都要进行相乘，相当于前一层的每个结点和后一层进行了全连接
'''


train_data=pd.read_csv('train.csv')
train_data.ix[train_data['Sex']=='male','Sex']=0
train_data.ix[train_data['Sex']=='female','Sex']=1
# target_num_map = {'male':0, 'female':1}
# train_data["Sex"]=train_data["Sex"].apply(lambda x: target_num_map[x])
# print(train_data["Sex"])

# train_data.ix[train_data.Embarked.isnull(),'Embarked']='C'
# train_data=train_data.replace({"Sex":{"male":0,"female":1},"Embarked":{"C":0,"S":1,"Q":2}})
# embarkedEncoder=preprocessing.LabelEncoder()
# train_data["Embarked"]=embarkedEncoder.fit_transform(train_data["Embarked"])
# print(train_data["Embarked"])

train_data['Embarked']=train_data['Embarked'].fillna('C')
train_data.ix[train_data['Embarked']=='S','Embarked']=0
train_data.ix[train_data['Embarked']=='C','Embarked']=1
train_data.ix[train_data['Embarked']=='Q','Embarked']=2
train_data.drop('Cabin',axis=1,inplace=True)
train_data=train_data.fillna(train_data.mean())
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)
data=train_data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
label=train_data[['Deceased','Survived']]
X_train,X_val,Y_train,Y_val=train_test_split(np.mat(data),np.mat(label),test_size=0.1,random_state=42)
x=tf.placeholder(tf.float32,[None,6],name='input')
y=tf.placeholder(tf.float32,[None,2],name='label')
weights1=tf.Variable(tf.random_normal([6,6]),name='weights1')
bias1=tf.Variable(tf.zeros([6]),name='bias1')
a=tf.nn.relu(tf.matmul(x,weights1)+bias1)
weights2=tf.Variable(tf.random_normal([6,2]),name='weights2')
bias2=tf.Variable(tf.zeros([2]),name='bias2')
z=tf.matmul(a,weights2)+bias2
y_pred=tf.nn.softmax(z)
#为什么这里传入的label未进行独热编码转换为向量形式，因为在前面的label处已经处理了
#在类别标签存活的基础上又加了个死亡，1代表存活0是死亡这就相当于自己以进行了独热编码
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=z))
correct_pred=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
acc_op=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init_learning_rate=0.02
global_step=tf.Variable(0)
learning_rate=tf.train.exponential_decay(init_learning_rate,global_step=global_step,decay_steps=100,decay_rate=0.8)
train_op=tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(300):
        total_loss=0.
        for i in range(len(X_train)):
            feed_dict={x:X_train[i],y:Y_train[i]}
            _,loss=sess.run([train_op,cost],feed_dict=feed_dict)
            total_loss+=loss
        print('Epoch: %4d, total loss = %.12f' % (epoch, total_loss))
        if epoch%10==0:
            accuracy=sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
            print("Accuracy on validation set: %.9f" % accuracy)
    print('training complete!')
    accuracy = sess.run(acc_op, feed_dict={x: X_val, y: Y_val})
    print("Accuracy on validation set: %.9f" % accuracy)
    pred = sess.run(y_pred, feed_dict={x: X_val})
    correct = np.equal(np.argmax(pred, 1), np.argmax(Y_val, 1))
    numpy_accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set (numpy): %.9f" % numpy_accuracy)

    # 读测试数据
    # test_data = pd.read_csv('test.csv')
    #
    # # 数据清洗, 数据预处理
    # test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
    # test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1
    #
    # age = test_data[['Age', 'Sex', 'Parch', 'SibSp', 'Pclass']]
    # age_notnull = age.loc[(test_data.Age.notnull())]
    # age_isnull = age.loc[(test_data.Age.isnull())]
    # X = age_notnull.values[:, 1:]
    # Y = age_notnull.values[:, 0]
    # rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    # rfr.fit(X, Y)
    # predictAges = rfr.predict(age_isnull.values[:, 1:])
    # test_data.loc[(test_data.Age.isnull()), 'Age'] = predictAges
    #
    # test_data['Embarked'] = test_data['Embarked'].fillna('S')
    # test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
    # test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
    # test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2
    #
    # test_data.drop(['Cabin'], axis=1, inplace=True)
    #
    # # 特征选择
    # X_test = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    #
    # # 评估模型
    # predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)
    #
    # # 保存结果
    # submission = pd.DataFrame({
    #     "PassengerId": test_data["PassengerId"],
    #     "Survived": predictions
    # })
    # submission.to_csv("titanic-submission.csv", index=False)