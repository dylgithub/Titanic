#encoding=utf-8
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
train_data=pd.read_csv('train.csv')
train_data.ix[train_data['Sex']=='male','Sex']=0
train_data.ix[train_data['Sex']=='female','Sex']=1
train_data['Embarked']=train_data['Embarked'].fillna('C')
train_data.ix[train_data['Embarked']=='S','Embarked']=0
train_data.ix[train_data['Embarked']=='C','Embarked']=1
train_data.ix[train_data['Embarked']=='Q','Embarked']=2
train_data.drop('Cabin',axis=1,inplace=True)
# train_data=train_data.fillna(train_data.mean())
age = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
age_notnull = age.loc[(train_data.Age.notnull())]
age_isnull = age.loc[(train_data.Age.isnull())]
X = age_notnull.values[:,1:]
Y = age_notnull.values[:,0]
#注意n_jobs这个参数，在bagging算法中尤为重要，因为bagging中决策树的训练互不影响所以可以并行进行训练
#n_jobs=1表示不并行，=n:n个并行，=-1表示CPU有多少core，就启动多少job
rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
predictAges = rfr.predict(age_isnull.values[:,1:])
train_data.loc[(train_data.Age.isnull()),'Age'] = predictAges
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)
data=train_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Embarked']]
label=train_data[['Deceased','Survived']]
X_train,X_val,Y_train,Y_val=train_test_split(np.mat(data),np.mat(label),test_size=0.1,random_state=42)
n_input=7
n_output=2
n_layer1=6
n_layer2=6
n_out=2
learning_rate=0.001
x=tf.placeholder(tf.float32,[None,n_input],name='input')
y=tf.placeholder(tf.float32,[None,n_output],name='label')



#三层神经网络
# weights1=tf.Variable(tf.random_normal([n_input,n_layer1]),name='weights1')
# bias1=tf.Variable(tf.zeros([n_layer1]),name='bias1')
# a1=tf.nn.relu(tf.matmul(x,weights1)+bias1)
# weights2=tf.Variable(tf.random_normal([n_layer1,n_layer2]),name='weights2')
# bias2=tf.Variable(tf.zeros([n_layer2]),name='bias2')
# a2=tf.nn.relu(tf.matmul(a1,weights2)+bias2)
# weights_out=tf.Variable(tf.random_normal([n_layer2,n_output]),name='weights_out')
# bias_out=tf.Variable(tf.zeros([n_output]),name='bias_out')
# z=tf.matmul(a2,weights_out)+bias_out


#两层神经网络
weights1=tf.Variable(tf.random_normal([n_input,n_layer1]),name='weights1')
bias1=tf.Variable(tf.zeros([n_layer1]),name='bias1')
a=tf.nn.relu(tf.matmul(x,weights1)+bias1)
weights_out=tf.Variable(tf.random_normal([n_layer1,n_output]),name='weights_out')
bias_out=tf.Variable(tf.zeros([n_output]),name='bias_out')
z=tf.matmul(a,weights_out)+bias_out


y_pred=tf.nn.softmax(z)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=z))
correct_pred=tf.equal(tf.arg_max(y,1),tf.arg_max(y_pred,1))
acc_op=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
train_op=tf.train.AdamOptimizer(learning_rate).minimize(cost)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(50):
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
    # X_test = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare','Embarked']]
    #
    # # 评估模型
    # predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)
    #
    # # 保存结果
    # submission = pd.DataFrame({
    #     "PassengerId": test_data["PassengerId"],
    #     "Survived": predictions
    # })
    # submission.to_csv("titanic-submission2.csv", index=False)