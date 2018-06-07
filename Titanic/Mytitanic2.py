#encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
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
train_data['Deceased']=train_data['Survived'].apply(lambda s:1-s)
data=train_data[['Sex','Age','Pclass','SibSp','Parch','Fare','Embarked']]
label=train_data[['Deceased','Survived']]
#划分数据集
X_train,X_val,Y_train,Y_val=train_test_split(np.mat(data),np.mat(label),test_size=0.15,random_state=30)
#定义网络参数
n_input=7
n_output=2
n_hidden1=70
total_train=50
REGULARIZATION_RATE=0.001
batch_size=10
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
#定义学习参数
# learning_rate=0.001
MOVING_AVERAGE_DECAY=0.99
global_step = tf.Variable(0, trainable=False)
x=tf.placeholder(tf.float32,[None,n_input],name='input')
y=tf.placeholder(tf.float32,[None,n_output],name='label')
#滑动平均值，改变训练轮数的变量
variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())
weight1=tf.Variable(tf.truncated_normal([n_input,n_hidden1]),name='weight1')
bias1=tf.Variable(tf.zeros(n_hidden1))
a=tf.nn.relu(tf.matmul(x,variable_averages.average(weight1))+variable_averages.average(bias1))
weight_out=tf.Variable(tf.truncated_normal([n_hidden1,n_output]),name='weight_out')
bias_out=tf.Variable(tf.zeros(n_output))
pred=tf.matmul(a,variable_averages.average(weight_out))+variable_averages.average(bias_out)
soft_pred=tf.nn.softmax(pred)

#计算损失值
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pred))
#设置指数衰减的学习率
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    int(np.shape(X_train)[0] / batch_size) + 1,
    LEARNING_RATE_DECAY)
# #L2正则化
# regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)#损失函数
# regularization = regularizer(weight1)
# cost = cost + regularization
#选择反向传播算法优化学习参数
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
train_op2 = tf.group(train_op, variable_averages_op)
correct_pred=tf.equal(tf.argmax(y,1),tf.argmax(soft_pred,1))
acc_op=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #进入循环开始训练
    for epoch in range(total_train):
        total_loss=0
        num_data=np.shape(X_train)[0]
        num=int(num_data/batch_size)+1
        for i in range(num):
            if i==(num-1):
                _, loss = sess.run([train_op2, cost], feed_dict={x: X_train[i * batch_size:],y: Y_train[i * batch_size:]})
                total_loss += loss
            else:
                _,loss=sess.run([train_op2,cost],feed_dict={x:X_train[i*batch_size:(i+1)*batch_size],y:Y_train[i*batch_size:(i+1)*batch_size]})
                total_loss+=loss
        print('Epoch: %4d, total loss = %.12f' % (epoch, total_loss))
        # if epoch%10==0:
        #     accuracy=sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
        #     print("Accuracy on validation set: %.9f" % accuracy)
    print('training complete!')
    accuracy = sess.run(acc_op, feed_dict={x: X_val, y: Y_val})
    print("Accuracy on validation set: %.9f" % accuracy)