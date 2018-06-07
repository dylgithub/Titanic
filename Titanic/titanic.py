#encoding=utf-8
import pandas as py
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import tensorflow as tf
train=py.read_csv('train.csv')
test=py.read_csv('test.csv')
target='Survived'
labelClass=train[target]
labelClass=labelClass.values
PassengerId=test['PassengerId']
del train['PassengerId']
del test['PassengerId']
del train['Survived']
del train['Name']
del test['Name']
del train['Ticket']
del test['Ticket']
del train['Cabin']
del test['Cabin']
del train['Embarked']
del test['Embarked']
# print(train['Sex'])
# print(train['Embarked'])
sexEncoder=preprocessing.LabelEncoder()
testSexEncoder=preprocessing.LabelEncoder()
embarkedEncoder=preprocessing.LabelEncoder()
testEmbarkedEncoder=preprocessing.LabelEncoder()
train['Sex']=sexEncoder.fit_transform(train['Sex'])
test['Sex']=testSexEncoder.fit_transform(test['Sex'])
# embarkedEncoder.fit_transform(train['Embarked'])
# test['Embarked']=testEmbarkedEncoder.fit_transform(test['Embarked'])
data=train.fillna(train.mean())
testData=test.fillna(test.mean())
testData=testData.astype(np.float64)
learning_rate=0.001
training_epochs=25
batch_size=50
display_step=1
#设置网络参数
n_hidden_1=256
n_hidden_2=256
n_input=6
n_output=1
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_output])
weights={
    'w1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1]),dtype=tf.float64),
    'w2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2]),dtype=tf.float64),
    'out':tf.Variable(tf.truncated_normal([n_hidden_2,n_output]),dtype=tf.float64),
}
biases={
    'b1':tf.Variable(tf.zeros([n_hidden_1]),dtype=tf.float64),
    'b2':tf.Variable(tf.zeros([n_hidden_2]),dtype=tf.float64),
    'out':tf.Variable(tf.zeros([n_output]),dtype=tf.float64)
}
#定义计算前向传播预测值的函数
def multilayer_perceptron(x,weights,biases):
    layer1=tf.nn.relu(tf.add(tf.matmul(x,weights['w1']),biases['b1']))
    layer2=tf.nn.relu(tf.add(tf.matmul(layer1,weights['w2']),biases['b2']))
    out_layer=tf.add(tf.matmul(layer2,weights['out']),biases['out'])
    return out_layer
pred=multilayer_perceptron(x,weights,biases)
#定义Loss和优化器
cost=tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        _, c = sess.run([optimizer, cost], feed_dict={x: data, y: np.mat(labelClass).T})
        # 计算平均Loss值
        avg_cost += c / np.shape(data)[0]
        # if (epoch + 1) % display_step == 0:
        #     print('Epoch:', '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("w1:",multilayer_perceptron(testData,weights,biases))
    print("Finshed")