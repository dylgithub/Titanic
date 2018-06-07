#encoding=utf-8
import numpy as np
import tensorflow as tf
from sklearn import metrics
a=[1,2,3]
b=a[1:]
# print(np.shape(np.mat(b).T))
def sigmoid(inX):
    pre=1.0/(1+np.exp(-inX))
    pre=np.array(pre)
    returnArr=np.zeros(np.shape(pre))
    returnArr[:,np.nonzero(pre>0.5)[1]]=1
    return returnArr
c=np.mat([0.4,0.6,0.1,0.8])
d=[0.0,1.,0.0,1.]
f=sigmoid(c)
# print(f)
weights=np.ones(6).reshape(2,3)
bias=np.array([2,3,4])
weights+=bias
# print(weights)
a=np.array([[1,2,3],[7,5,6],[8,4,7]])
b=np.argsort(a[:,1])[::-1]
# print(b)
# x=np.array([0,1,1,0])
# y=np.array([0,0,0.8,0])
# print(metrics.roc_auc_score(x,y))
# ee=[1,2,
# 3]
# print(ee[::-1])
# e=tf.nn.sigmoid_cross_entropy_with_logits(labels=d,logits=c)
# with tf.Session() as sess:
#     print(sess.run(e))