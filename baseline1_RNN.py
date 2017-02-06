# baseline 1 using RNN LSTM to predict

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.preprocessing
from matplotlib import pyplot

# load train set and test set
trainset_file = np.loadtxt('classification/cluster_0_trainset.csv',dtype=int)
testset_file = np.loadtxt('classification/cluster_0_testset.csv',dtype=int)
trainset_x = np.empty((0,140), dtype = int)
trainset_y = np.empty((0,14), dtype = int)
testset_x = np.empty((0,140), dtype = int)
testset_y = np.empty((0,14), dtype = int)
# stander = sklearn.preprocessing.MaxAbsScaler()
stander = sklearn.preprocessing.StandardScaler()

for i in trainset_file:
    data = pd.read_csv('flow_per_shop/'+str(i)+'.csv')
    data = data['count'].values
    data = stander.fit_transform(data)
    len_data = len(data)
    trainset_y = np.vstack((trainset_y,data[len_data-14:]))
    trainset_x = np.vstack((trainset_x,data[len_data-154:len_data-14]))

print trainset_y
print trainset_x
trainset_x = trainset_x.reshape([-1,10,14])

for i in testset_file:
    data = pd.read_csv('flow_per_shop/'+str(i)+'.csv')
    data = data['count'].values
    data = stander.fit_transform(data)
    len_data = len(data)
    testset_y = np.vstack((testset_y,data[len_data-14:]))
    testset_x = np.vstack((testset_x,data[len_data-154:len_data-14]))

testset_x = testset_x.reshape([-1,10,14])
print testset_y
print testset_x

# define RNN model
learning_steps = 2000
input_vec_size = lstm_size = 14
time_step_size = 10
layer1_size = 14
layer2_size = 14
label_size = 14

# define model

# initial for weight
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape,stddev=1.0))

def model(X,W1,B1,W2,B2,W,B,lstm_size):
    # X,input shape: (batch_size,time_step_size,input_vec_size)
    XT = tf.transpose(X,[1,0,2])
    # XT shape : (time_step_size,batch_size,input_vec_size)
    XR = tf.reshape(XT,[-1,lstm_size])
    # XR shape : (time_step_size * batch_size ,input_vec_size)
    X_split = tf.split(0,time_step_size,XR)
    # sequence_num array with each array(batch_size,input_vec_size )

    # defin lstm cell
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size,forget_bias=1,state_is_tuple=True)
    # get lstm cell output
    output , _states = tf.nn.rnn(lstm,X_split,dtype=tf.float32)
    # output : time_step_size arrays with each array (batch_size , LSTM_size) so (time_step_size , batch_size , LSTM_size )

    output = tf.transpose(output,perm=[1,0,2])
    output = tf.slice(output,[0,time_step_size-1,0],[-1,1,-1])
    output = tf.reshape(output,[-1,lstm_size])
    # output : (batch_size , 1 * LSTM_size)

    # linear activation
    # get the last output

    layer1 = tf.matmul(output, W1) + B1
    layer1 = tf.nn.dropout(layer1, keep_prob=0.5)

    layer2 = tf.matmul(layer1, W2) + B2
    layer2 = tf.nn.dropout(layer2, keep_prob=0.5)

    # return ( batch_size , 1 )
    return tf.matmul(layer2, W) + B, lstm.state_size

# define X Y
X = tf.placeholder(tf.float32,[None,time_step_size,input_vec_size])
Y = tf.placeholder(tf.float32,[None,lstm_size])

# get lstm size and output RUL

W1 = init_weight([lstm_size,layer1_size])
B1 = init_weight([layer1_size])

W2 = init_weight([layer1_size,layer2_size])
B2 = init_weight([layer2_size])

W = init_weight([layer2_size,label_size])
B = init_weight([label_size])

py_x , state_size = model(X,W1,B1,W2,B2,W,B,lstm_size)

cost = tf.reduce_sum(tf.square( py_x*100 - Y*100 ))
train_op = tf.train.AdamOptimizer(learning_rate=0.03,).minimize(cost)

# run
with tf.Session() as sess:

    # model value initialization
    sess.run(tf.initialize_all_variables())

    # learning_steps
    for step in range(learning_steps):
        print ''
        print '############################################################'
        print ''
        feed = {X:trainset_x,Y:trainset_y}
        cost_value,_ ,predict_value= sess.run([cost,train_op,py_x], feed_dict=feed)
        # print predict_value
        # print trainset_y
        print 'cost : '+str(cost_value)
        feed = {X: testset_x, Y: testset_y}
        predict_y = sess.run(py_x, feed_dict=feed)
        print predict_y
        print testset_y

    feed = {X: trainset_x, Y: trainset_y}
    result = sess.run(py_x, feed_dict=feed)
    counter = 0
    result_reversefit = np.empty((0,14))
    testset_y = np.empty((0,14))

    for i in testset_file:
        data = pd.read_csv('flow_per_shop/' + str(i) + '.csv')
        data = data['count'].values
        value = stander.fit_transform(result[counter])
        result_reversefit = np.vstack((result_reversefit,stander.fit(data[-21:-14]).inverse_transform(value)))
        counter +=1
        testset_y = np.vstack((testset_y, data[len_data - 14:]))

    print testset_y
    print result_reversefit

np.savetxt('result/baseline1_label.csv',testset_y,fmt='%d')
np.savetxt('result/baseline1_predict.csv',result_reversefit,fmt='%.5e')

# scoring
sum = 0.
for i in range(testset_y.shape[0]):
    for j in range(testset_y.shape[1]):
        sum += np.absolute((result_reversefit[i,j]-testset_y[i,j])/(result_reversefit[i,j]+testset_y[i,j]+0.000000001))
nt = float((testset_y.shape[0]*testset_y.shape[1]))
score = sum/nt
print score

# visualizing
for i in range(100):
    pyplot.figure()
    pyplot.plot(testset_y[i])
    pyplot.plot(result_reversefit[i])
    pyplot.show()


