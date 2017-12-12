import numpy as np
import math
import tensorflow as tf
import random
import air_vecs

if __name__ == "__main__":

    av = air_vecs.get_data()

    train_X, train_Y = av["train"]
    test_X, test_Y = av["test"]
    dev_X, dev_Y = av["dev"]

    train_X, train_Y = (np.array([0,1,2,4,5])[:, np.newaxis],
            np.array([0,2,4,8,10])[:, np.newaxis])
    dev_X, dev_Y = (np.array([0,1,2,4,5])[:, np.newaxis], np.array([0,2,4,8,10])[:, np.newaxis])
    test_X, test_Y = (np.array([0,1,2,4,6])[:, np.newaxis], np.array([0,2,4,8,12])[:, np.newaxis])

    n_samples = train_X.shape[0]

    learning_rate = 0.01
    training_epochs = 10000
    display_step = 100
    
    X = tf.placeholder(tf.float32, [None, 1])
    Y = tf.placeholder(tf.float32, [None, 1])

    W1 = tf.Variable(tf.random_normal([1, 10], stddev=1), name="weight1")
    b1 = tf.Variable(tf.random_normal([10]), name="bias1")
    
    W2 = tf.Variable(tf.random_normal([10,1], stddev=1), name="weight2")
    b2 = tf.Variable(tf.random_normal([1]), name="bias2")

    hidden1 = tf.add(tf.matmul(X, W1), b1)
    hidden1 = tf.nn.relu(hidden1)

    pred = tf.add(tf.matmul(hidden1, W2), b2)
    cost = tf.reduce_mean(tf.squared_difference(Y, pred))
    
    total_error = tf.reduce_sum(tf.square(tf.subtract(Y, tf.reduce_mean(Y))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(Y, pred)))
    R_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    dev_cost = 0

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={X:train_X, Y:train_Y})

            if (epoch+1)%display_step==0:
                c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
                print ("Epoch:","%05d"%(epoch+1), "cost=",c)
                old_dev=dev_cost
                dev_cost=sess.run(cost, feed_dict={X:dev_X, Y:dev_Y})
                print ("Dev Cost = ", dev_cost)
                print ("Change = ", dev_cost-old_dev)
                print ("")

        print ("Finished Training....")
        training_cost=sess.run(R_squared, feed_dict={X:test_X, Y:test_Y})
        print ("R Squared =", training_cost)
        training_cost=sess.run(cost, feed_dict={X:test_X, Y:test_Y})
        print ("Mean Squared Error =", training_cost)
        
        print ("")
        predicts = sess.run(pred, feed_dict={X:test_X[:10]})
        print (test_Y[:10] - predicts)
        print (predicts)
        print (test_Y[:10])
        print ("")
        print (sess.run(total_error, feed_dict={Y:train_Y})/len(train_Y))
   
    
