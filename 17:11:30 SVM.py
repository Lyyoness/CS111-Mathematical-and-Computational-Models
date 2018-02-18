def sudoGradientDescentFunction(b_init, m1_init, m2_init,
								learning_rate, training_steps):
	#initializing the gradient
	m1_grad, m2_grad, b_grad = 0

    for i in range(0, training_steps):
        m1_grad = #partial derivative with respect to m1
        m2_grad = #partial derivative with respect to m2
        b_grad = #partial derivative with respect to b
    new_m1 = m1_init - (learning_rate * m1_grad)
    new_m2 = m_2_init - (learning_rate * m2_grad)
    new_b = b_init - (learning_rate * b_grad)
    return [new_m1, new_m2, new_b]



import tensorflow as tf 
import numpy as np 
# import matplotlib.pyplot as plt 

# reading in data
filename = tf.train.string_input_producer(["cs111-svm-changed-labels.csv"])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename)
rec_def = [[1], [1], [1]]
input_1, input_2, col3 = tf.decode_csv(value, record_defaults=rec_def)

# parameters
learning_rate = 0.0005
training_points = 50
training_steps = 50000

# input features (x1 and x2)
x = tf.placeholder(tf.float32, [None,1])
x2 = tf.placeholder(tf.float32, [None,1])
# variables we are optimizing
m = tf.Variable(tf.random_uniform([1,1]))
m2 = tf.Variable(tf.random_uniform([1,1]))
b = tf.Variable(tf.random_uniform([1]))
# actual y values (class labels)
y_ = tf.placeholder(tf.float32, [None,1])
# model function
y = tf.matmul(x,m) + tf.matmul(x2,m2) + b
# cost function
cost = tf.reduce_mean(tf.log(1+tf.exp(-y_*y)))
# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initializing variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
	# Start populating the filename
  	coord = tf.train.Coordinator()
  	threads = tf.train.start_queue_runners(coord=coord)
  	sess.run(init)

  	xs = np.zeros([training_points+1,1])
  	ys = np.zeros([training_points+1,1])
  	label = np.zeros([training_points+1,1])

 	for i in range(0,training_points-1):
  		xs[i] = np.array([[sess.run(input_1)]])
		ys[i] = np.array([[sess.run(input_2)]])
		label[i] = np.array([[sess.run(col3)]])

	for i in range(training_steps):	
		feed = {x:xs, x2:ys, y_:label}
		_, cost_val = sess.run([optimizer, cost], feed_dict=feed)

		if i % 10000 == 0:
			print("After %d iterations: " %i)
			print(" the cost is ", "{:.4f}".format(cost_val))
			print("m: ", sess.run(m),"m2: ", sess.run(m2), " b: ", sess.run(b))

	coord.request_stop()
	coord.join(threads)

  	print("Optimization Finished!")
  	training_cost = sess.run(cost)
  	print ("Training cost is ", training_cost, " m: ", sess.run(W), " b: ", sess.run(b), '\n')


