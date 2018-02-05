import cPickle
import gzip
import numpy as np
from PIL import Image 
import glob
filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
#train = np.zeros((50000,785),dtype='int')
train = training_data[0]
t = training_data[1]
valid = validation_data[0]
v = validation_data[1]
test = test_data[0]
tst = test_data[1]
print "train"
print train
# ----------- Training Data -------------
#W0 matrix
w0 = np.random.rand(784,10)
print w0
#Unit matrix bk
bk = np.ones((50000,10),dtype='int')
print bk
#Finding A
A = np.dot(train,w0)
A = A + bk
print "MATRIX A"
print A
#Finding Y Matrix
yk = np.zeros((50000,10))
t = np.zeros((50000,10))
t = training_data[1]
print "TT"
new = np.zeros((50000,10))
for k in range(0,50000):
	val = t[k]
	new[k][val]=1
print "new"
print new
print np.shape(new)
#delta W 
dwe = np.zeros((50000,784))
traintranspose = np.transpose(train)
diff = np.subtract(yk,new)
dwe= np.dot(traintranspose,diff)
print "DWE"
print dwe
#checking the change in error
count = 0
eta = 0.4
w1=np.random.rand(784,10)
while(count!=500):
	A = np.dot(train,w1)
	A = (A + bk)
	for k in range(50000):
	  sum = 0
	  for j in range(10):
		sum = sum+(np.exp(A[k][j]))
	  yk[k]=np.exp(A[k])/sum
	err = np.dot(np.transpose(train),(yk-new))
	w1 = w1 - err*eta/50000
	count = count + 1
print w1
position = 0
ycl = np.zeros(50000)
for i in range(50000):
	max = 0
	for j in range(10):
		if(yk[i][j]>max):
			max = yk[i][j]
			position = j
	ycl[i] = position
print ycl
cn = 0
for i in range(50000):
	if(ycl[i]==t[i]):
		cn = cn+1
print "for training data"
print (cn*100)/50000
#-------- Validation Data -----------------
#W0 matrix
w0 = np.random.rand(784,10)
print w0
#Unit matrix bk
bk = np.ones((50000,10),dtype='int')
print bk
#Finding A
A = np.dot(valid,w0)
A = A + bk
print "MATRIX A"
print A
#Finding Y Matrix
yk = np.zeros((50000,10))
t = np.zeros((50000,10))
t = validation_data[1]
print "TT"
new = np.zeros((50000,10))
for k in range(0,50000):
	val = t[k]
	new[k][val]=1
print "new"
print new
print np.shape(new)
#delta W 
dwe = np.zeros((50000,784))
traintranspose = np.transpose(valid)
diff = np.subtract(yk,new)
dwe= np.dot(traintranspose,diff)
print "DWE"
print dwe
#checking the change in error
count = 0
eta = 0.4
w1=np.random.rand(784,10)
while(count!=500):
	A = np.dot(valid,w1)
	A = (A + bk)
	for k in range(50000):
	  sum = 0
	  for j in range(10):
		sum = sum+(np.exp(A[k][j]))
	  yk[k]=np.exp(A[k])/sum
	err = np.dot(np.transpose(valid),(yk-new))
	w1 = w1 - err*eta/50000
	count = count + 1
print w1
position = 0
ycl = np.zeros(50000)
for i in range(50000):
	max = 0
	for j in range(10):
		if(yk[i][j]>max):
			max = yk[i][j]
			position = j
	ycl[i] = position
print ycl
cn = 0
for i in range(50000):
	if(ycl[i]==t[i]):
		cn = cn+1
print "for validation data"
print (cn*100)/50000
# ---------Test Data ----------------
#W0 matrix
w0 = np.random.rand(784,10)
print w0
#Unit matrix bk
bk = np.ones((50000,10),dtype='int')
print bk
#Finding A
A = np.dot(test,w0)
A = A + bk
print "MATRIX A"
print A
#Finding Y Matrix
yk = np.zeros((50000,10))
t = np.zeros((50000,10))
t = validation_data[1]
print "TT"
new = np.zeros((50000,10))
for k in range(0,50000):
	val = t[k]
	new[k][val]=1
print "new"
print new
print np.shape(new)
#delta W 
dwe = np.zeros((50000,784))
traintranspose = np.transpose(test)
diff = np.subtract(yk,new)
dwe= np.dot(traintranspose,diff)
print "DWE"
print dwe
#checking the change in error
count = 0
eta = 0.4
w1=np.random.rand(784,10)
while(count!=500):
	A = np.dot(valid,w1)
	A = (A + bk)
	for k in range(50000):
	  sum = 0
	  for j in range(10):
		sum = sum+(np.exp(A[k][j]))
	  yk[k]=np.exp(A[k])/sum
	err = np.dot(np.transpose(test),(yk-new))
	w1 = w1 - err*eta/50000
	count = count + 1
print w1
position = 0
ycl = np.zeros(50000)
for i in range(50000):
	max = 0
	for j in range(10):
		if(yk[i][j]>max):
			max = yk[i][j]
			position = j
	ycl[i] = position
print ycl
cn = 0
for i in range(50000):
	if(ycl[i]==t[i]):
		cn = cn+1
print "for test data"
print (cn*100)/50000
# -------- End of Task 1 -------------

# -------- Task 2 begins -------------
Wji = np.random.rand(784,100)*0.1
Wkj = np.random.rand(100,10)*0.1
Bj = np.ones((1,100))
Bk = np.ones((1,10))
X = np.zeros((1,100))
Zj = np.zeros((1,100))


new = np.zeros((50000,10))
for k in range(0,50000):
	val = t[k]
	new[k][val]=1
Tk = new
for i in range(50000):
	sum = 0
	X= np.dot(train[i],Wji)
	Zj= (1/(1+np.exp(-X)))
	Ak = np.dot(Zj,Wkj)
	
	for v in range(10):
		sum += np.exp(Ak[v])
		#print "mn"
	Yk = np.exp(Ak)/sum
	Zjt = np.transpose(Zj)
	diff = np.subtract(Yk,Tk[i])
	dc2 = np.outer(Zjt, diff)
	tr1 = np.transpose(1-Zj)
	scalar = np.dot(Zj, np.transpose(1-Zj))
	#print "scalar"
	#print scalar
	#print "WKJ SHAPE", Wkj.shape
	Wkjt = np.transpose(Wkj)
	dc = scalar* (np.dot(diff,np.transpose(Wkj)))

	eta = 0.01
	Wji = Wji - (eta * np.outer(np.transpose(train[i]),dc))
	Wkj = Wkj - (eta * dc2)

X = np.dot(train,Wji)
Zj = (1/(1+np.exp(-X)))
Ak = np.dot(Zj, Wkj)
yk = np.zeros((50000,10))
for k in range(50000):
	  sum = 0
	  for j in range(10):
		sum = sum+(np.exp(Ak[k][j]))
	  yk[k]=np.exp(Ak[k])/sum
Yk = yk
T = t
count = 0
print "Neural networks training accuracy"
ycl = np.zeros(50000)

for i in range(50000):
	max = 0
	for j in range(10):
		if(yk[i][j]>max):
			max = yk[i][j]
			position = j
	ycl[i] = position

for i in range(0,50000):
	if(ycl[i]==T[i]):
		count = count +1


print (count*100)/50000

# ------End of task 2 -------
# -------Task 3 begins ----------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)	
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(2000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
			x:batch[0], y_: batch[1], keep_prob: 1.0})
		print "step %d, training accuracy %g"%(i, train_accuracy)
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
# -------End of task 3 ---------
#----------USPS Data -----------
uspsx = [] 
uspst = []
for j in range(0,10):
	for i in glob.glob("USPSdata/Numerals/"+str(i)+"/*.png"):
		uspst.append(i)
		pic = Image.open(i)
		pic = pic.resize((28,28))
		uspsx.append(list(pic.getData()))

length = len(uspsx)
uspsxmod = np.zeros(length, 784)
uspstmod = np.zeros(length, 1)

for i in (0,20000):
	for j in (0,784):
		uspsxmod[i][j] = 1-(uspsx[i][j]/255)
	for j in (0,1):
		uspstmod[i][j] = 1-(uspst[i][j]/255)
bk = np.ones((19999,10),dtype='int')
print bk
#Finding A
A = np.dot(uspsxmod,w1)
A = A + bk
print "MATRIX A"
print A
#Finding Y Matrix
yk = np.zeros((19999,10))
t = np.zeros((19999,10))
t = Yk
new = np.zeros((19999,10))
for k in range(0,19999):
	val = t[k]
	new[k][val[0]]=1
print "new"
print new
print np.shape(new)
#delta W 
dwe = np.zeros((19999,784))
uspsxmodtranspose = np.transpose(uspsxmod)
diff = np.subtract(yk,new)
dwe= np.dot(uspsxmodtranspose,diff)
print "DWE"
print dwe
#checking the change in error
count = 0
eta = 0.4
w1=np.random.rand(784,10)
while(count!=500):
	A = np.dot(uspsxmod,w1)
	A = (A + bk)
	for k in range(19999):
	  sum = 0
	  for j in range(10):
		sum = sum+(np.exp(A[k][j]))
	  yk[k]=np.exp(A[k])/sum
	err = np.dot(np.transpose(uspsxmod),(yk-new))
	w1 = w1 - err*eta/19999
	count = count + 1
print w1
position = 0
ycl = np.zeros(19999)
for i in range(19999):
	max = 0
	for j in range(10):
		if(yk[i][j]>max):
			max = yk[i][j]
			position = j
	ycl[i] = position
print ycl
cn = 0
for i in range(19999):
	if(ycl[i]==t[i]):
		cn = cn+1
print "USPS 1st task"
print (cn*100)/19999
#----End of task 1 for USPS data------
Bj = np.ones((1,100))
Bk = np.ones((1,10))
X = np.zeros((1,100))
Zj = np.zeros((1,100))
Yk = np.zeros((1,10))
Ak = np.zeros((1,10))
Tk = uspstmod
for i in (0,19999):
	sum = 0
	X= np.dot(uspsxmod[i],Wji)+Bj
	Zj= (1/(1+np.exp(-X)))
	Ak = np.dot(Zj,Wkj)+Bk
	print Ak
	for v in range(10):
		sum += np.exp(Ak[0][v])
	for j in range(10):
		Yk[0][j]=np.exp(Ak[0][j])/sum
	Zjt = np.transpose(Zj)
	diff = np.subtract(Yk,Tk[i])
	dc2 = np.dot(Zjt, diff)
	tr1 = np.transpose(1-Zj)
	scalar = np.dot(tr1, Zj)
	Wkjt = np.transpose(Wkj)
	dc = scalar* (np.outer(diff,Wkjt))
	print np.shape(dc)
	eta = 0.4
	Wji = Wji - (eta * np.dot(dc))
	Wkj = Wkj - (eta * dc2)	
X = np.dot(uspsxmod,Wji)+Bj
Zj = (1/(1+np.exp(-X)))
Ak = np.dot(Zj, Wkj)+Bk
for v in range(10):
	sum += np.exp(Ak[0][v])
for v in range(10):
	Yk[0][v]=np.exp(Ak[0][v])/sum
T = uspstmod
count = 0
for i in range(0,10):
	if(Yk[i]==T[i]):
		count = count +1
print "USPS second task"
print (count*100)/10