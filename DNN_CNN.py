# coding:utf-8
#使用tensorflow构建神经网络主要流程如下：
#	1、建立一个计算图 ， 这可以是任何的数学运算
#	2、初始化变量 ， 编译预先定义的变量
#	3、创建会话 ， 这是“神奇的开始的”地方 ！
#	4、在会话中运行图 ， 编译图形被传递到会话，它开始执行它。
#	5、关闭会话 ， 结束这次使用。
#注：当然实际设计代码时需要具体情况具体分析。

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import CNN_DNN_inference as inference 

flags = tf.app.flags
flags.DEFINE_integer("max_steps", 1000, "max number of iteration")
flags.DEFINE_integer("hidden_sizes", 100, "nodes number of hidden layer")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout", 0.9, "ratio of dropout")
flags.DEFINE_boolean("kernel", False, "using the CNN or not")
FLAGS = flags.FLAGS

#定义各项参数：迭代最大次数，学习率，dropout，神经网络类型选择,隐藏层神经元个数
max_steps = FLAGS.max_steps
learning_rate = FLAGS.learning_rate
dropout = FLAGS.dropout
kernel = FLAGS.kernel
hidden_sizes = FLAGS.hidden_sizes
#数据下载路径
data_dir = './temp/mnist/input_data'
#日志存放路径
log_dir = './temp/mnist/logs/mnist_with_summaries'

#训练和测试函数
def train_and_test(mnist):
	#启动图
	sess = tf.InteractiveSession()

	#定义程序输入数据: 
	#				x : mnist数据集中图片数据
	#				y_ : mnist数据集中图片对应的标签数据
	#				keep_prob : 定义dropout
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
		keep_prob = tf.placeholder(tf.float32)
	
	
	#调用前向传播网络得到输出：y_out
	y_out = inference.inference_op(x, hidden_sizes, keep_prob, kernel)

	#计算网络输出y_out和标签数据的交叉熵（cross entropy），并保存到tensorboard中
	with tf.name_scope('cross_entropy'):
		diff = tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y_)
		with tf.name_scope('total'):
			cross_entropy = tf.reduce_mean(diff)
	tf.summary.scalar('cross_entropy', cross_entropy)

	#将得到的cross entropy作为损失函数，利用Adam优化算法最小化损失函数
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	#计算准确率，并将识别正确的图片和识别错误的图片采样保存到tensorboard中
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
			wrong_prediction = tf.not_equal(tf.argmax(y_out,1), tf.argmax(y_,1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.name_scope('input_reshape'):
			image_shape_input = tf.reshape(x, [-1, 28, 28, 1])

	with tf.name_scope('correct_image'):
		correct_pred = tf.nn.embedding_lookup(image_shape_input,tf.reshape(tf.where(correct_prediction),[-1]))
		tf.summary.image('correct_pred', correct_pred, 10)
	with tf.name_scope('wrong_image'):
		wrong_pred = tf.nn.embedding_lookup(image_shape_input,tf.reshape(tf.where(wrong_prediction),[-1]))
		tf.summary.image('wrong_pred', wrong_pred, 10)

	tf.summary.scalar('accuracy', accuracy)

	#直接获取所有的数据汇总
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(log_dir + '/test')
	tf.global_variables_initializer().run()

	#定义feed_dict
	def feed_dict(train):
		if train:
			xs ,ys = mnist.train.next_batch(100)
			k = dropout
		else:
			xs , ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x:xs , y_:ys , keep_prob: k}
	saver = tf.train.Saver()
	#适用for循环进行迭代训练
	for i in range(max_steps):

		if i % 10 == 0:
			summary, acc = sess.run([merged, accuracy], feed_dict = feed_dict(False))
			test_writer.add_summary(summary,i)
			print "accuracy at step %s: %s " %(i, acc)

		else:
			if i % 100 == 99:
				#定义tensorflow运行选项。
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				#定义运行的元信息。可以记录下来运算的时间、内存占用这些信息。
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step], feed_dict = feed_dict(True))

				train_writer.add_run_metadata(run_metadata, 'step%03d'%i)
				train_writer.add_summary(summary,i)
				saver.save(sess,log_dir+"/model.ckpt", i )
				print "adding run metadata for ", i 
			else:
				summary,_ = sess.run([merged,train_step], feed_dict=feed_dict(True))
				train_writer.add_summary(summary,i)

	train_writer.close()
	test_writer.close()

#主函数
def main(argv=None):
	mnist = input_data.read_data_sets(data_dir, one_hot=True)
	train_and_test(mnist)

if __name__=='__main__':
	tf.app.run()
