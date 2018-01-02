


def load_tf_mnist(limit=None):
	from tensorflow.examples.tutorials.mnist import input_data
	train, valid, test = input_data.read_data_sets("MNIST_data/", one_hot=True)
	return train._images

def load_mnist(filename='mnist.pkl.gz'):
	import gzip, pickle
	f = gzip.open('mnist.pkl.gz','rb')
	train_set, valid_set, test_set = pickle.load(f)
	f.close()
	return train_set, valid_set, test_set
