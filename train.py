import numpy as np 
import tensorflow as tf 
import sys

from loader import *
from model import *



latent_dim = 30
latent_expansion = 1024
learning_rate = 2e-4

n_epochs = 20
batch_size = 100



print('loading mnist..')

dataset, _, _ = load_mnist()
dataset, _ = dataset
dataset = dataset.reshape(dataset.shape[0],28,28)
#dataset = dataset[:dataset.shape[0]/2]



print('building model..')

"""
gen_layers = [[None,latent_dim],
			[None,1,1,latent_expansion],
			[None,4,4,256],
			[None,7,7,64],
			[None,14,14,16],
			[None,28,28,1]]
gen_filters = [4,5,5,7]

dis_layers = [[None,28,28,1],
			[None,7,7,4],
			[None,4,4,8],
			[None,2,2,16],
			[None,1]]
dis_filters = [7,3,2]

"""
gen_layers = [[None,latent_dim],
				[None,latent_expansion],
				[None,2048],
				[None,1024],
				[None,900],
				[None,784]]
gen_filters = None

dis_layers = [[None,784],
				[None,1024],
				[None,512],
				[None,256],
				[None,1]]

dis_filters = None


sess = tf.InteractiveSession()
gan = GAN(sess=sess,
		latent_dim=latent_dim,
		gen_shapes=gen_layers,
		dis_shapes=dis_layers,
		gen_filters=gen_filters,
		dis_filters=dis_filters,
		samples_dir='samples2/')


gan.build_gan(conv=False)


print('begin training..')

gan.train_gan(n_epochs=30,
			dataset=dataset,
			batch_size=batch_size,
			lr=learning_rate,
			keep_prob=0.7,
			stabilize=True)









