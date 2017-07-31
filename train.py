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
dataset = dataset[:dataset.shape[0]/2]


#dataset = dataset[:len(dataset)/2]

print('building model..')


gen_layers = [[None,latent_dim],
			[None,1,1,latent_expansion],
			[None,4,4,256],
			[None,7,7,64],
			[None,14,14,16],
			[None,28,28,1]]
gen_filters = [4,5,5,5]

dis_layers = [[None,28,28,1],
			[None,7,7,16],
			[None,4,4,32],
			[None,1]]
dis_filters = [7,5]


sess = tf.InteractiveSession()
gan = GAN(sess=sess,
		latent_dim=latent_dim,
		gen_shapes=gen_layers,
		dis_shapes=dis_layers,
		gen_filters=gen_filters,
		dis_filters=dis_filters)


gan.build_gan()


print('begin training..')

gan.train_gan(n_epochs=20,
			dataset=dataset,
			batch_size=batch_size,
			lr=learning_rate,
			keep_prob=0.7,
			stabilize=True)







