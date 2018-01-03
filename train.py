import numpy as np 
import tensorflow as tf 
import sys

from loader import *
from model import *



latent_dim = 100

latent_expansion = 1024
learning_rate = 2e-4

n_epochs = 100
batch_size = 100



print('loading mnist..')

#dataset, _, _ = load_mnist()
#dataset, _ = dataset
#dataset = dataset.reshape(dataset.shape[0],28,28)

dataset= load_tf_mnist()



print('building model..')




sess = tf.InteractiveSession()


gan = simpleGAN(sess=sess,
				latent_dim=latent_dim,
				samples_dir='latent100/gan20vae20/')

gan.build_gan()

print('building vae')
gan.build_vae()

gan.init_variables()

print('begin training..')



gan.train_gan(n_epochs=20,
			dataset=dataset,
			batch_size=batch_size,
			lr=learning_rate,
			keep_prob=1.,
			stabilize=True)



gan.train_vae(n_epochs=20,
			dataset=dataset,
			batch_size=batch_size,
			lr=learning_rate,
			train_gen=True)


"""
gan.train_gan(n_epochs=20,
			dataset=dataset,
			batch_size=batch_size,
			lr=learning_rate,
			keep_prob=1.,
			stabilize=True,
			start_epoch=10)


gan.train_vae(n_epochs=20,
			dataset=dataset,
			batch_size=batch_size,
			lr=learning_rate,
			train_gen=True)

"""

