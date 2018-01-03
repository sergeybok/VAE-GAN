import tensorflow as tf 
import numpy as np



def lrelu(x, alpha=0.1):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def dense_layer(input,input_shape,output_shape,activation,weight=None,bias=None,name='dense_layer'):
	if not weight:
		if len(input_shape)>2:
			input_dim = input_shape[1]*input_shape[2]*input_shape[3]
		else:
			input_dim = input_shape[-1]
		initial = tf.truncated_normal([input_dim,output_shape[-1]],stddev=0.05)
		weight = tf.Variable(initial,name=(name+'_W'))
		bias = tf.Variable(tf.constant(0.01,shape=[output_shape[-1]]),name=(name+'_b'))
	return activation(tf.matmul(input,weight) + bias), [weight,bias]


def conv_layer(input,input_shape,stride,size,out_dim,weight=None,bias=None,name='conv_layer'):
	if weight == None:
		shape = [size,size,input_shape[3],out_dim]
		initial = tf.truncated_normal(shape, stddev=0.05)
		weight = tf.Variable(initial,name=(name+'_W'))
		bias = tf.Variable(tf.constant(0.001, shape=[out_dim]))
	return tf.nn.relu(tf.nn.conv2d(input,filter=weight,
				strides=stride,
				padding='SAME',
				name=name) + bias), [weight, bias]
	


def deconv_layer(input,stride,size,output_shape,input_shape,activation=tf.nn.elu,name='deconv_layer'):
	shape = [size,size,output_shape[3],input_shape[3]]
	initial = tf.truncated_normal(shape, stddev=0.05)
	weight = tf.Variable(initial,name=name)
	bias = tf.Variable(tf.constant(0.001, shape=[output_shape[3]]))
	return activation(tf.nn.conv2d_transpose(value=input,
					filter=weight,
					output_shape=output_shape,
					padding='SAME',
					strides=stride,
					name=name) + bias), [weight, bias]


class simpleGAN:
	def __init__(self,sess,latent_dim,samples_dir):
		self.latent_dim = latent_dim
		self.sess = sess
		self.samples_dir = samples_dir
		self.output_ht = 28
		self.output_wd = 28
		self.gen_params = []

	def init_variables(self):
		init = tf.global_variables_initializer()
		self.sess.run(init)


	def build_gen(self,input,weights=[]):
		if weights == None or  len(weights) == 0:
			weights = [None] * 4

		layer_1, layer_1_params = dense_layer(input,[None,self.latent_dim],[None,4096],
					weight=weights[0],bias=weights[1],activation=tf.nn.elu,name='gen_1')
		generated, layer_2_params = dense_layer(layer_1,[None,4096],[None,784],
					weight=weights[2],bias=weights[3],activation=tf.nn.sigmoid,name='gen_2')
		gen_params = layer_1_params + layer_2_params 

		return generated, gen_params



	def build_gan(self,optimizer=tf.train.AdamOptimizer):

		self.X = tf.placeholder(tf.float32, shape=[None,28,28,1], name='Xdata')
		self.Z = tf.placeholder(tf.float32, shape=[None,self.latent_dim], name='Zprior')
		self.LR = tf.placeholder(tf.float32)
		self.keep_prob = tf.placeholder(tf.float32)
		self.phase = tf.placeholder(tf.bool)

		#gen
		self.generated, self.gen_params = self.build_gen(input=self.Z,weights=self.gen_params)


		#dis
		d_in = tf.contrib.layers.flatten(self.X)
		dlayer1, dlayer1_params = dense_layer(d_in,[None,784],[None,2048],activation=tf.nn.elu,name='dis_1')
		self.data_prediction, dlayer2_params = dense_layer(dlayer1,[None,2048],[None,1],activation=tf.nn.sigmoid,name='dis_2')

		dlayer1_, _ = dense_layer(self.generated,[None,784],[None,2048],activation=tf.nn.elu,weight=dlayer1_params[0],bias=dlayer1_params[1])
		self.gen_prediction, _ = dense_layer(dlayer1_,[None,2048],[None,1],activation=tf.nn.sigmoid,weight=dlayer2_params[0],bias=dlayer2_params[1])
		self.d_params = dlayer1_params + dlayer2_params



		offset = 1e-7
		d_prediction = tf.clip_by_value(self.data_prediction, offset, 1 - offset)
		g_prediction = tf.clip_by_value(self.gen_prediction, offset, 1 - offset)

		self.cost_d = -(tf.log(d_prediction) + tf.log(1 - g_prediction))
		self.cost_g = -(tf.log(g_prediction))
		
		self.global_step = tf.Variable(0,trainable=False)

		self.optimizer = optimizer(self.LR)
		
		self.train_step_d = self.optimizer.minimize(self.cost_d,var_list=self.d_params)
		self.train_step_g = self.optimizer.minimize(self.cost_g,var_list=self.gen_params)
		

	def train_gan(self,n_epochs,dataset,batch_size,lr,start_epoch=0,
					keep_prob=1.,stabilize=False,stabilize_batch=20):

		n_batches = len(dataset) // batch_size
		for epoch in range(start_epoch,n_epochs):
			d_list = [0.7]*stabilize_batch
			g_list = []
			stable_d =0
			stable_g =0
			np.random.shuffle(dataset)
			for batch in range(n_batches):
				tx = dataset[batch*batch_size:(batch+1)*(batch_size)]
				tz = np.random.normal(0,1,(batch_size,self.latent_dim)).astype('float32')
				if not stabilize or ((sum(d_list[-stabilize_batch:])/stabilize_batch) > 0.2):
					_, dcost = self.sess.run([self.train_step_d, self.cost_d],
							feed_dict={self.X:tx.reshape((-1,28,28,1)),self.Z:tz,self.keep_prob:keep_prob,self.LR:lr,self.phase:True})
					prev_dcost = dcost.mean()
				else:
					dcost = self.sess.run([self.cost_d],
							feed_dict={self.X:tx.reshape((-1,28,28,1)),self.Z:tz,self.keep_prob:keep_prob,self.phase:True})
					stable_d +=1
				
				if not stabilize or ((sum(d_list[-stabilize_batch:])/stabilize_batch) < 1.5):
					_, gcost = self.sess.run([self.train_step_g,self.cost_g],
							feed_dict={self.X:tx.reshape((-1,28,28,1)),self.Z:tz,self.keep_prob:keep_prob,self.LR:lr,self.phase:True})
				else:
					_, gcost = self.sess.run([self.train_step_g,self.cost_g],
							feed_dict={self.X:tx.reshape((-1,28,28,1)),self.Z:tz,self.keep_prob:keep_prob,self.LR:(lr/2.0),self.phase:True})
					stable_g +=1
				d_list.append(np.mean(dcost))
				g_list.append(np.mean(gcost))
	
			print('epoch %i || dcost = %f | gcost = %f' % (epoch,sum(d_list)/float(len(d_list)), sum(g_list)/float(len(g_list))))
			if epoch%1 == 0:
				self.save_samples(self.generated,name=(self.samples_dir+('epoch_%i.png'%(epoch))))
			if stabilize:
				print('\t#stabilize_d %i || #stabilize_g %i' % (stable_d, stable_g))


	def build_vae(self,optimizer=tf.train.AdamOptimizer):
		self.vae_X = tf.placeholder(tf.float32, shape=[None,28,28,1], name='Xdata')
		#self.vae_Z = tf.placeholder(tf.float32, shape=[None,self.latent_dim], name='Zprior')
		self.vae_LR = tf.placeholder(tf.float32)
		
		#encoder
		enc_in = tf.contrib.layers.flatten(self.vae_X)
		vae_l1, vae_l1_params = dense_layer(enc_in,[None,784],[None,2048],activation=tf.nn.elu,name='enc_1')
		
		self.vae_mu, vae_mu_params = dense_layer(vae_l1,[None,2048],[None,self.latent_dim],activation=(lambda x:x),name='enc_mu')
		sigma_in = tf.concat([vae_l1,self.vae_mu],axis=1)
		dim_in = 2048 + self.latent_dim
		self.vae_sigma, vae_sigma_params = dense_layer(sigma_in,[None,dim_in],[None,self.latent_dim],activation=tf.nn.softplus,name='enc_sigma')
		self.encoder_params = vae_l1_params + vae_mu_params + vae_sigma_params


		self.Qz = tf.contrib.distributions.Normal(loc=self.vae_mu, scale=self.vae_sigma)

		#decoder using weights from gan generator
		decoder_in = self.Qz.sample()
		self.decoded, _ = self.build_gen(input=decoder_in,weights=self.gen_params)

		self.klloss = -(1)*tf.reduce_sum(1 + tf.log(self.vae_sigma**2) - self.vae_mu**2 - self.vae_sigma**2,0)
		self.klloss = tf.reduce_mean(self.klloss)/self.latent_dim

		offset = 1e-7
		obs = tf.clip_by_value(self.decoded, offset, 1 - offset)
		obs = tf.contrib.layers.flatten(obs)
		target = tf.contrib.layers.flatten(self.vae_X)
		self.logloss = tf.reduce_sum(-1*((target*tf.log(obs) + (1-target)*tf.log(1-obs))),0)
		self.logloss = tf.reduce_mean(self.logloss)

		self.vae_cost = tf.reduce_sum(self.logloss + self.klloss)

		self.vae_optimizer = optimizer(self.vae_LR)
		self.train_step_e = self.vae_optimizer.minimize(self.vae_cost,var_list=self.encoder_params)
		self.train_step_vae = self.optimizer.minimize(self.vae_cost,var_list=(self.encoder_params+self.gen_params))

	def train_vae(self,n_epochs,dataset,batch_size,lr,start_epoch=0,train_gen=False):
		n_batches = len(dataset) // batch_size
		for epoch in range(start_epoch,n_epochs):
			kl_list = []
			logloss_list = []
			np.random.shuffle(dataset)
			for batch in range(n_batches):
				tx = dataset[batch*batch_size:(batch+1)*(batch_size)]
				tz = np.random.normal(0,1,(batch_size,self.latent_dim)).astype('float32')
				if train_gen:
					_, cost_kl, cost_log = self.sess.run([self.train_step_vae,self.klloss,self.logloss],
							feed_dict={self.vae_X:tx.reshape((-1,28,28,1)),self.vae_LR:lr,self.LR:lr})
				else :
					_, cost_kl, cost_log = self.sess.run([self.train_step_e,self.klloss,self.logloss],
							feed_dict={self.vae_X:tx.reshape((-1,28,28,1)),self.vae_LR:lr,self.LR:lr})
				kl_list.append(np.mean(cost_kl))
				logloss_list.append(np.mean(cost_log))
	
			print('epoch %i || kl cost = %f | log cost = %f' % (epoch,sum(kl_list)/float(len(kl_list)), sum(logloss_list)/float(len(logloss_list))))
			if epoch%1 == 0:
				self.save_samples(self.generated,name=(self.samples_dir+('epoch_vae_%i.png'%(epoch))))




	def save_samples(self,output,name='gan_demo.png'):
		from scipy.misc import imsave
		dim = 1
		nx = 10
		ny = 10
		rng=3



		xvals = np.linspace(-rng,rng,nx)
		yvals = np.linspace(-rng,rng,ny)

		img = np.empty((self.output_ht*ny,self.output_wd*nx))

		for xi, xv in enumerate(xvals):
			for yi, yv in enumerate(yvals):
				if self.latent_dim == 2:
					z = np.array([[xv,yv]],dtype='float32')
				else: 
					z = np.random.normal(0,1,(1,self.latent_dim))
				x_giv_z = self.generated.eval(feed_dict={self.Z:z,self.keep_prob:0.7,self.phase:0})*255
				img[(nx-xi-1)*self.output_ht:(nx-xi)*self.output_ht,yi*self.output_wd:(yi+1)*self.output_wd] = x_giv_z[0].reshape(self.output_ht,self.output_wd)
		imsave(name,img)










