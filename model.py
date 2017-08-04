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
		bias = tf.Variable(tf.constant(0.01,shape=[output_shape[-1]]))
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


class GAN:
	def __init__(self,sess,latent_dim,gen_shapes,dis_shapes,
				gen_filters=None,dis_filters=None,samples_dir='samples/',
				out_height=28,out_width=28,out_dim=1,conv=True):
		
		self.sess = sess
		self.latent_dim = latent_dim
		self.gen_shapes=gen_shapes
		self.dis_shapes = dis_shapes
		self.gen_filters = gen_filters
		self.dis_filters = dis_filters
		self.output_ht = out_height
		self.output_wd = out_width
		self.out_dim = out_dim
		self.samples_dir = samples_dir
		self.conv = conv
	

	def build_fc_generator(self,Z,weights=[],activation=tf.nn.elu):
		batch_size = tf.shape(Z)[0]
		if len(weights) == 0:
			weights += [None]*((len(self.gen_shapes)-1)*2)
		cur_layer = tf.reshape(Z,[batch_size,self.gen_shapes[0][-1]])
		for i in range(1,len(self.gen_shapes)):
			cur_shape = self.gen_shapes[i-1]
			next_shape = self.gen_shapes[i]
			if i+1 == len(self.gen_shapes):
				activation = tf.nn.sigmoid #tf.nn.tanh
			cur_layer, cur_weight = dense_layer(cur_layer,cur_shape,next_shape,
									activation=activation,weight=weights[(i-1)*2],
									bias=weights[(i-1)*2+1],name=('gen_layer_%i'%i))
			weights[(i-1)*2] = cur_weight[0]
			weights[(i-1)*2+1] = cur_weight[1]

		return tf.reshape(cur_layer,[batch_size,28,28,1]), weights

	def build_fc_discriminator(self,X,keep,weights=[],activation=tf.nn.relu):
		batch_size = tf.shape(X)[0]
		if len(weights) == 0:
			weights += [None]*((len(self.dis_shapes)-1)*2)
		cur_layer = tf.reshape(X,[batch_size,self.dis_shapes[0][-1]])
		for i in range(1,len(self.dis_shapes)):
			cur_shape = self.dis_shapes[i-1]
			next_shape = self.dis_shapes[i]
			if i+1 == len(self.dis_shapes):
				activation = tf.nn.sigmoid
				cur_layer = tf.nn.dropout(cur_layer,keep)
			cur_layer, cur_weight = dense_layer(cur_layer,cur_shape,next_shape,
									activation=activation,weight=weights[(i-1)*2],
									bias=weights[(i-1)*2+1],name=('dis_layer_%i'%i))
			weights[(i-1)*2] = cur_weight[0]
			weights[(i-1)*2+1] = cur_weight[1]
		return cur_layer, weights



	def build_generator(self,Z,phase,weights=[]):
		batch_size = tf.shape(Z)[0]
		if len(weights) == 0:
			weights += [None]*((len(self.gen_shapes)-1)*2)
		count = 0
		cur_shape = self.gen_shapes[0]
		activation = tf.nn.elu
		next_shape = self.gen_shapes[1]
		cur_layer = Z
		i = 0

		while len(self.gen_shapes[i]) < 4:
			cur_shape = self.gen_shapes[i-1]
			next_shape = self.gen_shapes[i]
			cur_shape[0] = batch_size
			next_shape[0] = batch_size
			cur_layer, cur_weight = dense_layer(cur_layer,cur_shape,next_shape,
								activation=activation,name=('gen_layer_%i'%i))
			weights[(i-1)*2] = cur_weight[0]
			weights[(i-1)*2+1] = cur_weight[1]
			i+=1
		cur_shape = self.gen_shapes[i-1]
		next_shape = self.gen_shapes[i]
		cur_shape[0] = batch_size
		next_shape[0] = batch_size
		cur_layer = tf.reshape(cur_layer,next_shape)
		while i < len(self.gen_shapes):
			if i+1 == len(self.gen_shapes):
				activation = tf.nn.sigmoid
			cur_shape = self.gen_shapes[i-1]
			next_shape = self.gen_shapes[i]
			cur_shape[0] = batch_size
			next_shape[0] = batch_size
			stride_ht = next_shape[1] // cur_shape[1]
			rem_ht = next_shape[1] % cur_shape[1]
			stride_wd = next_shape[2] // cur_shape[2]
			rem_wd = next_shape[2] % cur_shape[2]
			
			if rem_ht > 0:
				if rem_ht %4 ==0:
					tf.pad(cur_layer,[[0,0],[rem_ht/4,rem_ht/4],[0,0],[0,0]])
				elif rem_ht %2 ==0:
					tf.pad(cur_layer,[[0,0],[rem_ht//4,rem_ht//4+1],[0,0],[0,0]])
				else:
					stride_ht+=1
			if rem_wd > 0:
				if rem_wd %4 ==0:
					tf.pad(cur_layer,[[0,0],[0,0],[rem_wd/4,rem_wd/4],[0,0]])
				elif rem_wd %2 ==0:
					tf.pad(cur_layer,[[0,0],[0,0],[rem_wd//4,rem_wd//4+1],[0,0]])
				else:
					stride_wd +=1

			cur_layer, cur_weight = deconv_layer(cur_layer,
							stride=[1,stride_ht,stride_wd,1],
							size=self.gen_filters[i-2],
							output_shape=next_shape,
							input_shape=cur_shape,
							activation=activation,
							weight=weights[(i-1)*2],
							bias=weights[(i-1)*2+1],
							name=('deconv_layer_%i'%i))
			cur_layer = tf.layers.batch_normalization(tf.reshape(cur_layer,
							[batch_size,next_shape[1],next_shape[2],next_shape[3]]),
							training=phase)
			weights[(i-1)*2] = cur_weight[0]
			weights[(i-1)*2+1] = cur_weight[1]
			i+=1

		return (cur_layer), weights
			

	def build_discriminator(self,X,keep,weights=[]):
		batch_size = tf.shape(X)[0]
		cur_shape = self.dis_shapes[0]
		cur_layer = X 
		activation = tf.nn.elu
		next_shape = cur_shape
		count = 0
		if len(weights) == 0:
			weights += [None]*((len(self.dis_shapes)-1)*2)
		for i in range(1,len(self.dis_shapes)):
			next_shape = self.dis_shapes[i]
			cur_shape = self.dis_shapes[i-1]
			if len(next_shape) < 4:
				count = i
				break
			stride_ht = cur_shape[1] // next_shape[1]
			stride_wd = cur_shape[2] // next_shape[2]
			rem_ht = cur_shape[1] % next_shape[1]
			rem_wd = cur_shape[2] % next_shape[2]
			if rem_ht > 0 :
				stride_ht +=1
			if rem_wd > 0 :
				stride_wd +=1
			cur_layer, cur_weight = conv_layer(cur_layer,
							input_shape=cur_shape,
							stride=[1,stride_ht,stride_wd,1],
							size=self.dis_filters[i-1],
							out_dim=next_shape[-1],
							weight=weights[(i-1)*2],
							bias=weights[(i-1)*2+1],
							name=('dis_layer_%i'%i))
			weights[(i-1)*2] = cur_weight[0]
			weights[(i-1)*2+1] = cur_weight[1]
		cur_layer = tf.contrib.layers.flatten(cur_layer)
		for i in range(count,len(self.dis_shapes)):
			if i+1 == len(self.dis_shapes):
				activation = tf.nn.sigmoid
				cur_layer = tf.nn.dropout(cur_layer,keep)
			next_shape = self.dis_shapes[i]
			cur_shape = self.dis_shapes[i-1]
			cur_layer,cur_weight = dense_layer(cur_layer,cur_shape,next_shape,
						activation=activation,weight=weights[(i-1)*2],
						bias=weights[(i-1)*2+1],name='dis_layer_final')
			weights[(i-1)*2] = cur_weight[0]
			weights[(i-1)*2+1] = cur_weight[1]
		return cur_layer, weights

	def build_d_discriminator(self,X,X_hat,keep,phase,weights=[]):
		cur_shape = self.dis_shapes[0]
		cur_layer = tf.concat([X,X_hat],0)
		batch_size = tf.shape(cur_layer)[0]
		activation = tf.nn.relu
		next_shape = cur_shape
		if len(weights) == 0:
			weights = [None]*((len(self.dis_shapes)-1)*2)
		for i in range(1,len(self.dis_shapes)):
			next_shape = self.dis_shapes[i]
			next_shape[0] = batch_size
			cur_shape = self.dis_shapes[i-1]
			if len(next_shape) < 4:
				break
			stride_ht = cur_shape[1] // next_shape[1]
			stride_wd = cur_shape[2] // next_shape[2]
			rem_ht = cur_shape[1] % next_shape[1]
			rem_wd = cur_shape[2] % next_shape[2]
			if rem_ht > 0 :
				stride_ht +=1
			if rem_wd > 0 :
				stride_wd +=1
			cur_layer, cur_weight = conv_layer(cur_layer,
							input_shape=cur_shape,
							stride=[1,stride_ht,stride_wd,1],
							size=self.dis_filters[i-1],
							out_dim=next_shape[-1],
							weight=weights[(i-1)*2],
							bias=weights[(i-1)*2+1],
							name=('dis_layer_%i'%i))
			cur_layer = tf.layers.batch_normalization(cur_layer,training=phase)
			weights[(i-1)*2] = cur_weight[0]
			weights[(i-1)*2+1] = cur_weight[1]
		cur_layer = tf.nn.dropout(tf.contrib.layers.flatten(cur_layer),keep)
		cur_layer,cur_weight = dense_layer(cur_layer,cur_shape,next_shape,
						activation=tf.nn.sigmoid,weight=weights[-2],
						bias=weights[-1],name='dis_layer_final')
		weights[-2] = cur_weight[0]
		weights[-1] = cur_weight[1]

		x_out, x_hat_out = tf.split(cur_layer,2,axis=0)
		return x_out,x_hat_out, weights

		
	def build_gan(self,optimizer=tf.train.AdamOptimizer):
		self.X = tf.placeholder(tf.float32, shape=[None,28,28,1], name='Xdata')
		self.Z = tf.placeholder(tf.float32, shape=[None,self.latent_dim], name='Zprior')
		self.LR = tf.placeholder(tf.float32)
		self.keep_prob = tf.placeholder(tf.float32)
		self.phase = tf.placeholder(tf.bool, name='phase')
		if self.conv:
			if len(gen_filters)+2!=len(gen_shapes) or len(dis_filters)+2!=len(dis_shapes):
				print ('incompatible filter and shape inputs')
				return
			self.generated, self.gen_params = self.build_generator(self.Z,self.phase)
			self.data_prediction, self.gen_prediction, self.d_params = self.build_discriminator(self.X,self.generated,keep=self.keep_prob,phase=self.phase)
		else:
			self.generated, self.gen_params = self.build_fc_generator(self.Z)
			self.data_prediction, self.d_params = self.build_fc_discriminator(self.X,keep=self.keep_prob)
			self.gen_prediction, self.d_params = self.build_fc_discriminator(self.generated,keep=self.keep_prob,weights=self.d_params)
			

		offset = 1e-7
		d_prediction = tf.clip_by_value(self.data_prediction, offset, 1 - offset)
		g_prediction = tf.clip_by_value(self.gen_prediction, offset, 1 - offset)

		self.cost_d = -(tf.log(d_prediction) + tf.log(1 - g_prediction))
		self.cost_g = -(tf.log(g_prediction))
		
		self.global_step = tf.Variable(0,trainable=False)

		self.optimizer = optimizer(self.LR)
		
		self.train_step_d = self.optimizer.minimize(self.cost_d,var_list=self.d_params)
		self.train_step_g = self.optimizer.minimize(self.cost_g,var_list=self.gen_params)
		
		#check model
		if self.conv and len(self.d_params)/2 != len(self.dis_shapes)-1:
			print ('dparam problem, len of dparam %i len of dis_shapes %i'
					%(len(self.d_params),len(self.dis_shapes)))
		if self.conv and  len(self.gen_params)/2 != len(self.gen_shapes)-1:
			print ('gen param problem, len of gen param %i len of gen_shapes %i'
					%(len(self.gen_params),len(self.gen_shapes)))
		

		init = tf.global_variables_initializer()
		self.sess.run(init)


	def train_gan(self,n_epochs,dataset,batch_size,lr,keep_prob=1.0,
				start_epoch=0,stabilize=False,stabilize_batch=20):
		#from timeit import default_timer as timer
		#start = timer()
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
					_, dcost = self.sess.run([self.train_step_d, self.cost_d],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.LR:lr,self.phase:True})
					prev_dcost = dcost.mean()
				else:
					dcost = self.sess.run([self.cost_d],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.phase:True})
					stable_d +=1
				
				if not stabilize or ((sum(d_list[-stabilize_batch:])/stabilize_batch) < 1.5):
					_, gcost = self.sess.run([self.train_step_g,self.cost_g],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.LR:lr,self.phase:True})
				else:
					_, gcost = self.sess.run([self.train_step_g,self.cost_g],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.LR:(lr/2.0),self.phase:True})
					stable_g +=1
				d_list.append(np.mean(dcost))
				g_list.append(np.mean(gcost))
	
			print('epoch %i || dcost = %f | gcost = %f' % (epoch,sum(d_list)/float(len(d_list)), sum(g_list)/float(len(g_list))))
			if epoch%1 == 0:
				self.save_samples(self.generated,name=(self.samples_dir+('epoch_%i.png'%(epoch))))
			if stabilize:
				print('\t#stabilize_d %i || #stabilize_g %i' % (stable_d, stable_g))
	


	def build_encoder(self,X,weights=[]):
		batch_size = tf.shape(X)[0]
		cur_shape = self.encoder_shapes[0]
		cur_layer = X 
		activation = tf.nn.relu
		next_shape = cur_shape
		if len(weights) == 0:
			weights = [None]*((len(self.encoder_shapes)-1)*2)
		for i in range(1,len(self.encoder_shapes)):
			cur_shape = self.encoder_shapes[i-1]
			next_shape = self.encoder_shapes[i]
			if i+1 == len(self.encoder_shapes):
				break
			stride_ht = cur_shape[1] // next_shape[1]
			stride_wd = cur_shape[2] // next_shape[2]
			rem_ht = cur_shape[1] % next_shape[1]
			rem_wd = cur_shape[2] % next_shape[2]
			if rem_ht > 0 :
				stride_ht +=1
			if rem_wd > 0 :
				stride_wd +=1
			cur_layer, cur_weight = conv_layer(cur_layer,
							input_shape=cur_shape,
							stride=[1,stride_ht,stride_wd,1],
							size=self.encoder_filters[i-1],
							out_dim=next_shape[-1],
							name=('encoder_layer_%i'%i))
			weights += cur_weight
		mu, mu_weights = dense_layer(cur_layer,
							input_shape=cur_shape,
							output_shape=next_shape,
							activation=lambda x:x)
		sigma, sigma_weights = dense_layer(cur_layer,
							input_shape=cur_shape,
							output_shape=next_shape,
							activation=tf.nn.softplus)
		weights += mu_weights
		weights += sigma_weights
		return mu, sigma, weights
			




	def build_vae(self,encoder_shapes,encoder_filters,optimizer=tf.train.AdamOptimizer,conv=True):
		self.encoder_shapes = encoder_shapes
		self.encoder_filters = encoder_filters
		self.encoder_X = tf.placeholder(tf.float32,shape=[None,28,28,1], name='encoder_X')

		self.mu, self.sigma, self.encoder_params = self.build_encoder(self.encoder_X)
		
		Qz = tf.contrib.distributions.Normal(mu=self.mu, sigma=self.sigma)
		z_sample = Qz.sample()

		self.decoded = self.build_generator(z_sample,self.phase,weights=self.gen_params)
		
		self.klloss = -(1)*tf.reduce_sum(1 + tf.log(z_sigma**2) - z_mu**2 - z_sigma**2,1)
		#sigmaloss = tf.reduce_sum((tf.ones_like(z_sigma)-z_sigma)**4 )

		offset = 1e-7
		obs = tf.clip_by_value(self.decoded, offset, 1 - offset)
		self.logloss = -1*(tf.reduce_sum(self.encoder_X*tf.log(obs) + (1-self.encoder_X)*tf.log(1-obs)))


		self.vae_cost = tf.reduce_mean(logloss + klloss)

		self.vae_optimizer = optimizer(self.LR)
		self.train_step_e = self.vae_optimizer.minimize(self.vae_cost,var_list=self.encoder_params)





	def save_samples(self,output,name='gan_demo.png'):
 		from scipy.misc import imsave
 		dim = 1
		nx = 15
		ny = 15
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





class simpleGAN:
	def __init__(self,sess,latent_dim,samples_dir):
		self.latent_dim = latent_dim
		self.sess =sess
		self.samples_dir = samples_dir
		self.output_ht = 28
		self.output_wd = 28

	def build_gan(self,optimizer=tf.train.AdamOptimizer):
		self.X = tf.placeholder(tf.float32, shape=[None,28,28,1], name='Xdata')
		self.Z = tf.placeholder(tf.float32, shape=[None,self.latent_dim], name='Zprior')
		self.LR = tf.placeholder(tf.float32)
		self.keep_prob = tf.placeholder(tf.float32)
		self.phase = tf.placeholder(tf.bool)

		#gen
		layer_1, layer_1_params = dense_layer(self.Z,[None,self.latent_dim],[None,1024],activation=tf.nn.elu)
		self.generated, layer_2_params = dense_layer(layer_1,[None,1024],[None,784],activation=tf.nn.sigmoid)
		self.gen_params = layer_1_params + layer_2_params


		#dis
		d_in = tf.contrib.layers.flatten(self.X)
		dlayer1, dlayer1_params = dense_layer(d_in,[None,784],[None,64],activation=tf.nn.tanh)
		self.data_prediction, dlayer2_params = dense_layer(dlayer1,[None,32],[None,1],activation=tf.nn.sigmoid)

		dlayer1_, _ = dense_layer(self.generated,[None,784],[None,64],activation=tf.nn.tanh,weight=dlayer1_params[0],bias=dlayer1_params[1])
		self.gen_prediction, _ = dense_layer(dlayer1_,[None,64],[None,1],activation=tf.nn.sigmoid,weight=dlayer2_params[0],bias=dlayer2_params[1])
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
		
		init = tf.global_variables_initializer()
		self.sess.run(init)

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
					_, dcost = self.sess.run([self.train_step_d, self.cost_d],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.LR:lr,self.phase:True})
					prev_dcost = dcost.mean()
				else:
					dcost = self.sess.run([self.cost_d],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.phase:True})
					stable_d +=1
				
				if not stabilize or ((sum(d_list[-stabilize_batch:])/stabilize_batch) < 1.5):
					_, gcost = self.sess.run([self.train_step_g,self.cost_g],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.LR:lr,self.phase:True})
				else:
					_, gcost = self.sess.run([self.train_step_g,self.cost_g],feed_dict={self.X:np.expand_dims(tx,3),self.Z:tz,self.keep_prob:keep_prob,self.LR:(lr/2.0),self.phase:True})
					stable_g +=1
				d_list.append(np.mean(dcost))
				g_list.append(np.mean(gcost))
	
			print('epoch %i || dcost = %f | gcost = %f' % (epoch,sum(d_list)/float(len(d_list)), sum(g_list)/float(len(g_list))))
			if epoch%1 == 0:
				self.save_samples(self.generated,name=(self.samples_dir+('epoch_%i.png'%(epoch))))
			if stabilize:
				print('\t#stabilize_d %i || #stabilize_g %i' % (stable_d, stable_g))





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










