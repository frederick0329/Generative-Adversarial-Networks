import sys
sys.path.append("..")
import numpy as np
import os
from gan import *
from mnist import *
from utils import *
from logger import *

gpu_number = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
    
class Trainer():
    def __init__(self, train_batch_size=100, test_batch_size=500, exp_name='exp2'):
        self.latent_size = 64
        self.hidden_size = 256

        self.dataset = MNIST(train_batch_size, test_batch_size)
        self.image_size = self.dataset.image_size        

        self.G = Generator()
        self.D = Discriminator()
        
        self.build_graph()
        self.logger = Logger(exp_dir=exp_name)
        self.tensorboard_path = '/tmp/gan/'+ exp_name

        # sample directoy
        self.sample_dir = os.path.join(exp_name, 'samples')

        # Create a directory if not exists
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # session config
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True

            # input placeholders
            self.image_placeholder = tf.placeholder(
            	tf.float32,
            	shape = [None, self.image_size],
            	name = 'real_images'
            )
            self.noise_placeholder = tf.placeholder(
                tf.float32,
                shape = [None, self.latent_size],
                name = 'noise'
            )
            
            # network          
            self.fake_images = self.G.build_network(self.noise_placeholder, self.latent_size, self.image_size, self.hidden_size)
            self.real_logits = self.D.build_network(self.image_placeholder, self.image_size, self.hidden_size)               
            self.fake_logits = self.D.build_network(self.fake_images, self.image_size, self.hidden_size)               

            self.real_scores = tf.reduce_mean(tf.sigmoid(self.real_logits))
            self.fake_scores = tf.reduce_mean(tf.sigmoid(self.fake_logits))

            # loss
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.real_logits, labels=tf.ones_like(self.real_logits)))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits)))
            self.d_loss = self.d_loss_real + self.d_loss_fake

            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.fake_logits, labels=tf.ones_like(self.fake_logits)))


            self.global_step = tf.Variable(initial_value=0, trainable=False)

            # optimizeir
            self.learning_rate = tf.Variable(initial_value=0.0002, trainable=False)

            train_vars = [x for x in tf.trainable_variables() if 'Discriminator' in x.name]
            d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.d_optim = d_optimizer.minimize(self.d_loss, global_step=self.global_step, var_list=train_vars, name='d_loss_optimizer')
            
            train_vars = [x for x in tf.trainable_variables() if 'Generator' in x.name]
            g_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.g_optim = g_optimizer.minimize(self.g_loss, var_list=train_vars, name='g_loss_optimizer')

            # train summary
            self.g_loss_placeholder = tf.placeholder(tf.float32, shape=())
            self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss_placeholder)
            
            self.d_loss_placeholder = tf.placeholder(tf.float32, shape=())
            self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss_placeholder)
            
            self.real_scores_placeholder = tf.placeholder(tf.float32, shape=())
            self.real_scores_summary = tf.summary.scalar('real_scores', self.real_scores_placeholder)
            
            self.fake_scores_placeholder = tf.placeholder(tf.float32, shape=())
            self.fake_scores_summary = tf.summary.scalar('fake_scores', self.fake_scores_placeholder)

    def train(self, epochs=1):
        with self.graph.as_default():
            writer = tf.summary.FileWriter(self.tensorboard_path, graph=tf.get_default_graph())
            with tf.Session(config=self.config) as sess:
                saver = tf.train.Saver(max_to_keep=100)
                all_initializer_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(all_initializer_op)
                for i in range(epochs):
                    total_g_loss = 0.0
                    total_d_loss = 0.0
                    total_real_scores = 0.0
                    total_fake_scores = 0.0
                    self.dataset.shuffle_dataset()
                    for j in range(self.dataset.train_batch_count):
                        batch_images, _ = self.dataset.next_train_batch(j)
                        # Generate noise to feed to the generator
                        z = np.random.normal(0., 1., size=[batch_images.shape[0], self.latent_size])
                        real_scores, fake_scores, d_loss,  _ = sess.run([self.real_scores, self.fake_scores, self.d_loss, self.d_optim], 
                                                                 feed_dict = {self.image_placeholder : batch_images, 
                                                                              self.noise_placeholder : z})
                        z = np.random.normal(0., 1., size=[batch_images.shape[0], self.latent_size])
                        g_loss, _ = sess.run([self.g_loss, self.g_optim], 
                                             feed_dict = {self.noise_placeholder : z})
                        total_d_loss += d_loss
                        total_g_loss += g_loss
                        total_real_scores += real_scores
                        total_fake_scores += fake_scores
                    avg_d_loss = total_d_loss / self.dataset.train_batch_count 
                    avg_g_loss = total_g_loss / self.dataset.train_batch_count
                    avg_real_scores = total_real_scores / self.dataset.train_batch_count
                    avg_fake_scores = total_fake_scores / self.dataset.train_batch_count
                    # logging training results
                    summary = sess.run(self.g_loss_summary, feed_dict={self.g_loss_placeholder: avg_g_loss})
                    writer.add_summary(summary, i)
                    summary = sess.run(self.d_loss_summary, feed_dict={self.d_loss_placeholder: avg_d_loss})
                    writer.add_summary(summary, i)
                    summary = sess.run(self.real_scores_summary, feed_dict={self.real_scores_placeholder: avg_real_scores})
                    writer.add_summary(summary, i) 
                    summary = sess.run(self.fake_scores_summary, feed_dict={self.fake_scores_placeholder: avg_fake_scores})
                    writer.add_summary(summary, i) 
                    self.logger.log('Training epoch {0}'.format(i))
                    self.logger.log('    g loss {0}, d loss {1}, real score {2}, fake score {3}'.format(avg_g_loss,
                                                                                                        avg_d_loss,
                                                                                                        avg_real_scores,
                                                                                                        avg_fake_scores))
                    # Sample images
                    z = np.random.normal(0., 1., size=[64, self.latent_size])
                    fake_images = sess.run(self.fake_images, feed_dict={self.noise_placeholder : z})
                    fake_images = fake_images.reshape(-1, 1, 28, 28)
                    save_image(denorm(fake_images), os.path.join(self.sample_dir, 'fake_images-{}.png'.format(i)))
                    # save model
                    if i % 20 == 0:
                        save_model_file = os.path.join(self.logger.exp_dir, 'Gan-model')
                        saver.save(sess, save_model_file, global_step=self.global_step)

if __name__ == "__main__":
    trainer = Trainer(exp_name='gan')
    trainer.train(epochs=400)
