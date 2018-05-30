import sys
sys.path.append("..")
import numpy as np
import os
import torch
import torch.nn as nn
from gan import *
from mnist import *
from logger import *
from utils import *

gpu_number = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
    
class Trainer():
    def __init__(self, train_batch_size=100, test_batch_size=100, exp_name='exp'):
        self.latent_size = 64
        self.hidden_size = 256

        self.dataset = MNIST(train_batch_size, test_batch_size)
        self.image_size = self.dataset.image_size
        
        self.D = Discriminator(self.image_size, self.hidden_size).cuda()
        self.G = Generator(self.latent_size, self.image_size, self.hidden_size).cuda()
        self.logger = Logger(exp_dir=exp_name)
        # define loss function (criterion) and optimizer
        self.criterion = nn.BCELoss()
        
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002)
        
        # create labels
        self.real_labels = torch.ones(train_batch_size, 1).cuda()
        self.fake_labels = torch.zeros(train_batch_size, 1).cuda()

        # sample directoy
        self.sample_dir = os.path.join(exp_name, 'samples')

        # Create a directory if not exists
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def train(self, epochs=1):
        for i in range(epochs):
            total_g_loss = 0.0
            total_d_loss = 0.0
            total_real_scores = 0.0
            total_fake_scores = 0.0
            self.dataset.shuffle_dataset()
            self.D.train()
            self.G.train()
            for j in range(self.dataset.train_batch_count):
                real_images, _ = self.dataset.next_train_batch(j)
                real_images = torch.tensor(real_images).cuda()
                # Train discriminator   
                real_scores = self.D(real_images)
                d_loss_real = self.criterion(real_scores, self.real_labels)

                z = torch.randn(real_images.shape[0], self.latent_size).cuda()
                fake_images = self.G(z)
                fake_scores = self.D(fake_images) 
                d_loss_fake = self.criterion(fake_scores, self.fake_labels).cuda()
                
                d_loss = d_loss_real + d_loss_fake
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                total_d_loss += d_loss.item()
                total_real_scores += real_scores.mean().item()
                total_fake_scores += fake_scores.mean().item()

                # Train generator
                z = torch.randn(real_images.shape[0], self.latent_size).cuda()
                fake_images = self.G(z)
                outputs = self.D(fake_images)

                g_loss = self.criterion(outputs, self.real_labels)    
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step() 

                total_g_loss += g_loss.item()

            # logging training results
            self.logger.log('Training epoch {0}'.format(i))
            self.logger.log('    g loss {0}, d loss {1}, real score {2}, fake score {3}'.format(total_g_loss / self.dataset.train_batch_count,
                                                                                                total_d_loss / self.dataset.train_batch_count, 
                                                                                                total_real_scores / self.dataset.train_batch_count, 
                                                                                                total_fake_scores / self.dataset.train_batch_count))
            
            # Sample images
            z = torch.randn(64, self.latent_size).cuda()
            fake_images = self.G(z)
            fake_images = fake_images.reshape(-1, 1, 28, 28)
            save_image(denorm(fake_images.detach().cpu().numpy()), os.path.join(self.sample_dir, 'fake_images-{}.png'.format(i)))
            
            if i % 20 == 0:
                save_model_file = os.path.join(self.logger.exp_dir, 'Gan-model_epoch' + str(i))
                state = {'epoch': i + 1,
                         'd_state_dict': self.D.state_dict(),
                         'd_optimizer' : self.d_optimizer.state_dict(),
                         'g_state_dict': self.G.state_dict(),
                         'g_optimizer' : self.g_optimizer.state_dict()
                        } 
                torch.save(state, save_model_file)


if __name__ == "__main__":
    trainer = Trainer(exp_name='exp2')
    trainer.train(epochs=400)
