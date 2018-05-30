import codecs
import os
import numpy as np

class MNIST(object):
    def __init__(self, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.mnist_dir = '../data/mnist'  
        
        self.train_images =  self.read_image_file(os.path.join(self.mnist_dir, "train-images-idx3-ubyte"))
        self.train_images = np.reshape(self.train_images, (self.train_images.shape[0], -1))
        self.train_labels = self.read_label_file(os.path.join(self.mnist_dir, "train-labels-idx1-ubyte"))
        self.test_images = self.read_image_file(os.path.join(self.mnist_dir, "t10k-images-idx3-ubyte"))
        self.test_images = np.reshape(self.test_images, (self.test_images.shape[0], -1))
        self.test_labels = self.read_label_file(os.path.join(self.mnist_dir, "t10k-labels-idx1-ubyte"))
        self.shuffle_dataset()
        self.train_batch_count = self.train_images.shape[0] // self.train_batch_size
        self.test_batch_count = self.test_images.shape[0] // self.test_batch_size
        self.image_size = self.train_images.shape[1]

    def get_int(self, b):
        return int(codecs.encode(b, 'hex'), 16)

    def parse_byte(self, b):
        if isinstance(b, str):
            return ord(b)
        return b


    def read_label_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2049
            length = self.get_int(data[4:8])
            labels = [self.parse_byte(b) for b in data[8:]]
            assert len(labels) == length
        return np.array(labels)

    def read_image_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            assert self.get_int(data[:4]) == 2051
            length = self.get_int(data[4:8])
            num_rows = self.get_int(data[8:12])
            num_cols = self.get_int(data[12:16])
            images = []
            idx = 16
            for l in range(length):
                img = []
                images.append(img)
                for r in range(num_rows):
                    row = []
                    img.append(row)
                    for c in range(num_cols):
                        row.append(self.parse_byte(data[idx]) / 255)
                        idx += 1
            assert len(images) == length
        return (np.array(images, dtype="f") - 0.5) / 0.5

    def shuffle_dataset(self):
        self.shuffle = np.random.permutation(self.train_images.shape[0])

    def normalize(self, batch_images):
        return (batch_images - self.pp_mean) / 128.0

    def next_train_batch(self, idx):
        batch_images = self.train_images[self.shuffle[idx * self.train_batch_size : (idx + 1) * self.train_batch_size]]
        batch_labels = self.train_labels[self.shuffle[idx * self.train_batch_size : (idx + 1) * self.train_batch_size]]
        return batch_images, batch_labels

    def next_test_batch(self, idx):
        batch_images = self.test_images[idx * self.test_batch_size : (idx + 1) * self.test_batch_size]
        batch_labels = self.test_labels[idx * self.test_batch_size : (idx + 1) * self.test_batch_size]
        return batch_images, batch_labels



