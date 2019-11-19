import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.models import load_model
from abc import ABCMeta, abstractmethod

from .utils import *

def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)

class BaseModel(metaclass=ABCMeta):
    '''
    Base class for non-conditional generative networks
    '''

    def __init__(self, **kwargs):
        '''
        Initialization
        '''
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs['name']

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')

        self.input_shape = kwargs['input_shape']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.test_mode = False
        self.trainers = {}


    def main_loop(self, datasets, epochs=100, batchsize=100, reporter=[]):
        '''
        Main learning loop
        '''
        # Create output directories if not exist
        out_dir = os.path.join(self.output, self.name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(res_out_dir):
            os.mkdir(res_out_dir)

        wgt_out_dir = os.path.join(out_dir, 'weights')
        if not os.path.isdir(wgt_out_dir):
            os.mkdir(wgt_out_dir)

        # Start training
        print('\n\n--- START TRAINING ---\n')
        num_data = len(datasets)
        for e in range(epochs):
            perm = np.random.permutation(num_data)
            start_time = time.time()
            for b in range(0, num_data, batchsize*2):
                bsize = min(batchsize*2, num_data - b)
                if bsize % 2 != 0 :
                    continue

                indx = perm[b:b+bsize]

                # Get batch and train on it
                x_batch = self.make_batch(datasets, indx)

                lr = 1.0e-5

                losses = self.train_on_batch(x_batch, lr)

                # Print current status
                ratio = 100.0 * (b + bsize) / num_data
                print(chr(27) + "[2K", end='')
                print('\rEpoch #%d | %d / %d (%6.2f %%) ' % \
                      (e + 1, b + bsize, num_data, ratio), end='')

                for k in reporter:
                    if k in losses:
                        print('| %s = %8.6f ' % (k, losses[k]), end='')

                # Compute ETA
                elapsed_time = time.time() - start_time
                eta = elapsed_time / (b + bsize) * (num_data - (b + bsize))
                print('| ETA: %s ' % time_format(eta), end='')

                sys.stdout.flush()

            print('')

            self.test(datasets, 10, e, res_out_dir)

            # Save current weights
            if e % 100 == 0:
                self.save_model(wgt_out_dir, e + 1)

    def make_batch(self, datasets, indx):
        '''
        Get batch from datasets
        '''
        return datasets[indx]

    def save_model(self, out_dir, epoch):
        folder = os.path.join(out_dir, 'epoch_%05d' % epoch)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            v.save_weights(filename)

    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            getattr(self, k).load_weights(filename)

    @abstractmethod
    def test(self, datasets, batchsize, epoch, output_path):
        '''
        Plase override "test" method in the derived model!
        '''
        pass


    @abstractmethod
    def train_on_batch(self, x_batch):
        '''
        Plase override "train_on_batch" method in the derived model!
        '''
        pass
