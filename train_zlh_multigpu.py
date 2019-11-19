import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

from models import VAEGAN_zlh_multigpu
#from datasets import mnist, svhn
from datasets.datasets import load_data

models = {
    'zlh_multigpu': VAEGAN_zlh_multigpu
}

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='face')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--datasize', type=int, default=-1)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=4096)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # Select GPU
    str_gpu=''
    for i in range(args.gpu):
        str_gpu += (str(i)+',')
    os.environ['CUDA_VISIBLE_DEVICES'] = str_gpu

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    if args.testmode:
        datasets = load_data('./datasets/face.hdf5')
        if args.model not in models:
            raise Exception('Unknown model:', args.model)

        model = models[args.model](
            input_shape=datasets.images.shape[1:],
            num_attrs=15,
            z_dims=args.zdims,
            output=args.output
        )
        model.load_model(args.resume)
        datasets.images = datasets.images * 2.0 - 1.0

        # TBD

    else:
        datasets = load_data('./datasets/face.hdf5')
        # Construct model
        if args.model not in models:
            raise Exception('Unknown model:', args.model)

        model = models[args.model](
            input_shape=datasets.images.shape[1:],
            num_id=15,
            z_dims=args.zdims,
            gpus=args.gpu,
            output=args.output
        )

        if args.resume is not None:
            model.load_model(args.resume)

    # Training loop
        datasets.images = datasets.images * 2.0 - 1.0
        samples = np.random.normal(size=(10, args.zdims)).astype(np.float32)
        model.main_loop(datasets, datasets.attr_names,
            epochs=args.epoch,
            batchsize=args.batchsize,
            reporter=['loss', 'g_loss', 'd_loss', 'c_loss', 'i_loss', 'e_loss', 'g1_loss', 'e1_loss'])

if __name__ == '__main__':
    main()
