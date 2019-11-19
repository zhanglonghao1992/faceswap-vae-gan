import os
import sys
import re

import numpy as np
import h5py

import requests
from PIL import Image


outfile = 'face.hdf5'
face_txt = 'face.txt'

def main():
    for p, d, f in os.walk('aligned_faces/'):
        if len(d) > 0:
            label_list = sorted(d)

    with open(face_txt, 'r') as lines:
        lines = [l.strip() for l in lines]
        num_images = len(lines)
        label_names = np.array(label_list, dtype=object)
        num_labels = len(label_names)

        labels = np.ndarray((num_images, num_labels), dtype='uint8')
        image_data = np.ndarray((num_images, 128, 128, 3), dtype='uint8')

        for i, line in enumerate(lines):
            im_path = line.split('\t')[0]
            label = line.split('\t')[2].split(' ')
            label = list(map(int, label))
            label = np.array(label)

            im = Image.open(im_path).resize((128,128))
            im = np.asarray(im, dtype='uint8')
            image_data[i] = im
            labels[i] = label

    # Create HDF5 file
    h5 = h5py.File(outfile, 'w')
    string_dt = h5py.special_dtype(vlen=str)
    dset = h5.create_dataset('images', data=image_data, dtype='uint8')
    dset = h5.create_dataset('label_names', data=label_names, dtype=string_dt)
    dset = h5.create_dataset('labels', data=labels, dtype='uint8')

    h5.flush()
    h5.close()


if __name__ == '__main__':
    main()