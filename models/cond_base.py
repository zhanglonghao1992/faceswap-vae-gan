import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .base import BaseModel

class CondBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super(CondBaseModel, self).__init__(**kwargs)
        self.id_names = None

    def main_loop(self, datasets, id_names, epochs=100, batchsize=100, reporter=[]):
        self.id_names = id_names
        super(CondBaseModel, self).main_loop(datasets, epochs, batchsize, reporter)

    def make_batch(self, datasets, indx):
        images = datasets.images[indx]
        ids = datasets.attrs[indx]
        return images, ids
