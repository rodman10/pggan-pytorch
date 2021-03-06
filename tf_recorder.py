import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
import tensorflow as tf
import os, sys
import utils as utils


class tf_recorder:
    def __init__(self, output_dir):
        utils.mkdir('%s/tensorboard'%output_dir)
        self.writer = tf.summary.create_file_writer('%s/tensorboard'%output_dir)
                              
    def add_scalar(self, index, val, niter):
        with self.writer.as_default():
            tf.summary.scalar(name=index, data=val, step=niter)
            self.writer.flush()

    def add_scalars(self, index, group_dict, niter):
        with self.writer.as_default():
            for k, val in group_dict.items():
                tf.summary.scalar(name=index, data=val, step=niter)
                self.writer.flush()

    def add_image_grid(self, index, ngrid, x, niter):
        grid = utils.make_image_grid(x, ngrid)
        with self.writer.as_default():
            self.writer.add_image(name=index, data=grid, step=niter)
            self.writer.flush()

    def add_image_single(self, index, val, niter):
        with self.writer.as_default():
            tf.summary.image(name=index, data=val, step=niter)
            self.writer.flush()

