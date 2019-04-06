#!/usr/bin/env python

# /********************************************************************
# Filename: model_small_conv_pool.py
# Author: AHN
# Creation Date: Apr 2019
# **********************************************************************/
#
# small.py from dlgo adapted for score estimation.
# Dense layers removed.
#

from pdb import set_trace as BP
import inspect
import os,sys,re,json, shutil
import numpy as np
from numpy.random import random
import argparse
import keras.layers as kl
import keras.models as km
import keras.optimizers as kopt
import keras.preprocessing.image as kp
import tensorflow as tf
from keras import backend as K

#===================================================================================================
class Model:
    #-------------------------------------------
    def __init__(self, boardsz, rate=0):
        self.boardsz = boardsz
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        nplanes = 1
        inputs = kl.Input( shape = ( self.boardsz, self.boardsz, nplanes), name = 'position')

        x = kl.BatchNormalization(axis=-1)(inputs)
        x = kl.Conv2D( 48, (7,7), activation='relu', padding='same', name='one')(inputs)
        x = kl.BatchNormalization(axis=-1)(x)
        x = kl.Conv2D( 32, (5,5), activation='relu', padding='same', name='two')(x)
        x = kl.BatchNormalization(axis=-1)(x)
        x = kl.Conv2D( 32, (5,5), activation='relu', padding='same', name='three')(x)
        x = kl.BatchNormalization(axis=-1)(x)
        x = kl.Conv2D( 32, (5,5), activation='relu', padding='same', name='four')(x)
        # Classification block
        x = kl.BatchNormalization(axis=-1)(x)
        x_class_conv = kl.Conv2D( 1, (1,1), padding='same', name='lastconv')(x)
        x_class_pool = kl.Flatten()( x_class_conv)
        # sigmoid, not softmax because each intersection needs a label
        output = kl.Activation( 'sigmoid', name='class')(x_class_pool)

        self.model = km.Model( inputs=inputs, outputs=output)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam( self.rate)
        else:
            opt = kopt.Adam()
        self.model.compile( loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#===================================================================================================
