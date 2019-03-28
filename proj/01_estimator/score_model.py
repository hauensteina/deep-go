#!/usr/bin/env python

# /********************************************************************
# Filename: score_model.py
# Author: AHN
# Creation Date: Mar 2019
# **********************************************************************/
#
# Model definition to estimate territory given a Go position
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

# # Look for modules further up
# SCRIPTPATH = os.path.dirname(os.path.realpath( __file__))
# sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

# import pylib.ahnutil as ut

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

        x = kl.Conv2D( 64, (5,5), activation='relu', padding='same', name='one_a')(inputs)
        x = kl.BatchNormalization(axis=-1)(x) # -1 for tf back end, 1 for theano
        x = kl.Conv2D( 128, (3,3), activation='relu', padding='same', name='one_b')(x)
        x = kl.BatchNormalization(axis=-1)(x)

        x = kl.Conv2D( 256, (3,3), activation='relu', padding='same', name='two_a')(x)
        x = kl.BatchNormalization(axis=-1)(x)
        # x = kl.Conv2D( 128, (3,3), activation='relu', padding='same', name='two_c')(x)
        # x = kl.BatchNormalization(axis=-1)(x)
        x = kl.MaxPooling2D()(x)

        x = kl.Conv2D( 512, (3,3), activation='relu', padding='same', name='two_a1')(x)
        x = kl.BatchNormalization(axis=-1)(x)
        # x = kl.Conv2D( 512, (3,3), activation='relu', padding='same', name='two_c1')(x)
        # x = kl.BatchNormalization(axis=-1)(x)

        x = kl.Conv2D( 1024,(3,3), activation='relu', padding='same', name='three_a1')(x)
        x = kl.BatchNormalization(axis=-1)(x)
        # x = kl.Conv2D( 512 , (3,3), activation='relu', padding='same', name='three_c1')(x)
        # x = kl.BatchNormalization(axis=-1)(x)
        x = kl.MaxPooling2D()(x)

        # Classification block
        x_class_conv = kl.Conv2D( 361, (1,1), padding='same', name='lastconv')(x)
        x_class_pool = kl.GlobalAveragePooling2D()( x_class_conv)
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
