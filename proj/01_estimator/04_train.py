#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Train model to estimate B and W territory in a Go position
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

#import coremltools

import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

# Look for modules further up
SCRIPTPATH = os.path.dirname(os.path.realpath( __file__))
sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

import pylib.ahnutil as ut
import score_model as m

num_cores = 8
GPU=0

if GPU:
    pass
else:
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto( intra_op_parallelism_threads=num_cores,\
                             inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
                             device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session( config=config)
    K.set_session( session)


BOARD_SZ = 19
BATCH_SIZE=100

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Train model to estimate B and W territory in a Go position
    Synopsis:
      %s --epochs <n> --rate <learning_rate>
    Description:
      Build a NN model with Keras, train on the data in the train subfolder.
    Example:
      %s --epochs 10
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--epochs", required=True, type=int)
    parser.add_argument( "--rate", required=False, type=float, default=0)
    args = parser.parse_args()

    # Model
    model = m.Model( BOARD_SZ, args.rate)
    wfname =  'nn_score.weights'
    if os.path.exists( wfname):
        model.model.load_weights( wfname)

    # Load encoded data
    train_feat = np.load( 'train_feat.npy')
    train_lab  = np.load( 'train_lab.npy')
    valid_feat = np.load( 'valid_feat.npy')
    valid_lab  = np.load( 'valid_lab.npy')

    # Train
    model.model.fit( train_feat, train_lab,
                     batch_size=BATCH_SIZE, epochs=args.epochs,
                    validation_data = (valid_feat, valid_lab))
    # ut.dump_n_best_and_worst( 10, model.model, images, meta, 'train')
    # ut.dump_n_best_and_worst( 10, model.model, images, meta, 'valid')

    # Save weights and model
    if os.path.exists( wfname):
        shutil.move( wfname, wfname + '.bak')
    model.model.save( 'nn_score.hd5')
    model.model.save_weights( wfname)

    # # Convert for iOS CoreML
    # coreml_model = coremltools.converters.keras.convert( model.model,
    #                                                      #input_names=['image'],
    #                                                      #image_input_names='image',
    #                                                      class_labels = ['b', 'e', 'w'],
    #                                                      predicted_feature_name='bew'
    #                                                      #image_scale = 1/128.0,
    #                                                      #red_bias = -1,
    #                                                      #green_bias = -1,
    #                                                      #blue_bias = -1
    # );

    # coreml_model.author = 'ahn'
    # coreml_model.license = 'MIT'
    # coreml_model.short_description = 'Classify go stones and intersections'
    # #coreml_model.input_description['image'] = 'A 23x23 pixel Image'
    # coreml_model.output_description['output1'] = 'A one-hot vector for classes black empty white'
    # coreml_model.save("nn_bew.mlmodel")

if __name__ == '__main__':
    main()
