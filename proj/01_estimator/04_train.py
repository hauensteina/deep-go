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
import os,sys,re,json, shutil, glob
import numpy as np
from numpy.random import random
import argparse
import keras.layers as kl
import keras.models as km
import keras.optimizers as kopt
import keras.preprocessing.image as kp
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

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
GPU=1

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
BATCH_SIZE = 100

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

#====================
class Generator:
    #--------------------------------------
    def __init__( self, data_directory):
        self.datadir = data_directory + '/chunks'
        self.fnames = glob.glob( self.datadir + '/feat_*.npy')

    #--------------------------------------
    def generate( self, batch_size=100):
        while(1):
            featname = np.random.choice( self.fnames)
            #print( '\ngetting batches from %s' % featname)
            labname  = featname.replace( 'feat_', 'lab_')
            feats = np.load( featname)
            labs = np.load( labname)
            if feats.shape[0] % batch_size:
                print( 'Warning: chunk size %d not a multiple of batch size %d' % (feats.shape[0],  batch_size))
            while feats.shape[0] >= batch_size:
                feat_batch, feats = feats[:batch_size], feats[batch_size:]
                lab_batch, labs = labs[:batch_size], labs[batch_size:]
                yield feat_batch, lab_batch

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

    # checkpoint
    filepath="model-improvement-{epoch:02d}-{val_acc:.2f}.hd5"
    checkpoint = ModelCheckpoint( filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    callbacks_list = [checkpoint]

    STEPS_PER_EPOCH = 100
    model.model.fit_generator( Generator( 'train').generate( BATCH_SIZE),
                               steps_per_epoch=STEPS_PER_EPOCH, epochs=args.epochs,
                               validation_data = Generator( 'valid').generate( BATCH_SIZE),
                               validation_steps=int(STEPS_PER_EPOCH/10),
                               callbacks=callbacks_list)
    #   , validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

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
