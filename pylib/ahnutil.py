# /********************************************************************
# Filename: ahnutil.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Various python utility funcs
#

from pdb import set_trace as BP
import os,sys,re,json
import fnmatch
import shutil
import glob
import random
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

import keras.preprocessing.image as kp
import keras.activations as ka
import keras.metrics as kmet
import keras.models as kmod
import keras.losses as klo
from keras.utils.np_utils import to_categorical
import keras

from keras import backend as K

#######################
# Keras related funcs
#######################

# Custom Softmax along axis 1 (channels).
# Use as an activation
#-----------------------------------------
def softMaxAxis1(x):
    return ka.softmax(x,axis=1)
# Make sure we can save and load a model with custom activation
ka.softMaxAxis1 = softMaxAxis1

# Custom metric returns 1.0 if all rounded elements
# in y_pred match y_true, else 0.0 .
#---------------------------------------------------------
def bool_match(y_true, y_pred):
    return K.switch(K.any(y_true-y_pred.round()), K.variable(0), K.variable(1))
# Make sure we can save and load a model with custom metric
kmet.bool_match = bool_match

# Custom metric returns the fraction of correctly set bits
# in y_pred vs y_true
#---------------------------------------------------------
def bitwise_match(y_true, y_pred):
    return 1.0 - K.mean(K.abs(y_true-y_pred.round()))
kmet.bitwise_match = bitwise_match

# Custom metric returns the fraction of correctly set elements
# in y_pred vs y_true
#-------------------------------------------------------------
def element_match(y_true, y_pred):
    return 1.0 - K.mean(K.abs(K.sign(y_true-K.round(y_pred))))
kmet.element_match = element_match


# Custom loss function to optimize element_match metric
# For some reason this is a bad idea and mse works better
#-------------------------------------------------------------
def element_loss(y_true, y_pred):
    return K.mean(K.abs(K.sign(y_true-K.round(y_pred))))
klo.element_loss = element_loss

# Custom loss.
# A simple crossentropy without checking anything.
# This works even if several prob vectors where flattened into one,
# like [[1,0],[1,0]] -> [1,0,1,0]
#--------------------------------------------------
def plogq(y_true, y_pred):
    res = -K.sum(y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon() )))
    return res
klo.plogq = plogq

# One-hot encode a list of integers
# (1,3,2) ->
# array([[ 0.,  1.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.],
#        [ 0.,  0.,  1.,  0.]])
#-------------------------------------
def onehot(x,num_classes=None):
    return to_categorical(x,num_classes)

# Feed one input to a model and return the result after some intermediate level
#----------------------------------------------------------------------------------
def get_output_of_layer( model, layer_name, input_data):
    intermediate_model = kmod.Model( inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    res = intermediate_model.predict( [input_data], batch_size=1 )
    return res


# Dump jpegs of model conv layer channels to file
#---------------------------------------------------------------------
def visualize_channels( model, layer_name, channels, img_, fname):
    img = img_.copy()
    img *= 2.0; img -= 1.0 # normalize to [-1,1] before feeding to model
    img = img.reshape( (1,) + img.shape) # Add dummy batch dimension
    channel_data = get_output_of_layer( model, layer_name, img)[0]
    nplots = len( channels) + 1 # channels plus orig
    ncols = 1
    nrows = nplots // ncols
    plt.figure( edgecolor='k')
    fig = plt.gcf()
    scale = 6.0
    fig.set_size_inches( scale*ncols, scale*nrows)

    # Show input image
    plt.subplot( nrows,ncols,1)
    ax = plt.gca()
    ax.get_xaxis().set_visible( False)
    ax.get_yaxis().set_visible( False)
    plt.imshow( img_) #  cmap='Greys')

    # Show output channels
    for idx,channel in enumerate( channels):
        data = channel_data[:,:,channel]
        # Normalization unnecessary, done automagically
        #mmin = np.min(data)
        #data -= mmin
        #mmax = np.max(data)
        #data /= mmax
        dimg  = cv2.resize( data, (img_.shape[1], img_.shape[0]), interpolation = cv2.INTER_NEAREST)
        plt.subplot( nrows, ncols, idx+2)
        ax = plt.gca()
        ax.get_xaxis().set_visible( False)
        ax.get_yaxis().set_visible( False)
        plt.imshow( dimg, cmap='binary', alpha=1.0)

    plt.tight_layout()
    plt.savefig( fname)

# Get the n indexes who predicted the wrong class,
# sorted descending by confidence
# Example:
# n_worst_results( 10, preds, meta['valid_classes'])
#----------------------------------------------------
def n_worst_results( n, preds, true_classes):
    pred_classes = [np.argmax(x) for x in preds]
    pred_confidences = [np.max(x) for x in preds]
    bad_indexes = [idx for idx,c in enumerate(true_classes) if c != pred_classes[idx]]
    bad_true_classes = np.array(true_classes)[bad_indexes]
    bad_pred_classes = np.array(pred_classes)[bad_indexes]
    sorted_indexes = sorted( bad_indexes, key=lambda idx: -pred_confidences[idx])
    worst_preds = np.array(pred_classes)[sorted_indexes]
    return sorted_indexes[:n], worst_preds

# Get the n indexes who predicted the correct class,
# sorted descending by confidence
# Example:
# n_worst_results( 10, preds, meta['valid_classes'])
#----------------------------------------------------
def n_best_results( n, preds, true_classes):
    pred_classes = [np.argmax(x) for x in preds]
    pred_confidences = [np.max(x) for x in preds]
    good_indexes = [idx for idx,c in enumerate(true_classes) if c == pred_classes[idx]]
    good_true_classes = np.array(true_classes)[good_indexes]
    good_pred_classes = np.array(pred_classes)[good_indexes]
    sorted_indexes = sorted( good_indexes, key=lambda idx: -pred_confidences[idx])
    # The easiest ones and the hardest ones
    return (sorted_indexes[:n], sorted_indexes[-n:])


# Save best and worst images for inspection
#-----------------------------------------------------------------------
def dump_n_best_and_worst( n, model, images, meta, sset='valid'):
    preds = model.predict(images['%s_data' % sset], batch_size=8)
    worst_indexes, worst_preds = n_worst_results( n, preds, meta['%s_classes' % sset])
    easiest_indexes, hardest_indexes = n_best_results( n, preds, meta['%s_classes' % sset])

    for i,idx in enumerate( worst_indexes):
        dsi( images['%s_data' % sset][idx],
             'worst_%s_%02d_%d' % (sset,i,worst_preds[i]) + os.path.basename( meta['%s_filenames' % sset][idx]))

    for i,idx in enumerate( easiest_indexes):
        dsi( images['%s_data' % sset][idx],
             'easiest_%s_%02d_' % (sset,i) + os.path.basename( meta['%s_filenames' % sset][idx]))

    for i,idx in enumerate( hardest_indexes):
        dsi( images['%s_data' % sset][idx],
             'hardest_%s_%02d_' % (sset,i) + os.path.basename( meta['%s_filenames' % sset][idx]))


################
# OS helpers
################

# Randomly split the sgf files in a folder into
# train, valid, test
#------------------------------------------------------------------
def split_files( folder, trainpct, validpct, substr='', nfiles=0):
    files = glob.glob( folder + '/*.sgf')
    files = [os.path.basename(f) for f in files];
    files = [f for f in files if substr in f];
    random.shuffle( files)
    if nfiles: files = files[:nfiles]
    ntrain = int( round( len( files) * (trainpct / 100.0)))
    nvalid = int( round( len( files) * (validpct / 100.0)))
    trainfiles = files[:ntrain]
    validfiles = files[ntrain:ntrain+nvalid]
    testfiles  = files[ntrain+nvalid:]

    os.makedirs( 'test/all_files')
    os.makedirs( 'train/all_files')
    os.makedirs( 'valid/all_files')

    for f in trainfiles:
        shutil.copy2( folder + '/' + f, 'train/all_files/' + f)
    for f in validfiles:
        shutil.copy2( folder + '/' + f, 'valid/all_files/' + f)
    for f in testfiles:
        shutil.copy2( folder + '/' + f, 'test/all_files/' + f)

# Return list of files matching filterstr
# Example: fing( '/tmp', '*.sgf')
#-------------------------------------------
def find( folder, filterstr):
    matches = []
    for root, dirnames, filenames in os.walk( folder):
        for filename in fnmatch.filter( filenames, filterstr):
            matches.append( os.path.join( root, filename))
    return matches


############
# Misc
############

# Return the eight symmetries of an nxn matrix
#-------------------------------------------------
def syms( nxn):
    res = []

    # Rotations
    tt = nxn.copy()
    res.append( tt)
    tt = np.rot90( tt)
    res.append( tt)
    tt = np.rot90( tt)
    res.append( tt)
    tt = np.rot90( tt)
    res.append( tt)

    # Flip and rot
    tt = nxn.copy()
    tt = np.flip( tt,1)
    res.append(tt)
    tt = np.rot90( tt)
    res.append( tt)
    tt = np.rot90( tt)
    res.append( tt)
    tt = np.rot90( tt)
    res.append( tt)

    return res
