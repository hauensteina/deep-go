#!/usr/bin/env python

# /********************************************************************
# Filename: 01_encode.py
# Author: AHN
# Creation Date: Apr, 2019
# **********************************************************************/
#
# Encode sgf files for later NN training.
# Learns to find the next move.
#

from pdb import set_trace as BP
import os, sys, re
import uuid
import argparse
import numpy as np
import time
from multiprocessing import Pool

# Look for modules further up
SCRIPTPATH = os.path.dirname( os.path.realpath( __file__))
sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import get_encoder_by_name
from dlgo.utils import print_board, print_move
from dlgo.scoring import compute_game_result

import pylib.ahnutil as ut

ENCODER = 'score_threeplane_encoder'
CHUNKSIZE = 100000
MAX_MOVES = 250

g_generator = None

#---------------------------
def usage( printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s -- Encode sgf files below a folder into numpy arrays and save them
    Synopsis:
      %s --folder <folder> --nprocs <nprocs>
    Description:
       Results go to <folder>/chunks.
       The --nprocs option decides how many processes to use.
    Example:
      %s --folder train
      Output will be in train_feat.npy and train_lab.npy
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-----------
def main():
    global g_generator

    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--folder", required=True)
    parser.add_argument( "--nprocs", required=False, type=int, default=1)
    args = parser.parse_args()

    g_generator = NextMoveDataGenerator( args.nprocs, data_directory = args.folder)
    g_generator.encode_sgf_files()


#-------------------------------
def worker( fname_indexes):
    feat_shape = g_generator.encoder.shape()
    bsz = g_generator.board_sz
    lab_shape = ( bsz * bsz, )
    features = []
    labels = []

    for idx,file_idx in enumerate( fname_indexes):
        if len(features) > CHUNKSIZE:
            g_generator.save_chunk( features, labels)

        if idx % 100 == 0:
            print( '%d / %d' % (idx, len( fname_indexes)))

        # Get score and number of moves
        sgfstr = open( g_generator.fnames[file_idx]).read()
        sgf = Sgf_game.from_string( sgfstr)
        game_state = GameState.new_game( g_generator.board_sz)
        move_counter = 0
        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            if color:
                if move_tuple:
                    row, col = move_tuple
                    point = Point( row+1, col+1)
                    # Save position and predicted move
                    encoded = g_generator.encoder.encode( game_state)
                    next_move = np.full( (bsz, bsz), 0, dtype='int8')
                    next_move[row, col] = 1
                    # Symmetries
                    labsyms  = ut.syms( next_move)
                    labsyms = [x.flatten() for x in labsyms]
                    featsyms = ut.syms( encoded)
                    features.extend( featsyms)
                    labels.extend( labsyms)
                else: # We're done with this game
                    break

                # Play the move
                move = Move.play( point)
                game_state = game_state.apply_move( move)
                move_counter += 1
                if move_counter > MAX_MOVES: break

# Generate encoded positions and labels to train a move predictor.
# The label is a 361 long one-hot vector indicating the next move.
# Stuff postions and labels into numpy arrays and save them to files, in chunks.
#==================================================================================
class NextMoveDataGenerator:

    #------------------------------------------------------------------------
    def __init__( self, nprocs, encoder=ENCODER, data_directory='train'):
        self.board_sz = 19
        self.nprocs = nprocs
        self.encoder = get_encoder_by_name( encoder, self.board_sz)
        self.data_dir = data_directory
        self.fnames = ut.find( self.data_dir, '*.sgf')

    # Save a chunk of features and labels and remove the chunk from input
    #----------------------------------------------------------------------
    def save_chunk( self, features, labels):
        chunkdir = '%s/chunks' % self.data_dir
        if not os.path.exists( chunkdir):
            os.makedirs( chunkdir)
        bname = uuid.uuid4().hex
        featfname = chunkdir + '/%s_feat.npy' % bname
        labfname  = chunkdir + '/%s_lab.npy' % bname
        np.save( featfname, np.array( features[:CHUNKSIZE]))
        np.save( labfname, np.array( labels[:CHUNKSIZE]))
        features[:] = features[CHUNKSIZE:]
        labels[:] = labels[CHUNKSIZE:]

    #-------------------------------
    def encode_sgf_files( self):
        # Generate files for each process to work on
        fname_indexes = [ [] for _ in range( self.nprocs) ]
        for idx,fname in enumerate( self.fnames):
            procnum = int( ( abs( hash( fname) / sys.maxsize) * self.nprocs))
            fname_indexes[procnum].append( idx)

        for procnum in range( self.nprocs):
           np.random.shuffle( fname_indexes[procnum])

        # Farm out to processes
        p = Pool( self.nprocs)
        p.map( worker, fname_indexes)
        #worker( fname_indexes[0]) # debug

if __name__ == '__main__':
    main()
