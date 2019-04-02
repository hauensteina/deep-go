#!/usr/bin/env python

# /********************************************************************
# Filename: 01_encode.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Encode sgf files for later NN training.
# Simple single plane encoder from DLGG.
#

from pdb import set_trace as BP
import os, sys, re
import argparse
import numpy as np

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

#REWINDS = (200,150,100,50,0) # How far to rewind from game end
REWINDS = (50, 75, 100, 125)
#REWINDS = (125, 150, 175, 200)
#REWINDS = (150, 175, 200, 225)

ENCODER = 'score_encoder'
CHUNKSIZE = 1024

# Generate encoded positions and labels to train a score estimator.
# Encode snapshots at N-150, N-100, etc in a game of length N in a single
# plane representation. The game must have been played out until no dead stones
# are left. The label is a 361 long vector of 0, 0.5, 1 indicating whether an
# intersection ends up being black, neutral, or white. Each snapshot for a game
# gets the label from the final position.
# Stuff postions and labels into two huge numpy arrays and save them to file.
#==================================================================================
class ScoreDataGenerator:

    #------------------------------------------------------------------------
    def __init__( self, encoder=ENCODER, data_directory='train'):
        self.board_sz = 19
        self.encoder = get_encoder_by_name( encoder, self.board_sz)
        self.data_dir = data_directory

    # Return number of moves, and territory points for b,w,dame
    #--------------------------------------------------------------
    def score_sgf( self, sgfstr):
        sgf = Sgf_game.from_string( sgfstr)
        game_state = GameState.new_game( self.board_sz)
        nmoves = 0
        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            if color:
                if move_tuple:
                    row, col = move_tuple
                    point = Point( row+1, col+1)
                    move = Move.play( point)
                else:
                    move = Move.pass_turn()
                game_state = game_state.apply_move(move)
                nmoves += 1
        territory, _ = compute_game_result( game_state)
        return nmoves, territory

    # Save a chunk of features and labels and remove the chunk from input
    #----------------------------------------------------------------------
    def save_chunk( self, chunknum, features, labels):
        chunkdir = '%s/chunks' % self.data_dir
        if not os.path.exists( chunkdir):
            os.makedirs( chunkdir)
        featfname = chunkdir + '/feat_%0.7d.npy' % chunknum
        labfname  = chunkdir + '/lab_%0.7d.npy' % chunknum
        np.save( featfname, np.array( features[:CHUNKSIZE]))
        np.save( labfname, np.array( labels[:CHUNKSIZE]))
        features[:] = features[CHUNKSIZE:]
        labels[:] = labels[CHUNKSIZE:]

    #----------------------------------------------
    def encode_sgf_files( self):
        fnames = ut.find( self.data_dir, '*.sgf')
        feat_shape = self.encoder.shape()
        lab_shape = ( self.board_sz * self.board_sz, )
        features = []
        labels = []
        # for each sgf_file
        chunknum = 0
        for idx,f in enumerate(fnames):
            if len(features) > CHUNKSIZE:
                self.save_chunk( chunknum, features, labels)
                chunknum += 1

            if idx % 100 == 0:
                print( '%d / %d' % (idx, len(fnames)))
            # Get score and number of moves
            sgfstr = open(f).read()
            nmoves, territory = self.score_sgf( sgfstr)
            label = territory.encode_sigmoid()

            sgf = Sgf_game.from_string( sgfstr)
            snaps = [nmoves - x for x in REWINDS]
            game_state = GameState.new_game( self.board_sz)
            move_counter = 0
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                if color:
                    if move_tuple:
                        row, col = move_tuple
                        point = Point( row+1, col+1)
                        move = Move.play( point)
                    else:
                        move = Move.pass_turn()

                    game_state = game_state.apply_move( move)
                    move_counter += 1
                    if move_counter in snaps:
                        encoded = self.encoder.encode( game_state)
                        #label = self.label_bdead( encoded, label)
                        # Get all eight symmetries
                        featsyms = ut.syms( encoded)
                        labsyms  = ut.syms( label)
                        labsyms = [x.flatten() for x in labsyms]
                        features.extend( featsyms)
                        labels.extend( labsyms)

    # Any b stone in encoded that isn't in label is dead.
    # Black dead = 1, all else is 0.
    #---------------------------------------------------------
    def label_bdead( self, encoded, label):
        res = np.full( (self.board_sz, self.board_sz), 0, dtype='int8')
        for r in range( 0, self.board_sz):
            for c in range( 0, self.board_sz):
                if encoded[r,c,0] != -1: continue
                if label[r,c] != 0:
                    res[r,c] = 1.0
        return res

#---------------------------
def usage( printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s -- Encode sgf files below a folder into numpy arrays and save them
    Synopsis:
      %s --folder <folder>
    Description:
       Results go to <folder>_feat.npy and <folder>_lab.npy
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
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--folder", required=True)
    args = parser.parse_args()

    proc = ScoreDataGenerator( data_directory = args.folder)
    proc.encode_sgf_files()


if __name__ == '__main__':
    main()
