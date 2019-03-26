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
SCRIPTPATH = os.path.dirname(os.path.realpath( __file__))
sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import get_encoder_by_name
from dlgo.utils import print_board, print_move
from dlgo.scoring import compute_game_result

import pylib.ahnutil as ut

# Generate encoded positions and labels to train a score estimator.
# Encode snapshots at N-200, N-150, etc in a game of length N in a single
# plane representation. The game must have been played out until no dead stones
# are left. The label is a 361 long vector of -1,0,1 indicating whether an
# intersection ends up being black, neutral, or white. Each snapshot for a game
# gets the label from the final position.
# Stuff all (plane,label) pairs into a huge numpy array and save to file.
#==========================
class ScoreDataGenerator:

    #------------------------------------------------------------------------
    def __init__( self, encoder='score_encoder', data_directory='train'):
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

    #----------------------------------------------
    def encode_sgf_files( self, num_samples=10):
        rewinds = (200,150,100,50,0) # How far to rewind from game end
        fnames = ut.find( self.data_dir, '*.sgf')

        feat_shape = self.encoder.shape()
        lab_shape = ( self.board_sz * self.board_sz, )
        features = []
        labels = []
        # for each sgf_file
        nsamples = 0
        for f in fnames:
            print(f)
            # Get score and number of moves
            sgfstr = open(f).read()
            nmoves, territory = self.score_sgf( sgfstr)
            label = territory.encode()

            sgf = Sgf_game.from_string( sgfstr)
            snaps = [nmoves - x for x in rewinds]
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
                        features.append( encoded)
                        labels.append( label)
                        nsamples += 1

        tt = np.array( features)
        np.save( '%s_feat.npy' % self.data_dir, tt)
        tt = np.array( labels)
        np.save( '%s_lab.npy' % self.data_dir, tt)

#---------------------------
def usage(printmsg=False):
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
