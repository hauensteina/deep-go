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
import os
import sys, glob, shutil, re
import numpy as np
#from keras.utils import to_categorical

# Look for modules further up
SCRIPTPATH = os.path.dirname(os.path.realpath( __file__))
sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import get_encoder_by_name
from dlgo.utils import print_board, print_move
from dlgo.scoring import compute_game_result
#from dlgo.data.sampling import Sampler

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
    def __init__( self, encoder='score_encoder', data_directory='sgf_data'):
        self.encoder = get_encoder_by_name( encoder, 19)
        self.data_dir = data_directory

    # Return number of moves, and territory points for b,w,dame
    #--------------------------------------------------------------
    def score_sgf( self, sgfstr):
        sgf = Sgf_game.from_string( sgfstr)
        game_state = GameState.new_game(19)
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
        fnames = [x for x in os.listdir( self.data_dir) if x.endswith( '.sgf')]
        nsamples = 0
        shape = self.encoder.shape()
        rewinds = (200,151,100,50,0) # How far to rewind from game end
        feature_shape = np.insert( shape, 0, len(rewinds) * len(fnames))
        features = np.zeros( feature_shape)
        # for each sgf_file
        for f in fnames:
            print(f)
            # Get score and number of moves
            sgfstr = open( self.data_dir + '/' + f).read()
            nmoves, territory = self.score_sgf( sgfstr)

            sgf = Sgf_game.from_string( sgfstr)
            snaps = [nmoves - x for x in rewinds]
            game_state = GameState.new_game(19)
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

                    game_state = game_state.apply_move(move)
                    move_counter += 1
                    if move_counter in snaps:
                        features[nsamples] = self.encoder.encode( game_state)
                        #@@@ cont here: compute and store scoring label
                        nsamples += 1

        BP()
        pass


#-----------
def main():
    proc = ScoreDataGenerator()
    proc.encode_sgf_files()

if __name__ == '__main__':
    main()
