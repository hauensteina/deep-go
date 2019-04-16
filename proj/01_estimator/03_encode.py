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

#REWINDS = (200,150,100,50,0) # How far to rewind from game end
REWINDS = (50, 100, 150, -50, -60, -70, -80, -90, -100, 110, -120)
#REWINDS = (125, 150, 175, 200)
#REWINDS = (150, 175, 200, 225)

ENCODER = 'score_threeplane_encoder'
#ENCODER = 'score_twoplane_encoder'
#ENCODER = 'score_stringonly_encoder'
#ENCODER = 'score_string_generator'
#ENCODER = 'score_encoder'
CHUNKSIZE = 1000

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

    g_generator = ScoreDataGenerator( args.nprocs, data_directory = args.folder)
    g_generator.encode_sgf_files()

#-------------------------------
def worker( fname_rewinds):
    feat_shape = g_generator.encoder.shape()
    lab_shape = ( g_generator.board_sz * g_generator.board_sz, )
    features = []
    labels = []

    for idx,fname_rewind in enumerate( fname_rewinds):
        if len(features) > CHUNKSIZE:
            g_generator.save_chunk( features, labels)

        if idx % 100 == 0:
            print( '%d / %d' % (idx, len( fname_rewinds)))

        # Get score and number of moves
        file_idx, rewind = fname_rewind
        sgfstr = open( g_generator.fnames[file_idx]).read()
        nmoves, territory = g_generator.score_sgf( sgfstr)
        label = territory.encode_sigmoid()

        sgf = Sgf_game.from_string( sgfstr)
        game_state = GameState.new_game( g_generator.board_sz)
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
            if (rewind < 0 and move_counter == nmoves + rewind) or (rewind > 0 and move_counter == rewind):
                encoded = g_generator.encoder.encode( game_state)
                # Get all eight symmetries
                featsyms = ut.syms( encoded)
                labsyms  = ut.syms( label)
                labsyms = [x.flatten() for x in labsyms]
                features.extend( featsyms)
                labels.extend( labsyms)
                break

# Generate encoded positions and labels to train a score estimator.
# Encode snapshots at N-150, N-100, etc in a game of length N.
# The game must have been played out until no dead stones
# are left. The label is a 361 long vector of 0 or 1 indicating whether an
# intersection ends up being black or white. Each snapshot for a game
# gets the label from the final position.
# Stuff postions and labels into numpy arrays and save them to files, in chunks.
#==================================================================================
class ScoreDataGenerator:

    #------------------------------------------------------------------------
    def __init__( self, nprocs, encoder=ENCODER, data_directory='train'):
        self.board_sz = 19
        self.nprocs = nprocs
        self.encoder = get_encoder_by_name( encoder, self.board_sz)
        self.data_dir = data_directory
        self.fnames = ut.find( self.data_dir, '*.sgf')

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
        # Generate files and rewinds for each process to work on
        fname_rewinds = [ [] for _ in range( self.nprocs) ]
        for idx,fname in enumerate( self.fnames):
            for rewind in REWINDS:
                tstr = '%d_%d' % (idx,rewind)
                procnum = int( ( abs( hash( tstr) / sys.maxsize) * self.nprocs))
                fname_rewinds[procnum].append( (idx, rewind) )

        for procnum in range( self.nprocs):
           np.random.shuffle( fname_rewinds[procnum])

        # Farm out to processes
        p = Pool( self.nprocs)
        p.map( worker, fname_rewinds)
        #worker( fname_rewinds[0])

if __name__ == '__main__':
    main()
