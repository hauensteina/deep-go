#!/usr/bin/env python

# /********************************************************************
# Filename: 01_encode.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Encode sgf files for later NN training.
# Three plane encoder from DLGG.
#

from pdb import set_trace as BP
import os, sys, re
import uuid
import argparse
import numpy as np
import time
from multiprocessing import Pool

import keras.models as kmod
from keras import backend as K
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session( config=config)
K.set_session( session)


# Look for modules further up
SCRIPTPATH = os.path.dirname( os.path.realpath( __file__))
sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import get_encoder_by_name
from dlgo.utils import print_board, print_move
from dlgo.scoring import compute_game_result, Territory, GameResult
from dlgo.utils import p2idx

import pylib.ahnutil as ut

#REWINDS = (200,150,100,50,0) # How far to rewind from game end
REWINDS = (50, 100, 150, -50, -60, -70, -80, -90, -100, -110, -120)
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
        self.scoremodel = kmod.load_model( 'nn_score.hd5')

    # Return number of moves, and territory points for b,w,dame.
    # Finds the smallest move where we can score, then scores.
    #--------------------------------------------------------------
    def score_sgf( self, sgfstr):
        BOARDSZ = 19

        # Find the first index where a condition is true, by bisection.
        # This assumes that there is a point where all to the left are false,
        # and all to the right are true.
        #----------------------------------------------------------------------
        def min_true( data, N, testfunc):
            top = N-1
            bottom = 0
            idx = -1
            while top >= bottom:
                mid = int( (top+bottom) / 2 )
                if testfunc( data, mid):
                    top = mid-1
                    idx = mid
                else:
                    bottom = mid+1
            return idx

        #-------------------------
        def run_net( game_state):
            enc  = get_encoder_by_name( 'score_threeplane_encoder', BOARDSZ)
            feat = np.array( [ enc.encode( game_state) ] )
            lab  = self.scoremodel.predict( [feat], batch_size=1)
            white_probs = lab[0].tolist()
            return feat, lab, white_probs

        #------------------------
        def goto_move( moves, k):
            game_state = GameState.new_game( self.board_sz)
            # Go to move k
            for (idx,item) in enumerate(moves):
                if idx > k: break
                color, move_tuple = item.get_move()
                if color:
                    if move_tuple:
                        row, col = move_tuple
                        point = Point( row+1, col+1)
                        move = Move.play( point)
                    else:
                        move = Move.pass_turn()
                    game_state = game_state.apply_move(move)
            return game_state

        # A neutral point cannot border on territory.
        # Stones cannot be neutral.
        #---------------------------------------------
        def bad_neutrals( white_probs, board):
            for idx, prob in enumerate( white_probs):
                if color( prob) != 'n': continue
                point = self.encoder.decode_point_index( idx)
                if board.get( point) is not None:
                    return True
                neighs = point.neighbors()
                for neigh in neighs:
                    if not board.is_on_grid( neigh): continue
                    if board.get( neigh) is None:
                        nidx = self.encoder.encode_point( neigh)
                        if color( white_probs[nidx]) != 'n':
                            return True
            return False

        #-------------------------
        def color( wprob):
            #NEUTRAL_THRESH = 0.40
            #NEUTRAL_THRESH = 0.15
            NEUTRAL_THRESH = 0.30
            if abs(0.5 - wprob) < NEUTRAL_THRESH: return 'n'
            elif wprob > 0.5: return 'w'
            else: return 'b'

        #-----------------------------
        def scorable( moves, k):
            NEUTRAL_COUNT_THRESH = 5
            game_state = goto_move( moves, k)
            # Try to score
            _,_,white_probs = run_net( game_state)
            neutral_count = sum( 1 for p in white_probs if color(p) == 'n')
            if bad_neutrals( white_probs, game_state.board): return False
            if neutral_count < NEUTRAL_COUNT_THRESH: return True
            return False

        #----------------------------------------------------
        def save_sgf( sgfstr, scorable_move_idx, bpoints):
            score = 361 - 2 * bpoints
            if score > 0:
                res = 'W+%d' % score
            else:
                res = 'B+%d' % -score

            fname = 'tt/%s_%d_%s.sgf' %  (uuid.uuid4().hex[:7], scorable_move_idx, res)
            open( fname, 'w').write( sgfstr)

        # Fix terrmap such that all stones in a string are alive or dead.
        # Decide by average score.
        #----------------------------------------------------------------
        def enforce_strings( terrmap, game_state):
            BSZ = game_state.board.num_rows
            strs = game_state.board.get_go_strings()
            for gostr in strs:
                avg_col = 0.0
                for idx,point in enumerate(gostr.stones):
                    prob_white = white_probs[ p2idx( point, BSZ)]
                    avg_col = avg_col * (idx/(idx+1)) + prob_white / (idx+1)

                truecolor = 'territory_b' if avg_col < 0.5 else 'territory_w'

                for point in gostr.stones:
                    terrmap[point] = truecolor

            colcounts = {'territory_b':0, 'territory_w':0, 'dame':0}
            for p in terrmap: colcounts[terrmap[p]] += 1
            return colcounts['territory_b'],colcounts['territory_w'],colcounts['dame']

        #--------------------------------------
        def score( white_probs, game_state):
            BSZ = game_state.board.num_rows
            terrmap = {}
            for r in range( 1, BSZ+1):
                for c in range( 1, BSZ+1):
                    p = Point( row=r, col=c)
                    prob_white = white_probs[ p2idx( p, BSZ)]
                    if color( prob_white) == 'w':
                        terrmap[p] = 'territory_w'
                    elif color( prob_white) == 'b':
                        terrmap[p] = 'territory_b'
                    else:
                        terrmap[p] = 'dame'

            bpoints, wpoints, dame = enforce_strings( terrmap, game_state)

            player = game_state.next_player
            for i in range(dame):
                if player == Player.black:
                    bpoints += 1
                else:
                    wpoints += 1
                player = player.other
            territory = Territory( terrmap)
            return (territory, GameResult( bpoints, wpoints, komi=0))
        # END score()

        sgf = Sgf_game.from_string( sgfstr)
        movelist = list(sgf.main_sequence_iter())
        N = len(movelist)
        scorable_move_idx = min_true( movelist, N, scorable)
        game_state = goto_move( movelist, scorable_move_idx)
        _,_,white_probs = run_net( game_state)
        terr, res = score( white_probs, game_state)
        save_sgf( sgfstr, scorable_move_idx, res.b)
        return scorable_move_idx,terr
    # END score_sgf()

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
        if self.nprocs > 1:
            p = Pool( self.nprocs)
            p.map( worker, fname_rewinds)
        else: # singel thread
            worker( fname_rewinds[0])

if __name__ == '__main__':
    main()
