#!/usr/bin/env python

# /*********************************
# Filename: leelabot.py
# Creation Date: Apr, 2019
# Author: AHN
# **********************************/
#
# A bot trained from leela selfplay games.
#

from pdb import set_trace as BP
import os, sys, re
import numpy as np

import keras.models as kmod
from keras import backend as K
import tensorflow as tf

# Look for modules further up
SCRIPTPATH = os.path.dirname( os.path.realpath( __file__))
sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

from dlgo import goboard
from dlgo.agent.base import Agent
from dlgo.agent.helpers_fast import is_point_an_eye
from dlgo.goboard_fast import Move
from dlgo.gotypes import Point
from dlgo.encoders.base import get_encoder_by_name

ENCODER = 'score_threeplane_encoder'
BSZ = 19

#===========================
class LeelaBot( Agent):

    #----------------------------
    def __init__( self, model):
        Agent.__init__( self)
        self.model = model
        path = os.path.dirname(__file__)
        self.encoder = get_encoder_by_name( ENCODER, BSZ)

    #-------------------------------
    def predict( self, game_state):
        encoded_state = self.encoder.encode( game_state)
        input_tensor = np.array( [encoded_state])
        return self.model.predict( input_tensor)[0]

    #-----------------------------------
    def select_move( self, game_state):
        num_moves = BSZ * BSZ
        move_probs = self.predict( game_state)
        candidates = np.arange( num_moves)
        ranked_moves = sorted( candidates, key = lambda idx: move_probs[idx], reverse=True)
        print( 'Ranked:'); print( ranked_moves[:3])
        print( 'Probs:'); print( sorted(move_probs, reverse=True)[:3])
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index( point_idx)
            if (game_state.is_valid_move( goboard.Move.play( point))
                  and not is_point_an_eye( game_state.board, point, game_state.next_player)):
                print( 'played: %d' % point_idx)
                return goboard.Move.play( point)

        return goboard.Move.pass_turn()
