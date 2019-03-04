#!/usr/bin/env python

# /********************************************************************
# Filename: naive.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# A Go playing agent making random moves
#

from pdb import set_trace as BP

import random
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.goboard_slow import Move
from dlgo.gotypes import Point

#=========================
class RandomBot(Agent):
    #-------------------------------------
    def select_move( self, game_state):
        candidates = []
        for r in range( 1, game_state.board.num_rows + 1):
            for c in range( 1, game_state.board.num_cols + 1):
                candidate = Point( r, c)
                valid = game_state.is_valid_move( Move.play(candidate))
                if (valid and
                    not is_point_an_eye( game_state.board,
                                         candidate,
                                         game_state.next_player)):
                    candidates.append( candidate)

        if not candidates:
            print( '%s pass' % str(game_state.next_player))
            return Move.pass_turn()

        print( '%s no pass' % str(game_state.next_player))
        res = Move.play( random.choice( candidates))
        return res
