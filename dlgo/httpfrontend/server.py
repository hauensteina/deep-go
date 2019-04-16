#!/usr/bin/env python

# /********************************************************************
# Filename: server.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# REST API to get the next move from a bot, or score the game
#

from pdb import set_trace as BP
import os
import numpy as np

from flask import Flask
from flask import jsonify
from flask import request

import keras.models as kmod
from keras import backend as K
import tensorflow as tf


from dlgo.gotypes import Player, Point
from dlgo.utils import print_board, print_move
from dlgo import agent
from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords
from dlgo.scoring import compute_game_result, compute_nn_game_result
from dlgo.gosgf import Sgf_game
from dlgo.encoders.base import get_encoder_by_name

__all__ = [
    'get_web_app',
]

SCOREMODEL = None

#----------------------------
def setup_nnscore_model():
    global SCOREMODEL
    num_cores = 8
    GPU = 0

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

        path = os.path.dirname(__file__)
        SCOREMODEL = kmod.load_model( path + '/nn_score.hd5')
        SCOREMODEL._make_predict_function()

#---------------------------
def get_web_app(bot_map):
    """Create a flask application for serving bot moves.

    The bot_map maps from URL path fragments to Agent instances.

    The /static path will return some static content (including the
    jgoboard JS).

    Clients can get the post move by POSTing json to
    /select-move/<bot name>

    Example:

    >>> myagent = agent.RandomBot()
    >>> web_app = get_web_app({'random': myagent})
    >>> web_app.run()

    Returns: Flask application instance
    """
    here = os.path.dirname(__file__)
    static_path = os.path.join(here, 'static')
    app = Flask(__name__, static_folder=static_path, static_url_path='/static')
    setup_nnscore_model()

    @app.route('/select-move/<bot_name>', methods=['POST'])
    # Ask the named bot for the next move
    #--------------------------------------
    def select_move(bot_name):
        content = request.json
        board_size = content['board_size']
        game_state = goboard.GameState.new_game( board_size)
        # Replay the game up to this point.
        for move in content['moves']:
            if move == 'pass':
                next_move = goboard.Move.pass_turn()
            elif move == 'resign':
                next_move = goboard.Move.resign()
            else:
                next_move = goboard.Move.play( point_from_coords(move))
            game_state = game_state.apply_move( next_move)
            #print_board( game_state.board)
        bot_agent = bot_map[bot_name]
        bot_move = bot_agent.select_move( game_state)
        if bot_move is None or bot_move.is_pass:
            bot_move_str = 'pass'
        elif bot_move.is_resign:
            bot_move_str = 'resign'
        else:
            bot_move_str = coords_from_point( bot_move.point)
        return jsonify({
            'bot_move': bot_move_str,
            'diagnostics': bot_agent.diagnostics()
        })

    @app.route('/score', methods=['POST'])
    # Score the current position using naive Tromp Taylor
    #------------------------------------------------------
    def score():
        content = request.json
        board_size = content['board_size']
        game_state = goboard.GameState.new_game( board_size)
        # Replay the game up to this point.
        for move in content['moves']:
            if move == 'pass':
                next_move = goboard.Move.pass_turn()
            elif move == 'resign':
                next_move = goboard.Move.resign()
            else:
                next_move = goboard.Move.play( point_from_coords(move))
            game_state = game_state.apply_move( next_move)

        territory, res = compute_game_result( game_state)
        return jsonify( {'result': res, 'territory': territory.__dict__ })

    @app.route('/nnscore', methods=['POST'])
    # Score the current position using our convolutional network
    #-------------------------------------------------------------
    def nnscore():
        content = request.json
        board_size = content['board_size']
        game_state = goboard.GameState.new_game( board_size)
        # Replay the game up to this point.
        for move in content['moves']:
            if move == 'pass':
                next_move = goboard.Move.pass_turn()
            elif move == 'resign':
                next_move = goboard.Move.resign()
            else:
                next_move = goboard.Move.play( point_from_coords(move))
            game_state = game_state.apply_move( next_move)

        enc  = get_encoder_by_name( 'score_threeplane_encoder', board_size)
        feat = np.array( [ enc.encode( game_state) ] )
        lab  = SCOREMODEL.predict( [feat], batch_size=1)

        territory, res = compute_nn_game_result( lab)
        return jsonify( {'result': res, 'territory': territory.__dict__ })

    @app.route('/sgf2list', methods=['POST'])
    # Convert sgf main var to coordinate list of moves
    #----------------------------------------------------
    def sgf2list():
        f = request.files['file']
        sgfstr = f.read()
        sgf = Sgf_game.from_string( sgfstr)
        player_white = sgf.get_player_name('w')
        player_black = sgf.get_player_name('b')
        winner = sgf.get_winner()
        fname = f.filename

        res = {}
        moves = []

        #------------------------
        def move2coords( move):
            row, col = move
            p = Point( row + 1, col + 1)
            coords = coords_from_point( p)
            return coords

        # Deal with handicap in the root node
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for idx, move in enumerate( setup):
                    if idx > 0: moves.append( 'pass')
                    moves.append( move2coords( move))

        # Nodes in the main sequence
        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            point = None
            if color is not None:
                if move_tuple is not None:
                    moves.append( move2coords( move_tuple))
                else:
                    moves.append( 'pass')
            # Deal with handicap stones as individual nodes
            elif item.get_setup_stones()[0]:
                move = list( item.get_setup_stones()[0])[0]
                if moves: moves.append( 'pass')
                moves.append( move2coords( move))

        return jsonify( {'result': {'moves':moves, 'pb':player_black, 'pw':player_white, 'winner':winner, 'fname':fname} } )

    return app
