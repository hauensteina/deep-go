#!/usr/bin/env python

# /********************************************************************
# Filename: server.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# REST-Api to get the next move from a bot, or score the game
#

from pdb import set_trace as BP
import os

from flask import Flask
from flask import jsonify
from flask import request

from dlgo.gotypes import Player, Point
from dlgo.utils import print_board, print_move
from dlgo import agent
from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords
from dlgo.scoring import compute_game_result
from dlgo.gosgf import Sgf_game

__all__ = [
    'get_web_app',
]

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
    # Score the current position
    #-----------------------------
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

        res = compute_game_result( game_state)
        return jsonify( {'result': res})

    @app.route('/sgf2list', methods=['POST'])
    # Convert sgf main var to coordinate list of moves
    #----------------------------------------------------
    def sgf2list():
        f = request.files['file']
        sgfstr = f.read()
        sgf = Sgf_game.from_string( sgfstr)
        res = []

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
                    if idx > 0: res.append( 'pass')
                    res.append( move2coords( move))

        # Nodes in the main sequence
        for item in sgf.main_sequence_iter():
            color, move_tuple = item.get_move()
            point = None
            if color is not None:
                if move_tuple is not None:
                    res.append( move2coords( move_tuple))
                else:
                    res.append( 'pass')
            # Deal with handicap stones as individual nodes
            elif item.get_setup_stones()[0]:
                move = list( item.get_setup_stones()[0])[0]
                if res: res.append( 'pass')
                res.append( move2coords( move))

        return jsonify( {'result': res})

    return app
