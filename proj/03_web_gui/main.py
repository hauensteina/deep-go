#!/usr/bin/env python

# /********************************************************************
# Filename: web_gui/main.py
# Author: AHN
# Creation Date: Apr, 2019
# **********************************************************************/
#
# A web interface for Go experiments. Adapted from dlgo.
#

from pdb import set_trace as BP
import os, sys, re
import numpy as np
from datetime import datetime
import uuid
from io import BytesIO

import flask
from flask import jsonify,request,Response,send_file

import keras.models as kmod
from keras import backend as K
import tensorflow as tf

from gotypes import Point
from smart_random_bot import SmartRandomBot
from leelabot import LeelaBot
from leela_gtp_bot import LeelaGTPBot
from get_bot_app import get_bot_app
from sgf import Sgf_game
from go_utils import coords_from_point, point_from_coords
import goboard_fast as goboard
from encoder_base import get_encoder_by_name
from scoring import compute_nn_game_result

SCOREMODEL = None
LEELABOTMODEL = None

#----------------------------
def setup_models():
    print( '>>>>>>>>>> setup start')
    global SCOREMODEL
    global LEELABOTMODEL

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

        #path = os.path.dirname(__file__)
        SCOREMODEL = kmod.load_model( 'static/models/nn_score.hd5')
        SCOREMODEL._make_predict_function()
        LEELABOTMODEL = kmod.load_model( 'static/models/nn_leelabot.hd5')
        LEELABOTMODEL._make_predict_function()
    print( '>>>>>>>>>> setup end')

#------------
def main():
    print( 'Point your browser at http://127.0.0.1:5000/static/index.html')
    setup_models()
    smart_random_agent = SmartRandomBot()
    leelabot = LeelaBot( LEELABOTMODEL, SCOREMODEL )
    leela_cmd = 'leelaz -w best-network -p 1 --noponder --cpu-only'
    leela_gtp_bot = LeelaGTPBot( leela_cmd.split() )

    # Get an app with 'select-move/<botname>' endpoints
    app = get_bot_app( {'smartrandom':smart_random_agent, 'leelabot':leelabot, 'leela_gtp_bot':leela_gtp_bot} )

    @app.after_request
    #---------------------
    def add_header(r):
        """
        Add headers to both force latest IE rendering engine or Chrome Frame,
        and also to cache the rendered page for 10 minutes.
        """
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        r.headers['Cache-Control'] = 'public, max-age=0'
        return r

    #--------------------------------------
    # Add some more endpoints to the app
    #--------------------------------------

    @app.route('/histo', methods=['POST'])
    # Take a bunch of numbers, number of bins, min, max and return a histo.
    #------------------------------------------------------------------------
    def histo():
        data,nbins,mmin,mmax = request.json
        counts,borders = np.histogram( data, nbins, [mmin, mmax])
        counts = counts.tolist()
        borders = borders.tolist()
        centers = [ (borders[i] + borders[i+1]) / 2.0 for i in range(len(borders)-1) ]
        res = list(zip( centers, counts))
        return jsonify( res)

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

        territory, res = compute_nn_game_result( lab, game_state.next_player)
        white_probs = lab[0].tolist()
        return jsonify( {'result':res, 'territory':territory.__dict__ , 'white_probs':white_probs} )


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
        komi = sgf.get_komi()
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

        return jsonify( {'result': {'moves':moves, 'pb':player_black, 'pw':player_white,
                                    'winner':winner, 'komi':komi, 'fname':fname} } )

    # Convert a list of moves like ['Q16',...] to sgf
    #---------------------------------------------------
    def moves2sgf( moves):
        sgf = '(;FF[4]SZ[19]\n'
        sgf += 'SO[leela-one-playout.herokuapp.com]\n'
        dtstr = datetime.now().strftime('%Y-%m-%d')
        sgf += 'DT[%s]\n' % dtstr

        movestr = ''
        result = ''
        color = 'B'
        for move in moves:
            othercol = 'W' if color == 'B' else 'B'
            if move == 'resign':
                result = 'RE[%s+R]' % othercol
            elif move == 'pass':
                movestr += ';%s[tt]' % color
            else:
                #BP()
                p = point_from_coords( move)
                col_s = 'abcdefghijklmnopqrstuvwxy'[p.col - 1]
                row_s = 'abcdefghijklmnopqrstuvwxy'[19 - p.row]
                movestr += ';%s[%s%s]' % (color,col_s,row_s)
            color = othercol

        sgf += result
        sgf += movestr
        sgf += ')'
        return sgf

    @app.route('/save-sgf', methods=['GET'])
    # Convert moves to sgf.
    # Moves come like 'Q16D4...' to shorten URL.
    #-------------------------------------------------------------
    def save_sgf():
        moves = request.args.get( 'moves')
        movearr = []
        m = ''
        for c in moves:
            if c > '9': # a letter
                if m: movearr.append(m)
                m = c
            else:
                m += c
        if m: movearr.append(m)
        result = moves2sgf( movearr)
        fname = uuid.uuid4().hex + '.sgf'
        fh = BytesIO( result.encode('utf8'))
        resp = send_file( fh, as_attachment=True, attachment_filename=fname)
        return resp

    app.run( host='0.0.0.0')

if __name__ == '__main__':
    main()
