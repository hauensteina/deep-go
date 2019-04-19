#!/usr/bin/env python

# /********************************************************************
# Filename: score19_server.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Start a web interface to test 19x19 scoring
#

from pdb import set_trace as BP
import os, sys, re
import numpy as np

import keras.models as kmod
from keras import backend as K
import tensorflow as tf

#from dlgo.agent.naive_fast import FastRandomBot
from dlgo.agent.naive_fast_smart import FastSmartRandomBot
from dlgo.agent.leelabot import LeelaBot
from dlgo import mcts
from dlgo.httpfrontend.server import get_web_app

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

        path = os.path.dirname(__file__)
        SCOREMODEL = kmod.load_model( path + 'dlgo/httpfrontend/nn_score.hd5')
        SCOREMODEL._make_predict_function()
        LEELABOTMODEL = kmod.load_model( 'dlgo/httpfrontend/nn_leelabot.hd5')
        LEELABOTMODEL._make_predict_function()
    print( '>>>>>>>>>> setup end')

print( 'Point your browser at http://127.0.0.1:5000/static/score19.html')
setup_models()
smart_random_agent = FastSmartRandomBot()
leelabot = LeelaBot( LEELABOTMODEL, SCOREMODEL )

web_app = get_web_app( {'smartrandom':smart_random_agent, 'leelabot':leelabot}, SCOREMODEL )

web_app.run( host='0.0.0.0')
