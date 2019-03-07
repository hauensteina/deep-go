#!/usr/bin/env python

# /********************************************************************
# Filename: score19_server.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Start a web interface to test 19x19 scoring
#

from dlgo.agent.naive_fast import FastRandomBot
from dlgo import mcts
from dlgo.httpfrontend.server import get_web_app

print( 'Point your browser at http://127.0.0.1:5000/static/score19.html')
random_agent = FastRandomBot()
web_app = get_web_app({'random':random_agent})
web_app.run()
