#!/usr/bin/env python

# /********************************************************************
# Filename: bot_v_bot.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Start a web interface for bot play
#

from dlgo.agent.naive_fast import FastRandomBot
from dlgo import mcts
from dlgo.httpfrontend.server import get_web_app

random_agent = FastRandomBot()
MCTS_ROUNDS = 1000
#MCTS_ROUNDS = 100
MCTS_TEMPERATURE = 0.8
mcts_agent = mcts.MCTSAgent( MCTS_ROUNDS, MCTS_TEMPERATURE)
#web_app = get_web_app({'random':random_agent})
web_app = get_web_app({'mcts':mcts_agent})
web_app.run()
