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
from dlgo.httpfrontend.server import get_web_app

random_agent = FastRandomBot()
web_app = get_web_app({'random':random_agent})
web_app.run()
