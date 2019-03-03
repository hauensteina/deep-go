#!/usr/bin/env python

# /********************************************************************
# Filename: bot_v_bot.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Start a web interface to play the random bot
#

from dlgo.agent.naive import RandomBot
from dlgo.httpfrontend.server import get_web_app

random_agent = RandomBot()
web_app = get_web_app({'random':random_agent})
web_app.run()
