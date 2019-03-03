#!/usr/bin/env python

# /********************************************************************
# Filename: bot_v_bot.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Let two bots play against each other. ASCII output to screen.
#

from dlgo import gotypes
from dlgo.agent import naive
from dlgo import goboard_slow as goboard
from dlgo.utils import print_board, print_move
from dlgo.utils import print_board, print_move
import time

#--------------
def main():
    board_size = 9
    game = goboard.GameState.new_game( board_size)
    bots = {
        gotypes.Player.black: naive.RandomBot(),
        gotypes.Player.white: naive.RandomBot()
    }
    while not game.is_over():
        time.sleep(0.3)
        print( chr(27) + '[2J') # clear screen
        print_board( game.board)
        bot_move = bots[ game.next_player].select_move( game)
        print_move( game.next_player, bot_move)
        game = game.apply_move( bot_move)

if __name__ == '__main__':
    main()
