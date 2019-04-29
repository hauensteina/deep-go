#!/usr/bin/env python

# /*********************************
# Filename: leela_gtp_bot.py
# Creation Date: Apr, 2019
# Author: AHN
# **********************************/
#
# A wrapper bot to use several leela processes to find the next move.
# For use from a website were people can play leela.
#

from pdb import set_trace as BP
import os, sys, re
import numpy as np
import signal
import time

import subprocess
from threading import Thread,Lock,Event
import atexit

import goboard_fast as goboard
from agent_base import Agent
from agent_helpers import is_point_an_eye
from goboard_fast import Move
from gotypes import Point, Player
from go_utils import point_from_coords

g_response = None

MOVE_TIMEOUT = 10 # seconds
#===========================
class LeelaGTPBot( Agent):
    # Listen on a stream in a separate thread until
    # a line comes in. Process line in a callback.
    #=================================================
    class Listener:
        #------------------------------------------------------------
        def __init__( self, stream, result_handler, error_handler):
            self.stream = stream
            self.result_handler = result_handler

            #--------------------------------------
            def wait_for_line( stream, callback):
                global g_result
                while True:
                    line = stream.readline().decode()
                    if line:
                        result_handler( line)
                    else: # probably my process died
                        error_handler()
                        break

            self.thread = Thread( target = wait_for_line,
                                  args = (self.stream, self.result_handler))
            self.thread.daemon = True
            self.thread.start()

    #--------------------------------------
    def __init__( self, leela_cmdline):
        Agent.__init__( self)
        self.leela_cmdline = leela_cmdline
        self.handler_lock = Lock()
        self.response_event = Event()

        self.leela_proc, self.leela_listener = self._start_leelaproc()
        atexit.register( self._kill_leela)

    #------------------------------
    def _start_leelaproc( self):
        proc = subprocess.Popen( self.leela_cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        listener = LeelaGTPBot.Listener( proc.stdout,
                                         self._result_handler,
                                         self._error_handler)
        return proc, listener

    #-------------------------
    def _kill_leela( self):
        if self.leela_proc.pid: os.kill( self.leela_proc.pid, signal.SIGKILL)

    # Parse leela response and trigger event to
    # continue execution.
    #---------------------------------------------
    def _result_handler( self, leela_response):
        global g_response
        #with self.handler_lock:
        line = leela_response
        print( '<-- ' + line)
        if '=' in line:
            resp = line.split('=')[1].strip()
            print( '<== ' + resp)
            g_response = self._resp2Move( resp)
            self.response_event.set()

    # Resurrect a dead Leela
    #---------------------------
    def _error_handler( self):
        #with self.handler_lock:
            print( 'Leela died. Resurrecting.')
            self._kill_leela()
            self.leela_proc, self.leela_listener = self._start_leelaproc()
            print( 'Leela resurrected')

    # Convert Leela response string to a Move we understand
    #------------------------------
    def _resp2Move( self, resp):
        res = None
        if 'pass' in resp:
            res = Move.pass_turn()
        elif 'resign' in resp:
            res = Move.resign()
        elif len(resp.strip()) == 2:
            p = point_from_coords( resp)
            res = Move.play( p)
        return res

    # Send a command to leela
    #--------------------------
    def _leelaCmd( self, cmdstr):
        cmdstr += '\n'
        p = self.leela_proc
        p.stdin.write( cmdstr.encode('utf8'))
        p.stdin.flush()
        print( '--> ' + cmdstr)

    #-------------------------------------------
    def select_move( self, game_state, moves):
        global g_response
        res = None
        p = self.leela_proc
        # Reset the game
        self._leelaCmd( 'clear_board')

        # Make the moves
        color = 'b'
        # for move in moves:
        #     p.stdin.write( b'play %s %s' % (color, move)); p.stdin.flush()
        #     color = 'b' if color == 'w' else 'w'

        # Ask for new move
        self._leelaCmd( 'genmove ' + color)
        # # Hang until the move comes back
        # self.response_event.clear()
        # success = self.response_event.wait( MOVE_TIMEOUT)
        # if not success: # I guess leela died
        #     self._error_handler()
        #     return ''
        time.sleep(2)
        print( 'reponse: ' + str(g_response))
        if g_response:
            res = g_response
        g_response = None
        return res

    # Turn an idx 0..360 into a move
    #---------------------------------
    def _idx2move( self, idx):
        point = self.encoder.decode_point_index( idx)
        return goboard.Move.play( point)
