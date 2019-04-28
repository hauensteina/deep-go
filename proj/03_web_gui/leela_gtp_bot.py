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

import subprocess
from threading import Thread,Lock,Event
import atexit

import goboard_fast as goboard
from agent_base import Agent
from agent_helpers import is_point_an_eye
from goboard_fast import Move
from gotypes import Point, Player

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

            #------------------------------------
            def wait_for_line( stream, callback):
                while True:
                    line = stream.readline()
                    if line:
                        result_handler( line)
                    else: # probably my process died
                        error_handler()
                        break

            self.thread = Thread( target = wait_for_line,
                                  args = (self.stream, self.result_handler))
            self.thread.daemon = True
            self.thread.start()

    #------------------------------------------------
    def __init__( self, leela_cmdline):
        Agent.__init__( self)
        self.leela_cmdline = leela_cmdline
        self.handler_lock = Lock()
        self.response_event = Event()

        self.leela_proc, self.leela_listener = self._start_leelaproc()
        atexit.register( self._kill_leela)

    #-------------------------
    def _start_leelaproc( self):
        proc = subprocess.Popen( self.leela_cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        listener = LeelaGTPBot.Listener( proc.stdout,
                                         self._result_handler,
                                         self._error_handler)
        return proc, listener

    #-------------------
    def _kill_leela( self):
        if self.leela_proc.pid: os.kill( self.leela_proc.pid, signal.SIGKILL)

    # Parse leela response and trigger event to
    # continue execution.
    #---------------------------------------------
    def _result_handler( self, leela_response):
        BP()
        with self.handler_lock:
            line = leela_response.decode()
            print( '--> %s' % line)
            if line.startswith( '='):
                move = line.split[1]
                _result_handler.result = move
                self.response_event.set()

    # Resurrect a dead Leela
    #---------------------------
    def _error_handler( self):
        with self.handler_lock:
            print( 'Leela died. Resurrecting.')
            self._kill_leela()
            self.leela_proc, self.leela_listener = self._start_leelaproc()
            print( 'Leela resurrected')

    #------------------------------------------
    def select_move( self, game_state, moves):
        p = self.leela_proc
        # Reset the game
        p.stdin.write( b'clear_board'); p.stdin.flush()

        # Make the moves
        color = 'b'
        for move in moves:
            p.stdin.write( b'play %s %s' % (color, move)); p.stdin.flush()
            color = 'b' if color == 'w' else 'w'

        # Ask for new move
        p.stdin.write( ('genmove %s' % color).encode('utf8')); p.stdin.flush()
        # Hang until the move comes back
        self.response_event.clear()
        success = self.response_event.wait( MOVE_TIMEOUT)
        if not success: # I guess leela died
            self._error_handler()
            return ''
        return self._result_handler.result

    # Turn an idx 0..360 into a move
    #---------------------------------
    def _idx2move( self, idx):
        point = self.encoder.decode_point_index( idx)
        return goboard.Move.play( point)
