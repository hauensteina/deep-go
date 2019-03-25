#!/usr/bin/env python

# /********************************************************************
# Filename: scoring.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Count a go position. All dead stones must have been removed already.
#

from __future__ import absolute_import
from collections import namedtuple
from dlgo.gotypes import Player, Point
import numpy as np

#===================
class Territory:

    #----------------------------------------
    def __init__( self, territory_map):
        self.n_intersections = len( territory_map)
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []
        self.black_points = []
        self.white_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
                self.black_points.append( point)
            elif status == Player.white:
                self.num_white_stones += 1
                self.white_points.append( point)
            elif status == 'territory_b':
                self.num_black_territory += 1
                self.black_points.append( point)
            elif status == 'territory_w':
                self.num_white_territory += 1
                self.white_points.append( point)
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append( point)

    # Turn yourself into a 1D np array of -1,0,1 for b,e,w .
    # This is the label we use to train a territory estimator.
    #------------------------------------------------------------
    def encode( self):
        res = np.zeros( self.n_intersections)
        size = int( np.sqrt( self.n_intersections))
        for p in self.black_points:
            idx = (p.row - 1) * size + p.col -1
            res[idx] = -1
        for p in self.white_points:
            idx = (p.row - 1) * size + p.col -1
            res[idx] = 1
        return res


#=========================================================
class GameResult( namedtuple( 'GameResult', 'b w komi')):
    @property
    def winner(self):
        if self.b > self.w + self.komi:
            return Player.black
        return Player.white

    @property
    def winning_margin(self):
        w = self.w + self.komi
        return abs(self.b - w)

    def __str__(self):
        w = self.w + self.komi
        if self.b > w:
            return 'B+%.1f' % (self.b - w,)
        return 'W+%.1f' % (w - self.b,)

# Map a board into territory and dame.
# Any points that are completely surrounded by a single color are
# counted as territory; it makes no attempt to identify even
# trivially dead groups.
#-------------------------------
def evaluate_territory( board):
    status = {}
    for r in range( 1, board.num_rows + 1):
        for c in range( 1, board.num_cols + 1):
            p = Point(row=r, col=c)
            if p in status:  # <1>
                continue
            stone = board.get(p)
            if stone is not None:  # <2>
                status[p] = board.get(p)
            else:
                group, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:  # <3>
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.black else 'w'
                    fill_with = 'territory_' + stone_str
                else:
                    fill_with = 'dame'  # <4>
                for pos in group:
                    status[pos] = fill_with
    return Territory( status)

# <1> Skip the point, if you already visited this as part of a different group.
# <2> If the point is a stone, add it as status.
# <3> If a point is completely surrounded by black or white stones, count it as territory.
# <4> Otherwise the point has to be a neutral point, so we add it to dame.
# end::scoring_evaluate_territory[]


# Find the contiguous section of a board containing a point. Also
# identify all the boundary points.
# This is like finding strings, but also for empty intersections.
#------------------------------------------------------------------
def _collect_region( start_pos, board, visited=None):
    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()
    all_points = [start_pos]
    all_borders = set()
    visited[start_pos] = True
    here = board.get( start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta_r, delta_c in deltas:
        next_p = Point( row=start_pos.row + delta_r, col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get( next_p)
        if neighbor == here:
            points, borders = _collect_region( next_p, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add( neighbor)
    return all_points, all_borders

# Entry point.
#--------------------------------------
def compute_game_result( game_state):
    territory = evaluate_territory( game_state.board)
    return (territory,
            GameResult(
                territory.num_black_territory + territory.num_black_stones,
                territory.num_white_territory + territory.num_white_stones,
                komi=7.5)
    )
