#!/usr/bin/env python

# /********************************************************************
# Filename: helpers.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Go agent helper funcs
#

from pdb import set_trace as BP
from dlgo.gotypes import Point

#-------------------------------------------
def is_point_an_eye( board, point, color):
    BP()
    if board.get( point) is not None:
        return False
    for neighbor in point.neighbors():
        if board.is_on_grid( neighbor):
            neighbor_color = board.get( neighbor)
            if neighbor_color != color:
                return False

    friendly_corners = 0
    off_board_corners = 0
    corners = [
        Point( point.row - 1, point.col - 1),
        Point( point.row - 1, point.col + 1),
        Point( point.row + 1, point.col - 1),
        Point( point.row + 1, point.col + 1)
    ]
    for corner in corners:
        if board.is_on_grid( corner):
            corner_color = board.get( corner)
            if corner_color is None or corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1

    BP()
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4
    return friendly_corners >= 3
