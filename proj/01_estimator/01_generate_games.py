#!/usr/bin/env python

# /********************************************************************
# Filename: generate_games.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Use leela to generate fully played out games for scoring.
#

from pdb import set_trace as BP
import os,sys
import subprocess
import argparse

ROOT = '~/deep-go'
CMD = ("cd %s/leela-zero/build; validation/validation -g 1 -k sgf "
      "-n elfv1 -o '-g -p 1 --noponder -t 1 -d -r 0 -w' " +
      "-n elfv1 -o '-g -p 1 --noponder -t 1 -d -r 0 -w' ") % ROOT
# validation modified to generate 100 games per call
GAMES_PER_CALL = 100

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s -- Use leela to generate fully played out games for scoring
    Synopsis:
      %s --ngames <N> --outfolder <folder>
    Example:
      %s --ngames 100 --outfolder sgf
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--ngames", required=True, type=int)
    parser.add_argument( "--outfolder", required=True)
    args = parser.parse_args()

    os.makedirs( args.outfolder)

    ncalls = int( args.ngames / GAMES_PER_CALL)
    for i in range( ncalls):
        print( 'Generating games %d to %d ...' % (i * GAMES_PER_CALL, (i+1) * GAMES_PER_CALL))
        subprocess.check_output( CMD, shell=True)
        subprocess.check_output( 'mv %s/leela-zero/build/sgf/* %s' % (ROOT, args.outfolder), shell=True)

if __name__ == '__main__':
    main()
