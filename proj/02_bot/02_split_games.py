#!/usr/bin/env python

# /********************************************************************
# Filename: split_games.py
# Author: AHN
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Divide sgf games in a folder into train, valid, test sets
#

from pdb import set_trace as BP
import os,sys,re
import argparse

# Look for modules further up
SCRIPTPATH = os.path.dirname(os.path.realpath( __file__))
sys.path.append( re.sub(r'/proj/.*',r'/', SCRIPTPATH))

import pylib.ahnutil as ut

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Divide sgf files in a folder into train, valid, test sets
    Synopsis:
      %s --folder <folder> --trainpct <n> --validpct <n> [--ngames <n>]
    Description:
      Splits the sgf files in folder into train, valid, and test files.
      The --ngames option limits the total number of total games going to
      the train, valid, test folders.
    Example:
      %s --folder ../sgf_all --trainpct 80 --validpct 20 --ngames 100
      The remaining 10pct will be test data
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
    parser.add_argument( "--folder",      required=True)
    parser.add_argument( "--trainpct",    required=True,  type=int)
    parser.add_argument( "--validpct",    required=True,  type=int)
    parser.add_argument( "--ngames",      required=False, type=int)
    args = parser.parse_args()
    #np.random.seed(0) # Make things reproducible
    ut.split_files( args.folder, args.trainpct, args.validpct, nfiles=args.ngames)

if __name__ == '__main__':
    main()
