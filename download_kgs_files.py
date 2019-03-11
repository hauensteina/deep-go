#!/usr/bin/env python

# /********************************************************************
# Filename: download_kgs_files.py
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Download archived KGS game records
#

from dlgo.data.index_processor import KGSIndex

index = KGSIndex()
index.download_files()
