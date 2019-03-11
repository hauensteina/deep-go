#!/usr/bin/env python

# /********************************************************************
# Filename: encode_some_kgs_files.py
# Creation Date: Mar, 2019
# **********************************************************************/
#
# Encode some KGS games for machine learning
#

from dlgo.data.processor import GoDataProcessor

processor = GoDataProcessor()
features, labels = processor.load_go_data( 'train', 100)
