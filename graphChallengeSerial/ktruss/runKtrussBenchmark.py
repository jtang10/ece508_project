import numpy as np
import scipy as sp
from ktruss import ktruss
import os, sys, argparse

#Use the pandas package if available
#import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_filename", nargs="?", action="store", type=str, default="../data/ktruss_example.tsv")
args = parser.parse_args()

inc_mtx_file = args.input_filename

if not os.path.isfile(inc_mtx_file):
	print("File doesn't exist: '{}'!".format(inc_mtx_file))
	sys.exit(1)

E = ktruss(inc_mtx_file,3)
print("ktruss result: \n{}".format(E))


###################################################
# Graph Challenge benchmark
# Developer: Dr. Vijay Gadepally (vijayg@mit.edu)
# MIT

# Modified by Jingning. Fixed issues for Python3 
# compatibility and np dtype issue
###################################################

###################################################
# The ktruss_example.tsv stores the coordinates of
# the indidence matrix rather than the matrix
# itself. The actual incidence matrix is 
#                0  1  1  0
#                1  1  0  0
#                1  0  0  1
#                0  0  1  1
#                1  0  1  0
# where rows mean edges and columns mean verticies
# The graph looks like this (vertax is 1-indexed)
#                  1-----2
#                  | \   |
#                  |  \  |
#                  |   \ |
#                  |    \|
#                  4-----3        
####################################################           

