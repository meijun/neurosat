import math
import numpy as np
import random
import datetime
import subprocess
import pickle
import sys
import os
import argparse
from options import add_neurosat_options
from neurosat import NeuroSAT

parser = argparse.ArgumentParser()
add_neurosat_options(parser)

parser.add_argument('valid_dir', action='store', type=str, help='Directory with directories of validation data')
parser.add_argument('restore_id', action='store', type=int)
parser.add_argument('restore_epoch', action='store', type=int)
parser.add_argument('n_rounds', action='store', type=int)

opts = parser.parse_args()
setattr(opts, 'run_id', None)
setattr(opts, 'n_saves_to_keep', 1)

print(opts)

g = NeuroSAT(opts)
g.restore()

results = g.validate(opts.valid_dir)
for (valid_filename, evalid_cost, evalid_mat) in results:
    print("%s %.4f (%.2f, %.2f, %.2f, %.2f)" % (valid_filename, evalid_cost, evalid_mat.ff, evalid_mat.ft, evalid_mat.tf, evalid_mat.tt))
