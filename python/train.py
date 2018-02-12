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

parser.add_argument('train_dir', action='store', type=str, help='Directory with training data')
parser.add_argument('--run_id', action='store', dest='run_id', type=int, default=None)
parser.add_argument('--restore_id', action='store', dest='restore_id', type=int, default=None)
parser.add_argument('--restore_epoch', action='store', dest='restore_epoch', type=int, default=None)
parser.add_argument('--n_epochs', action='store', dest='n_epochs', type=int, default=100000, help='Number of epochs through data')
parser.add_argument('--n_saves_to_keep', action='store', dest='n_saves_to_keep', type=int, default=4, help='Number of saved models to keep')

opts = parser.parse_args()

setattr(opts, 'commit', subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())
setattr(opts, 'hostname', subprocess.check_output(['hostname']).strip())

if opts.run_id is None: opts.run_id = random.randrange(sys.maxsize)

print(opts)

if not os.path.exists("snapshots/"):
    os.mkdir("snapshots")

g = NeuroSAT(opts)

for epoch in range(opts.n_epochs):
    result = g.train_epoch(epoch)
    (efilename, etrain_cost, etrain_mat, lr, etime) = result
    print("[%d] %.4f (%.2f, %.2f, %.2f, %.2f) [%ds]" % (epoch, etrain_cost, etrain_mat.ff, etrain_mat.ft, etrain_mat.tf, etrain_mat.tt, etime))
