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

parser.add_argument('solve_dir', action='store', type=str)
parser.add_argument('restore_id', action='store', type=int)
parser.add_argument('restore_epoch', action='store', type=int)
parser.add_argument('n_rounds', action='store', type=int)

opts = parser.parse_args()
setattr(opts, 'run_id', None)
setattr(opts, 'n_saves_to_keep', 1)

print(opts)

g = NeuroSAT(opts)
g.restore()

filenames = [opts.solve_dir + "/" + f for f in os.listdir(opts.solve_dir)]
for filename in filenames:
    with open(filename, 'rb') as f:
        problems = pickle.load(f)

    for problem in problems:
        solutions = g.find_solutions(problem)
        for batch, solution in enumerate(solutions):
            print("[%s] %s" % (problem.dimacs[batch], str(solution)))
