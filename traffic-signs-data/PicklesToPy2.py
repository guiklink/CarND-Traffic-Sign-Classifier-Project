#!/usr/bin/env python3

import pickle
import os


for file in os.listdir():
	with open(file, "rb") as f:
	    w = pickle.load(f)
	    pickle.dump(w, open("{}_py2.pickle".format(file.split('.')[0]),"wb"), protocol=2)