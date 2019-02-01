import os
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")
import pathlib

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="", help="Input file")
    parser.add_argument("--out", default="", help="Output image")

    return parser.parse_args()


def main(cargs):
    histo_file = cargs.file
    out_image = cargs.out
    bins = np.loadtxt(histo_file, dtype=np.double, usecols=(0,1))
    bins[ bins==0 ] = np.nan
    plt.plot(bins[:,0], bins[:,1], '.')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    cargs = parse_args(sys.argv)
    main(cargs)
