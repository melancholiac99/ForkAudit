import argparse
from pathlib import Path
from .common import DEFAULT_RUN, setup

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['clean','screen','split']);parser.add_argument('--run',type=Path,default=DEFAULT_RUN);args=parser.parse_args()
    setup(args.run)
    from . import cpu
    getattr(cpu,args.command)(args.run)

if __name__=='__main__':main()
