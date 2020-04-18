import argparse
import logging

from .downscaling import downscale
from .splitting import split
from .tiling import tile
from .learning import train
from .evaluation import test, final_test
from .grid_searching import grid_search
from .visualization import visualize

from .argument_parser import ArgumentParser


def config():
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="DeepSlide",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(parser_class=ArgumentParser)
    downscale.add_parser(subparsers)
    split.add_parser(subparsers)
    tile.add_parser(subparsers)
    train.add_parser(subparsers)
    test.add_parser(subparsers)
    grid_search.add_parser(subparsers)
    visualize.add_parser(subparsers)
    final_test.add_parser(subparsers)
    args = parser.parse_args()

    args.func(args)
