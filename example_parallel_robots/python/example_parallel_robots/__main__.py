from argparse import ArgumentParser

from .robot_options import ROBOTS
from .loader_tools import load

ROBOTS = sorted(ROBOTS.keys())

parser = ArgumentParser(description=load.__doc__)
parser.add_argument("robot", nargs="?", default=ROBOTS[0], choices=ROBOTS)
parser.add_argument("--options", nargs="?", default=None)

args = parser.parse_args()

load(args.robot, args.options)