from llff.poses.pose_utils import gen_poses
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('scenedir', type=str,
                    help='input scene directory')
args = parser.parse_args()

if __name__=='__main__':
    gen_poses(args.scenedir)