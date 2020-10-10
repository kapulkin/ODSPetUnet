import os
import argparse
import cv2

from broccole.CocoDataset import CocoDataset
from broccole.CocoDatasetBuilder import CocoDatasetBuilder
from broccole.logUtils import init_logging

def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)    
    args = parser.parse_args()
    return args

def main():
    init_logging('saveDataset.log')

    args = parse_args()
    datasetDir = args.datasetDir
    dataset = CocoDatasetBuilder(
        os.path.join(datasetDir, 'annotations/instances_train2017.json'),
        os.path.join(datasetDir, 'train2017'),
    ) \
        .addClasses(classes=[1]).build()

    humanDatasetDir = os.path.join(datasetDir, 'human')
    if not os.path.exists(humanDatasetDir):
        os.makedirs(humanDatasetDir)

    CocoDataset.save(dataset, humanDatasetDir)

if __name__ == '__main__':
    main()