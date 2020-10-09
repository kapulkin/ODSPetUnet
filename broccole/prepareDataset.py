import argparse
import os

from broccole.CocoDataset import CocoDataset
from broccole.CocoDatasetBuilder import CocoDatasetBuilder
from broccole.logUtils import init_logging

def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    datasetDir = args.datasetDir

    init_logging()

    humanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017')).addClasses([1]).build()
    CocoDataset.save(humanDataset, os.path.join(datasetDir, 'human'))

    nonHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017')).selectAll().filterNonClasses([1]).build(shuffle=True)
    CocoDataset.save(nonHumanDataset, os.path.join(datasetDir, 'nonHuman'))

    valHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017')).addClasses([1]).build()
    CocoDataset.save(valHumanDataset, os.path.join(datasetDir, 'valHuman'))

    valNonHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017')).selectAll().filterNonClasses([1]).build(shuffle=True)
    CocoDataset.save(valNonHumanDataset, os.path.join(datasetDir, 'valNonHuman'))

if __name__ == '__main__':
    main()
