import segmentation_models as sm
import argparse
import os

import cv2
import tensorflow as tf
import logging

from broccole.CocoDataset import CocoDataset
from broccole.CocoDatasetBuilder import CocoDatasetBuilder
from broccole.SegmentationDataset import SegmentationDataset
from broccole.model import makeModel

logger = logging.getLogger(__name__)

def inference(trainDataset: CocoDataset, checkpointFilePath: str):
    model, preprocess_input = makeModel()

    if checkpointFilePath is not None:
        model.load_weights(checkpointFilePath)
        logger.info('model weights from %s are loaded', checkpointFilePath)

    x_train, y_train = trainDataset.readBatch(1)
    x_train = preprocess_input(x_train)

    masks = model.predict(x_train)
    score = model.evaluate(x_train, y_train)

    logger.info("test loss, test acc: %s", score)
    for i in range(masks.shape[0]):
        image = x_train[i]
        correctMask = y_train[i]
        mask = masks[i]
        cv2.imshow('image', image)
        cv2.imshow('mask', mask * 255)
        cv2.imshow('correctMask', correctMask * 255)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)
    parser.add_argument('--checkpointFilePath', help='path to checkpoint', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    datasetDir = args.datasetDir
    checkpointFilePath = args.checkpointFilePath

    dataset = SegmentationDataset(datasetDir)

    inference(dataset, checkpointFilePath)


if __name__ == '__main__':
    main()