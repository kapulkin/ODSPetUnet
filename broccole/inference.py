import segmentation_models as sm
import argparse
import os

import cv2
import tensorflow as tf
import logging

from broccole.CocoDataset import CocoDataset
from broccole.CocoDatasetBuilder import CocoDatasetBuilder
from broccole.model import makeModel

logger = logging.getLogger(__name__)

def inference(trainDataset: CocoDataset):
    model, preprocess_input = makeModel()

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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    datasetDir = args.datasetDir

    dataset = CocoDatasetBuilder(
        os.path.join(datasetDir, 'annotations/instances_train2017.json'),
        os.path.join(datasetDir, 'train2017'),
    ) \
        .addClasses(classes=[1]).build()
    
    inference(dataset)


if __name__ == '__main__':
    main()