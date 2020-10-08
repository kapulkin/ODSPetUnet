import segmentation_models as sm

import argparse
import os
import cv2
import tensorflow as tf
import numpy as np
import logging

from broccole.SegmentationDataset import SegmentationDataset
from broccole.model import makeModel
from broccole.logUtils import init_logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)    
    parser.add_argument('--batchSize', help='batch size', type=int, default=1)
    parser.add_argument('--epochs', help='epochs count', type=int, default=1)
    args = parser.parse_args()
    return args

def train(
    humanDataset: SegmentationDataset,
    nonHumanDataset: SegmentationDataset,
    valHumanDataset: SegmentationDataset,
    valNonHumanDataset: SegmentationDataset,
    trainingDir: str,
    batchSize: int = 1,
    epochs: int = 1
):
    model, preprocess_input = makeModel()

    validationPacketSize = 32 * 32
    x_val_h, y_val_h = valHumanDataset.readBatch(validationPacketSize)
    x_val_nh, y_val_nh = valNonHumanDataset.readBatch(validationPacketSize)
    x_val = np.vstack([x_val_h, x_val_nh])
    y_val = np.vstack([y_val_h, y_val_nh])
    x_val = preprocess_input(x_val)

    checkPointPath = os.path.join(trainingDir, 'u-net-resnet18.chpt')
    checkPointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkPointPath,
                                                save_weights_only=True,
                                                verbose=1)

    packetSize = 16 * 16
    nonHumanPacketSize = (packetSize * len(nonHumanDataset)) // len(humanDataset)

    for epoch in range(epochs):
        logger.info('epoch %d', epoch)
        humanDataset.reset()
        nonHumanDataset.reset()

        packets = len(humanDataset) // packetSize
        for _ in range(packets - 1):
            x_train_h, y_train_h = humanDataset.readBatch(packetSize)
            x_train_nh, y_train_nh = nonHumanDataset.readBatch(nonHumanPacketSize)
            x_train = np.vstack([x_train_h, x_train_nh])
            y_train = np.vstack([y_train_h, y_train_nh])
            x_train = preprocess_input(x_train)

            model.fit(
                x=x_train,
                y=y_train,
                batch_size=batchSize,
                epochs=1,
                validation_data=(x_val, y_val),
                callbacks=[checkPointCallback],
            )

        x_train_h, y_train_h = humanDataset.readBatch(packetSize)
        x_train_nh, y_train_nh = nonHumanDataset.readBatch(nonHumanPacketSize)
        x_train = np.vstack([x_train_h, x_train_nh])

        model.fit(
            x=x_train,
            y=y_train,
            batch_size=batchSize,
            epochs=1,
            validation_data=(x_val, y_val),
            callbacks=[checkPointCallback],
        )

    modelPath = os.path.join(trainingDir, 'u-net-resnet18.tfmodel')
    model.save(modelPath)

def main():
    init_logging()

    args = parse_args()
    datasetDir = args.datasetDir

    humanDataset = SegmentationDataset(os.path.join(datasetDir, 'human'))
    nonHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'nonHuman'))
    valHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'valHuman'))
    valNonHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'valNonHuman'))
    train(humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset, datasetDir, args.batchSize, args.epochs)

if __name__ == '__main__':
        main()