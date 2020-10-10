import segmentation_models as sm

import argparse
import os
import cv2
import tensorflow as tf
import numpy as np
import logging

from broccole.CocoDatasetBuilder import CocoDatasetBuilder
from broccole.SegmentationDataset import SegmentationDataset
from broccole.model import makeModel
from broccole.logUtils import init_logging, usedMemory

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)    
    parser.add_argument('--trainingDir', help='path to directory to save models', type=str)    
    parser.add_argument('--datasetType', help='prepared, coco or coco on kaggle', type=str)
    parser.add_argument('--batchSize', help='batch size', type=int, default=1)
    parser.add_argument('--epochs', help='epochs count', type=int, default=1)
    parser.add_argument('--checkpointFilePath', help='path to checkpoint', type=str)
    args = parser.parse_args()
    return args

def train(
    humanDataset: SegmentationDataset,
    nonHumanDataset: SegmentationDataset,
    valHumanDataset: SegmentationDataset,
    valNonHumanDataset: SegmentationDataset,
    trainingDir: str,
    checkpointFilePath: str,
    batchSize: int = 1,
    epochs: int = 1
):
    model, preprocess_input = makeModel()

    if checkpointFilePath is not None:
        model.load_weights(checkpointFilePath)
        logger.info('model weights from %s are loaded', checkpointFilePath)


    validationPacketSize = 16 * 16
    x_val_h, y_val_h = valHumanDataset.readBatch(validationPacketSize)
    x_val_nh, y_val_nh = valNonHumanDataset.readBatch(validationPacketSize)
    x_val = np.concatenate((x_val_h, x_val_nh))
    y_val = np.concatenate((y_val_h, y_val_nh))
    x_val = preprocess_input(x_val)

    checkPointPath = os.path.join(trainingDir, 'u-net-resnet18.chpt')
    checkPointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkPointPath,
                                                save_weights_only=True,
                                                verbose=1)

    packetSize = 8 * 8
    nonHumanPacketSize = max((packetSize * len(nonHumanDataset)) // len(humanDataset), 1)

    for epoch in range(epochs):
        logger.info('epoch %d', epoch)
        humanDataset.reset()
        nonHumanDataset.reset()

        checkPointPath = os.path.join(trainingDir, 'u-net-resnet18_epoch_{}.chpt'.format(epoch))

        x_train = None
        y_train = None
        try:
            packets = len(humanDataset) // packetSize
            for _ in range(packets - 1):
                logger.debug('reading batch, memory used %f', usedMemory())
                x_train_h, y_train_h = humanDataset.readBatch(packetSize)
                logger.debug('reading human batch, memory used %f', usedMemory())
                x_train_nh, y_train_nh = nonHumanDataset.readBatch(nonHumanPacketSize)
                logger.debug('reading nonHuman batch, memory used %f', usedMemory())
                del x_train
                del y_train
                x_train = np.concatenate((x_train_h, x_train_nh))
                y_train = np.concatenate((y_train_h, y_train_nh))
                del x_train_h
                del x_train_nh
                del y_train_h
                del y_train_nh
                logger.debug('concatenate batches, memory used %f', usedMemory())            
                x_train = preprocess_input(x_train)
                logger.debug('preprocess x_train, memory used %f', usedMemory())

                if ((humanDataset.index + nonHumanDataset.index) % 1000) < (packetSize + nonHumanPacketSize):
                    callbacks = [checkPointCallback]
                else:
                    callbacks = []

                logger.debug('start train, memory used %f', usedMemory())
                model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batchSize,
                    epochs=1,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks,
                )
                logger.debug('trained on %d samples, memory used %f', humanDataset.index + nonHumanDataset.index, usedMemory())

            x_train_h, y_train_h = humanDataset.readBatch(packetSize)
            x_train_nh, y_train_nh = nonHumanDataset.readBatch(nonHumanPacketSize)
            del x_train
            del y_train
            x_train = np.concatenate((x_train_h, x_train_nh))
            y_train = np.concatenate((y_train_h, y_train_nh))
            del x_train_h
            del x_train_nh
            del y_train_h
            del y_train_nh
            x_train = preprocess_input(x_train)

            history = model.fit(
                x=x_train,
                y=y_train,
                batch_size=batchSize,
                epochs=1,
                validation_data=(x_val, y_val),
                callbacks=[checkPointCallback],
            )
            logger.info('epoch trained %s', str(history))
        except Exception as e:
            logger.error('Exception %s', str(e))

    modelPath = os.path.join(trainingDir, 'u-net-resnet18.tfmodel')
    model.save(modelPath)
    logger.info('model saved')

def openSegmentationDatasets(datasetDir: str):
    humanDataset = SegmentationDataset(os.path.join(datasetDir, 'human'))
    nonHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'nonHuman'))
    valHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'valHuman'))
    valNonHumanDataset = SegmentationDataset(os.path.join(datasetDir, 'valNonHuman'))
    return humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset

def openCocoDatasets(datasetDir: str):
    humanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_train2017.json'), os.path.join(datasetDir, 'train2017')).addClasses([1]).build()
    nonHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_train2017.json'), os.path.join(datasetDir, 'train2017')).selectAll().filterNonClasses([1]).build(shuffle=True)
    valHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017')).addClasses([1]).build()
    valNonHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017')).selectAll().filterNonClasses([1]).build(shuffle=True)
    return humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset

def openKaggleCocoDatasets(datasetDir: str):
    humanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations_trainval2017/annotations/instances_train2017.json'), os.path.join(datasetDir, 'train2017/train2017')).addClasses([1]).build()
    nonHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations_trainval2017/annotations/instances_train2017.json'), os.path.join(datasetDir, 'train2017/train2017')).selectAll().filterNonClasses([1]).build(shuffle=True)
    valHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations_trainval2017/annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017/val2017')).addClasses([1]).build()
    valNonHumanDataset = CocoDatasetBuilder(os.path.join(datasetDir, 'annotations_trainval2017/annotations/instances_val2017.json'), os.path.join(datasetDir, 'val2017/val2017')).selectAll().filterNonClasses([1]).build(shuffle=True)
    return humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset

def main():
    init_logging('training')

    args = parse_args()
    datasetDir = args.datasetDir
    datasetType = args.datasetType
    trainingDir = args.trainingDir if args.trainingDir is not None else datasetDir
    checkpointFilePath = args.checkpointFilePath

    if datasetType == 'prepared':
        humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset = openSegmentationDatasets(datasetDir)
    elif datasetType == 'coco':
        humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset = openCocoDatasets(datasetDir)
    elif datasetType == 'kaggle':
        humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset = openKaggleCocoDatasets(datasetDir)
    train(humanDataset, nonHumanDataset, valHumanDataset, valNonHumanDataset, trainingDir, checkpointFilePath, args.batchSize, args.epochs)

if __name__ == '__main__':
        main()