import json
import cv2
import os
import random
import numpy as np
from pycocotools import mask as cocoMask
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class CocoDataset:
    def __init__(self, annotations: Dict, images, imagesDir: str, shuffle: bool = True):
        self.annotations = annotations
        self.images = images
        self.imagesDir = imagesDir

        self.indices = list(range(len(self.images)))
        if shuffle:
            random.shuffle(self.indices)
        self.index = 0

    def __len__(self):
        return len(self.images)

    def reset(self):
        self.index = 0

    def readBatch(self, batchSize: int = None):
        if batchSize is None:
            batchSize = len(self)
        imagesBatch = []
        masksBatch = []
        while self.index < len(self.indices) and len(imagesBatch) < batchSize:
            datasetIndex = self.indices[self.index]
            imageInfo = self.images[datasetIndex]
            imageName = imageInfo["file_name"]
            image = cv2.imread(os.path.join(self.imagesDir, imageName))
            annotations = self.annotations[imageInfo["id"]]
            commonMask = None
            for annotation in annotations:
                segmentation = annotation["segmentation"]
                isRle = annotation["iscrowd"]
                width = imageInfo["width"]
                height = imageInfo["height"]
                rle = cocoMask.frPyObjects(segmentation, height, width)
                mask = cocoMask.decode(rle)
                del rle
                if len(mask.shape) == 3:
                    mask = np.bitwise_or.reduce(mask, 2)

                if commonMask is None:
                    commonMask = mask
                else:
                    commonMask = np.logical_or(commonMask, mask)
            commonMask = commonMask.astype(np.float32)

            if image is not None and commonMask is not None:
                if image.shape[0] != 480 or image.shape[1] != 640:
                    image = cv2.resize(image, (640, 480))
                if commonMask.shape[0] != 480 or commonMask.shape[1] != 640:
                    commonMask = cv2.resize(commonMask, (640, 480))

                imagesBatch.append(image)
                masksBatch.append(commonMask)
            self.index += 1
        
        if len(imagesBatch) == 0 or len(masksBatch) == 0:
            return None, None
        imagesBatchStack = np.stack(imagesBatch)
        masksBatchStack = np.stack(masksBatch)

        del imagesBatch
        del masksBatch

        return (imagesBatchStack, masksBatchStack)

    @staticmethod
    def save(dataset, datasetDir: str, maxCount: int = None, startIndex: int = None):
        if not os.path.exists(datasetDir):
            os.makedirs(datasetDir)
        
        packetSize = 16
        size = min(len(dataset), maxCount) if maxCount is not None else len(dataset)
        packets = size // packetSize

        dataset.reset()
        if startIndex is not None:
            dataset.index = startIndex
        i = 0
        for _ in range(packets):
            images, masks = dataset.readBatch(packetSize)
            if images is None or masks is None:
                continue
            for j in range(images.shape[0]):
                cv2.imwrite(os.path.join(datasetDir, 'image{}.jpg'.format(i)), images[j])
                cv2.imwrite(os.path.join(datasetDir, 'mask{}.png'.format(i)), masks[j])
                i += 1
            
            logger.debug("%d traing pairs (image, mask) saved", i)

        images, masks = dataset.readBatch(size - i)
        images, masks = dataset.readBatch(packetSize)
        if not (images is None or masks is None):
            for j in range(images.shape[0]):
                cv2.imwrite(os.path.join(datasetDir, 'image{}.jpg'.format(i)), images[j])
                cv2.imwrite(os.path.join(datasetDir, 'mask{}.png'.format(i)), masks[j])
                i += 1

        logger.debug("%d traing pairs (image, mask) saved", i)

