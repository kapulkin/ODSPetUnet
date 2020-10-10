import cv2
import os
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class SegmentationDataset:
    def __init__(self, datasetDir: str, shuffle: bool = True):
        self.datasetDir = datasetDir
        filesCount = len([name for name in os.listdir(datasetDir) if os.path.isfile(os.path.join(datasetDir, name))])
        self.trainPairsCount = filesCount // 2
        self.indices = list(range(self.trainPairsCount))
        if shuffle:
            random.shuffle(self.indices)

        self.index = 0

    def __len__(self):
        return self.trainPairsCount

    def reset(self):
        self.index = 0

    def readBatch(self, batchSize: int = None):
        if batchSize is None:
            batchSize = len(self)

        imagesBatch = []
        masksBatch = []
        while self.index < len(self) and len(imagesBatch) < batchSize:
            datasetIndex = self.indices[self.index]
            image = cv2.imread(os.path.join(self.datasetDir, 'image{}.jpg'.format(datasetIndex)))
            mask = cv2.imread(os.path.join(self.datasetDir, 'mask{}.png'.format(datasetIndex)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            if image is not None and mask is not None:
                imagesBatch.append(image)
                masksBatch.append(mask)

            self.index += 1

        imagesBatchStack = np.stack(imagesBatch)
        masksBatchStack = np.stack(masksBatch)

        del imagesBatch
        del masksBatch

        return (imagesBatchStack, masksBatchStack)


