import cv2
import os
import numpy as np

import logging

logger = logging.getLogger(__name__)

class SegmentationDataset:
    def __init__(self, datasetDir: str):
        self.datasetDir = datasetDir
        filesCount = len([name for name in os.listdir(datasetDir) if os.path.isfile(os.path.join(datasetDir, name))])
        self.trainPairsCount = filesCount // 2
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
            image = cv2.imread(os.path.join(self.datasetDir, 'image{}.jpg'.format(i)))
            mask = cv2.imread(os.path.join(self.datasetDir, 'mask{}.png'.format(i)))
            if image is not None and mask is not None:
                imagesBatch.append(image)
                masksBatch.append(mask)

            self.index += 1

        imagesBatch = np.stack(imagesBatch)
        masksBatch = np.stack(masksBatch)

        return (imagesBatch, masksBatch)


