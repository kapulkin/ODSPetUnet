import json
import cv2
import os
import random
import numpy as np
from pycocotools import mask as cocoMask

class CocoDataset:
    def __init__(self, annotationFilePath: str, imagesDir: str, masksDir: str, shuffle: bool = True):
        with open(annotationFilePath, 'r') as annotationFile:
            annotations = json.load(annotationFile)

        self.images = annotations["images"]

        self.annotations = {}
        for annotation in annotations["annotations"]:
            id = annotation["image_id"]
            if id in self.annotations:
                imageAnnotations = self.annotations[id]
            else:
                imageAnnotations = []
                self.annotations[id] = imageAnnotations
            imageAnnotations.append(annotation)

        self.imagesDir = imagesDir
        self.masksDir = masksDir
        self.indices = list(range(len(annotations["images"])))
        if shuffle:
            random.shuffle(self.indices)
        self.index = 0
    
    def readBatch(self, batchSize: int):
        batch = []
        while self.index < len(self.indices) and len(batch) < batchSize:
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
                if len(mask.shape) == 3:
                    mask = np.bitwise_or.reduce(mask, 2)

                if commonMask is None:
                    commonMask = mask
                else:
                    commonMask = np.logical_or(commonMask, mask)
            commonMask = commonMask.astype(np.uint8)
            maskName = os.path.splitext(imageName)[0] + ".png"
            fileMask = cv2.imread(os.path.join(self.masksDir, maskName))

            if image is not None and commonMask is not None:
                batch.append([image, commonMask, fileMask])
            self.index += 1
        return batch
