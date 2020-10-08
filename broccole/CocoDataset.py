import json
import cv2
import os
import random
import numpy as np
import tensorflow as tf
from pycocotools import mask as cocoMask

class CocoDataset:
    def __init__(self, annotationFilePath: str, imagesDir: str, masksDir: str, classes = [], shuffle: bool = True):
        with open(annotationFilePath, 'r') as annotationFile:
            annotations = json.load(annotationFile)

        self.classes = { cls: cls for cls in classes } if len(classes) > 0 else None
        self.annotations = {}
        for annotation in annotations["annotations"]:
            if self.classes is not None and annotation["category_id"] in self.classes:
                id = annotation["image_id"]
                if id in self.annotations:
                    imageAnnotations = self.annotations[id]
                else:
                    imageAnnotations = []
                    self.annotations[id] = imageAnnotations
                imageAnnotations.append(annotation)

        self.images =  [ image for image in annotations["images"] if image["id"] in self.annotations ]

        self.imagesDir = imagesDir
        self.masksDir = masksDir
        self.indices = list(range(len(self.images)))
        if shuffle:
            random.shuffle(self.indices)
        self.index = 0

    def __len__(self):
        return len(self.images)

    def readBatch(self, batchSize: int = None, loadMaskImages: bool = False):
        if batchSize is None:
            batchSize = len(self.images)
        imagesBatch = []
        masksBatch = []
        maksImagesBacth = []
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
                if len(mask.shape) == 3:
                    mask = np.bitwise_or.reduce(mask, 2)

                if commonMask is None:
                    commonMask = mask
                else:
                    commonMask = np.logical_or(commonMask, mask)
            commonMask = commonMask.astype(np.float32)
            maskName = os.path.splitext(imageName)[0] + ".png"
            maskImage = cv2.imread(os.path.join(self.masksDir, maskName))

            if image is not None and commonMask is not None:
                if image.shape[0] != 480 or image.shape[1] != 640:
                    image = cv2.resize(image, (640, 480))
                if commonMask.shape[0] != 480 or commonMask.shape[1] != 640:
                    commonMask = cv2.resize(commonMask, (640, 480))

        

                imagesBatch.append(image)
                masksBatch.append(commonMask)
                if loadMaskImages:
                    maksImagesBacth.append(maskImage)
            self.index += 1
        imagesBatch = tf.convert_to_tensor(np.stack(imagesBatch))
        masksBatch = tf.convert_to_tensor(np.stack(masksBatch))

        return (imagesBatch, masksBatch, maksImagesBacth) if loadMaskImages else (imagesBatch, masksBatch)
