import json
import cv2
import os
import random
import numpy as np
from typing import List
from pycocotools import mask as cocoMask
import logging

logger = logging.getLogger(__name__)

from broccole.CocoDataset import CocoDataset

class CocoDatasetBuilder:
    def __init__(self, annotationFilePath: str, imagesDir: str):
        with open(annotationFilePath, 'r') as annotationFile:
            self.cocoAnnotations = json.load(annotationFile)
            logger.debug("coco annotations are loaded")
        self.imagesDir = imagesDir

        self.annotations = {}

    def selectAll(self):
        self.annotations = {}
        for annotation in self.cocoAnnotations["annotations"]:
            id = annotation["image_id"]
            if id in self.annotations:
                imageAnnotations = self.annotations[id]
            else:
                imageAnnotations = []
                self.annotations[id] = imageAnnotations
            imageAnnotations.append(annotation)
        return self

    def addClasses(self, classes: List):
        if len(classes) == 0:
            raise Exception('classes is empty list')
        classes = { cls: cls for cls in classes }

        for annotation in self.cocoAnnotations["annotations"]:
            if annotation["category_id"] in classes:
                id = annotation["image_id"]
                if id in self.annotations:
                    imageAnnotations = self.annotations[id]
                else:
                    imageAnnotations = []
                    self.annotations[id] = imageAnnotations
                imageAnnotations.append(annotation)
        return self

    def filterNonClasses(self, classes: List):
        if len(classes) == 0:
            raise Exception('classes is empty list')
        classes = { cls: cls for cls in classes }

        annotations = {}

        for id, imageAnnotations in self.annotations.items():
            hasClass = False
            for annotation in imageAnnotations:
                id = annotation["image_id"]
                if annotation["category_id"] in classes:
                    hasClass = True
                    break
            if not hasClass:
                annotations[id] = imageAnnotations
        self.annotations = annotations

        return self

    def addNonClasses(self, classes: List, maxCount: int = None, shuffle: bool = False):
        if len(classes) == 0:
            raise Exception('classes is empty list')
        classes = { cls: cls for cls in classes }

        indices = list(range(len(self.cocoAnnotations["annotations"])))
        if shuffle:
            random.shuffle(indices)

        count = 0
        for index in indices:
            annotation = self.cocoAnnotations["annotations"][index]
            if maxCount is not None and count >= maxCount:
                break

            if annotation["category_id"] not in classes:
                id = annotation["image_id"]
                if id in self.annotations:
                    imageAnnotations = self.annotations[id]
                else:
                    imageAnnotations = []
                    self.annotations[id] = imageAnnotations
                imageAnnotations.append(annotation)
                count += 1
        return self

    def build(self, shuffle: bool = False):
        self.images =  [ image for image in self.cocoAnnotations["images"] if image["id"] in self.annotations ]

        self.indices = list(range(len(self.images)))
        if shuffle:
            random.shuffle(self.indices)
        self.index = 0

        return CocoDataset(self.annotations, self.images, self.imagesDir, shuffle)

    def __len__(self):
        return len(self.annotations)
