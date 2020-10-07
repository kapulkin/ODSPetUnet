import segmentation_models as sm
import os
import cv2

from broccole.CocoDataset import CocoDataset

def main():    
    #TODO:
    # load coco dataset: images, mask
    # init model
    # train
    # iterate over dataset with batches
    # 
    datasetDir = 'D:\\data\\segmentation\\data\\coco'
    cocoDataset = CocoDataset(
        os.path.join(datasetDir, 'annotations/instances_train2017.json'), #'annotations/panoptic_train2017.json'),
        os.path.join(datasetDir, 'train2017'),
        os.path.join(datasetDir, 'annotations/panoptic_train2017')
    )

    batch = cocoDataset.readBatch(10)
    for imageMask in batch:
        cv2.imshow('image', imageMask[0])
        cv2.imshow('mask', imageMask[1] * 255)
        fileMask = imageMask[2]
        if fileMask is not None:
            cv2.imshow('fileMask', fileMask)
        else:
            cv2.destroyWindow("fileMask")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()