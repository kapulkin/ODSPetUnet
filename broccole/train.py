import segmentation_models as sm
import argparse
import os
import cv2

from broccole.CocoDataset import CocoDataset

def train(trainDataset: CocoDataset, validationDataset: CocoDataset):
    encoder = 'resnet18'
    preprocess_input = sm.get_preprocessing(encoder)

    model = sm.Unet(encoder, classes=1, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    validationPacketSize = 32 * 32
    x_val, y_val = validationDataset.readBatch(validationPacketSize)
    x_val = preprocess_input(x_val)

    packetSize = 16 * 16
    batchSize = 1 # 16
    epochs = 100
    for epoch in range(epochs):
        print('epoch {}'.format(epoch))
        packets = len(trainDataset) // packetSize
        for _ in range(packets - 1):
            x_train, y_train = trainDataset.readBatch(packetSize)
            x_train = preprocess_input(x_train)

            model.fit(
                x=x_train,
                y=y_train,
                batch_size=batchSize,
                epochs=1,
                validation_data=(x_val, y_val),
            )

        packet = trainDataset.readBatch(packetSize)

        model.fit(
            x=x_train,
            y=y_train,
            batch_size=16,
            epochs=1,
            validation_data=(x_val, y_val),
        )

def showDataset(dataset: CocoDataset):
    batch = dataset.readBatch(10, loadMaskImages=True)
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

def parse_args():
    parser = argparse.ArgumentParser(description='train U-Net')
    parser.add_argument('--datasetDir', help='path to directory with dataset', type=str)    
    args = parser.parse_args()
    return args

def main():
    #TODO:
    # load coco dataset: images, mask
    # init model
    # train
    # iterate over dataset with batches
    # 

    args = parse_args()
    # datasetDir = 'D:\\data\\segmentation\\data\\coco'
    datasetDir = args.datasetDir
    trainDataset = CocoDataset(
        os.path.join(datasetDir, 'annotations/instances_train2017.json'),
        os.path.join(datasetDir, 'train2017'),
        os.path.join(datasetDir, 'annotations/panoptic_train2017'),
        classes=[1]
    )

    valDataset = CocoDataset(
        os.path.join(datasetDir, 'annotations/instances_val2017.json'),
        os.path.join(datasetDir, 'val2017'),
        os.path.join(datasetDir, 'annotations/panoptic_val2017'),
        classes=[1]
    )

    train(trainDataset, valDataset)


if __name__ == '__main__':
    main()