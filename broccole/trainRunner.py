import subprocess

def main():
    for i in range(100):
        print('Epoch {}'.format(i))
        subprocess.run([
            "python", "-m", "broccole.trainPrepared",
            "--datasetDir", "E:\\data\\segmentation\\data\\coco", "--datasetType", "coco", "--batchSize", "8", "--epochs", "1", "--checkpointFilePath", "E:\\data\\segmentation\\data\\coco\\u-net-resnet18.chpt"
        ])

if __name__ == "__main__":
    main()