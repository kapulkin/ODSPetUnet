import segmentation_models as sm

def makeModel():
    encoder = 'resnet18'
    preprocess_input = sm.get_preprocessing(encoder)

    model = sm.Unet(encoder, classes=1, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    return model, preprocess_input