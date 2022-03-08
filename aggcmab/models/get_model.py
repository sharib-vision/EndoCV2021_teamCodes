import sys
import segmentation_models_pytorch as smp
import torch

class SMP_W(torch.nn.Module):
    def __init__(self, decoder = smp.FPN, encoder_name='resnet34', encoder2_name=None, in_channels=3,
                                          encoder_weights='imagenet', classes=1, mode='train'):
        super(SMP_W, self).__init__()
        if encoder2_name is None: encoder2_name=encoder_name
        self.m1 = decoder(encoder_name=encoder_name, in_channels=in_channels, encoder_weights=encoder_weights, classes=classes)
        self.m2 = decoder(encoder_name=encoder2_name, in_channels=in_channels+classes, encoder_weights=encoder_weights, classes=classes)
        self.n_classes = classes
        self.mode=mode
    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(torch.cat([x, x1], dim=1))
        if self.mode!='train':
            return x2
        return x1,x2

def get_arch(model_name, in_c=3, n_classes=1):
    ## FPNET ##
    if model_name == 'fpnet_resnet18':
        model = smp.FPN(encoder_name='resnet18', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnet18_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet18', in_channels=in_c, classes=n_classes)

    elif model_name == 'fpnet_mobilenet':
        model = smp.FPN(encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_mobilenet_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)

    elif model_name == 'unet_mobilenet':
        model = smp.Unet(encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)
    elif model_name == 'unet_mobilenet_W':
        model = SMP_W(decoder=smp.Unet, encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)

    elif model_name == 'deeplab_mobilenet':
        model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_mobilenet_W':
        model = SMP_W(decoder=smp.DeepLabV3Plus, encoder_name='mobilenet_v2', in_channels=in_c, classes=n_classes)



    elif model_name == 'fpnet_resnet34':
        model = smp.FPN(encoder_name='resnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnet34_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_resnet34':
        model = smp.DeepLabV3Plus(encoder_name='resnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'deeplab_resnet34_W':
        model = SMP_W(decoder=smp.DeepLabV3Plus, encoder_name='resnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'unet_resnet34_W':
        model = smp.Unet(encoder_name='resnet34', in_channels=in_c, classes=n_classes)
    elif model_name == 'unet_resnet34_W':
        model = SMP_W(decoder=smp.Unet, encoder_name='resnet34', in_channels=in_c, classes=n_classes)


    elif model_name == 'fpnet_resnet50':
        model = smp.FPN(encoder_name='resnet50', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_resnet50_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='resnet50', in_channels=in_c, classes=n_classes)

    elif model_name == 'fpnet_dpn68':
        model = smp.FPN(encoder_name='dpn68', in_channels=in_c, classes=n_classes)
    elif model_name == 'fpnet_dpn68_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='dpn68', in_channels=in_c, classes=n_classes)

    elif model_name == 'fpnet_dpn92':
        model = smp.FPN(encoder_name='dpn92', in_channels=in_c, classes=n_classes, encoder_weights='imagenet+5k')
    elif model_name == 'fpnet_dpn92_W':
        model = SMP_W(decoder=smp.FPN, encoder_name='dpn92', in_channels=in_c, classes=n_classes, encoder_weights='imagenet+5k')

    ########################

    else: sys.exit('not a valid model_name, check models.get_model.py')

    setattr(model, 'n_classes', n_classes)

    if 'dpn' in model_name:
        mean, std = [124 / 255, 117 / 255, 104 / 255], [1 / (.0167 * 255)] * 3
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return model, mean, std



