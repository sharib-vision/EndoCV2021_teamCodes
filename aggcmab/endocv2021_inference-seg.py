#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021

@author: endocv2021@generalizationChallenge
"""

# import network

import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import skimage
from skimage import io
from skimage.transform import resize as rsz_sk
from  tifffile import imsave
from models.get_model import get_arch

def create_predFolder(task_type):
    directoryName = 'EndoCV2021'
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)
        
    if not os.path.exists(os.path.join(directoryName, task_type)):
        os.mkdir(os.path.join(directoryName, task_type))
        
    return os.path.join(directoryName, task_type)

def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_classes", type=int, default=1, help="num classes (default: None)")

    # Deeplab Options

    parser.add_argument("--model_name", type=str, default='fpnet_mobilenet_W', help='model name')

    parser.add_argument("--ckpt_path", type=str, default='/home/aggcmab/code/checkpoints/F1/fpnet_mobilenet_W/', help='checkpoint path')

    parser.add_argument("--im_size", help='delimited list input, could be 512, or 480,600', type=str, default='512,640')

    parser.add_argument("--gpu_id", type=str, default='1', help="GPU ID")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")

    return parser
    
def mymodel():
    '''
    Returns
    -------
    model : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    '''
    opts = get_argparser().parse_args()

    im_size = tuple([int(item) for item in opts.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    print(opts.model_name)


    model, mean, std = get_arch(opts.model_name, n_classes=opts.n_classes)
    checkpoint = torch.load(osp.join(opts.ckpt_path, 'model_checkpoint.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    # model = nn.DataParallel(model)
    model.to(device)
    model.mode = 'eval'
    model.eval()

    return model, mean, std, tg_size, device


        
if __name__ == '__main__':
    '''
     You are not allowed to print the images or visualizing the test data according to the rule. 
     We expect all the users to abide by this rule and help us have a fair challenge "EndoCV2021-Generalizability challenge"
     
     FAQs:
         1) Most of my predictions do not have polyp.
            --> This can be the case as this is a generalisation challenge. The dataset is very different and can produce such results. In general, not all samples 
            have polyp.
        2) What format should I save the predictions.
            --> you can save it in the tif or jpg format. 
        3) Can I visualize the data or copy them in my local computer to see?
            --> No, you are not allowed to do this. This is against challenge rules. No test data can be copied or visualised to get insight. Please treat this as unseen image.!!!
        4) Can I use my own test code?
            --> Yes, but please make sure that you follow the rules. Any visulization or copy of test data is against the challenge rules. We make sure that the 
            competition is fair and results are replicative.
    '''
    model, mean, std, tg_size, device = mymodel()
    task_type = 'segmentation'
    # set image folder here!
    directoryName = create_predFolder(task_type)
    
    # ----> three test folders [https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/wiki/EndoCV2021-Leaderboard-guide]
    subDirs = ['EndoCV_DATA1', 'EndoCV_DATA2', 'EndoCV_DATA3']
    print(subDirs)

    for j in range(0, len(subDirs)):
        
        # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
        imgfolder='/project/def-sponsor00/endocv2021-test-noCopyAllowed-v1/' + subDirs[j]
        
        # set folder to save your checkpoints here!
        saveDir = os.path.join(directoryName , subDirs[j]+'_pred')
    
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        imgfiles = detect_imgs(imgfolder, ext='.jpg')
    
        from torchvision import transforms

        data_transforms = transforms.Compose([
            transforms.Resize(tg_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        file = open(saveDir + '/'+"timeElaspsed" + subDirs[j] +'.txt', mode='w')
        timeappend = []
    
        for imagePath in imgfiles[:]:
            """plt.imshow(img1[:,:,(2,1,0)])
            Grab the name of the file. 
            """
            filename = (imagePath.split('/')[-1]).split('.jpg')[0]
            print('filename is printing::=====>>', filename)
            
            img1 = Image.open(imagePath).convert('RGB').resize((256,256), resample=0)
            image = data_transforms(img1)
            # perform inference here:
            images = image.to(device, dtype=torch.float32)
            
            #            
            img = skimage.io.imread(imagePath)
            size=img.shape
            start.record()
            #
            outputs = model(images.unsqueeze(0))
            #
            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            timeappend.append(start.elapsed_time(end))
            #

            probs = outputs.squeeze().sigmoid().detach().cpu()
            preds = (probs > 0.5).numpy()
            probs = probs.numpy()

            pred = (preds * 255.0).astype(np.uint8)
            prob = (probs * 255.0).astype(np.uint8)

            img_mask = rsz_sk(pred, (size[0], size[1]), anti_aliasing=True)
            img_prob = rsz_sk(prob, (size[0], size[1]), anti_aliasing=True)

            io.imsave(saveDir + '/' + filename + '_mask.jpg', (img_mask * 255.0).astype('uint8'))
            io.imsave(saveDir + '/' + filename + '_prob.jpg', (img_prob * 255.0).astype('uint8'))
            
            
            file.write('%s -----> %s \n' % 
               (filename, start.elapsed_time(end)))
            
    
            # TODO: write time in a text file
        
        file.write('%s -----> %s \n' % 
           ('average_t', np.mean(timeappend)))
