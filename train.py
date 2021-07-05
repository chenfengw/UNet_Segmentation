import torch
from torch.autograd import Variable
from torch import optim
import dataLoader
import argparse
from torch.utils.data import DataLoader
import model
import os
import utils
import scipy.io as io
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--imageRoot', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cse152-252-sp20-public/hw3_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', help='path to input images' )
parser.add_argument('--experiment', default='u_net', help='the path to store training model')
parser.add_argument('--nepoch', type=int, default=50, help='the number of epochs being trained')
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )

# -------- parse param --------
opt = parser.parse_args()
print(opt)

# -------- set variabels --------
colormap = io.loadmat(opt.colormap )['cmap']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device is {device}")
os.system('mkdir %s' % opt.experiment) # make path to save model

# -------- Initialize network --------
if opt.experiment == "dilation":
    encoder = model.encoderDilation()
    decoder = model.decoderDilation()
    print(f"training {opt.experiment}")
elif opt.experiment == "spp":
    encoder = model.encoderSPP()
    decoder = model.decoderSPP()
    print(f"training {opt.experiment}")
elif opt.experiment == "u_net":
    encoder = model.encoder()
    decoder = model.decoder()
    print(f"training {opt.experiment}")
else:
    ValueError("Please choose dilation, ssp, u_net for experiment")

# load Resnet param
model.loadPretrainedWeight(encoder)
encoder = encoder.to(device)
decoder = decoder.to(device)

# -------- Initialize optimizer  --------
optimizer = optim.Adam([
                {'params': encoder.parameters()},
                {'params': decoder.parameters()}
            ], lr=3e-4)

# -------- Initialize dataLoader --------
segDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        labelRoot = opt.labelRoot,
        fileList = opt.fileList,
        imWidth = 224, 
        imHeight = 224
        )
segLoader = DataLoader(segDataset, batch_size=32, num_workers=8, shuffle=True)

# -------- training --------
tb_writer = SummaryWriter()
iteration = 0

for epoch in range(opt.nepoch):
    for i, dataBatch in enumerate(segLoader):
        iteration += 1

        # Read data
        imBatch = Variable(dataBatch['im']).to(device)               # [32, 3, 224, 224]
        labelBatch = Variable(dataBatch['label']).to(device)         # [32, 21, 224, 224]           
        labelIndexBatch = Variable(dataBatch['labelIndex']).to(device) # [32, 1, 224, 224]
        maskBatch = Variable(dataBatch['mask']).to(device) # [32, 1, 224, 224]

        # Test network
        x1, x2, x3, x4, x5 = encoder(imBatch)
        pred = decoder(imBatch, x1, x2, x3, x4, x5)  # pred.shape = [32, 21, 224, 224]
        loss = torch.mean(pred * labelBatch)
        
        # Train network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calcuate traning acc
        hist = utils.computeAccuracy(pred, labelIndexBatch, maskBatch)
        avg_acc, class_acc = utils.compute_batch_accuracy(hist, opt.numClasses)
        tb_writer.add_scalar('training loss', loss.item(), iteration)
        tb_writer.add_scalar('training accuracy', avg_acc, iteration)
        
        if iteration % 100 == 0:
            print(f"Epoch {epoch}, iteration {iteration}, avg_acc {avg_acc}")
    
    # save model parameter
    torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch+1))
    torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch+1))
tb_writer.close()
