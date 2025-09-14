# -*- coding: utf-8 -*-
"""
Author: Yuxuan(Jarret) Jiang
Email: jiangyx96@gmail.com
Date Created: 2025-08
Version: 1.0.0
Description:  This code applies a trained neural network to predict image segmentation results for a specified video.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm
import scipy.io as sio
import init1

dir_data = 'Dataset'
dir_model = 'output'
dir_res = 'output'

# --- Read video and pre-process ---
video_name='c210904160305137.avi'
video, framerate = init1.loadvideo(os.path.join(dir_data,video_name))
video = video.astype(np.float32)
# [channel, frame, depth, width]

# standardize
mean = np.array([5.823330688476562500e+01, 5.823078155517578125e+01, 5.823069381713867188e+01]).reshape(3, 1, 1, 1)
std = np.array([6.343706893920898438e+01,6.343493652343750000e+01, 6.343499755859375000e+01]).reshape(3, 1, 1, 1)

video2 = (video - mean)/std
video2 = video2.transpose(1,0,2,3).astype(np.float32)
# [frame, channel, depth, width]

# --- Load Model ----
model_name='deeplabv3_resnet50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.segmentation.__dict__[model_name](pretrained=False, aux_loss=False)
model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1

model = torch.nn.DataParallel(model)
model.to(device)

checkpoint = torch.load(os.path.join(dir_model, "best.pt"))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# --- Prediction by the model ---
batch_size = 10
logit = np.empty((0,1,448,448))
with torch.no_grad():
    video2 = torch.tensor(video2).to(device)
    for ii in tqdm.tqdm(range(0,video.shape[1],batch_size)):
        logit0=model(video2[ii:(ii+batch_size),:,:,:])['out'].detach().cpu().numpy()
        logit = np.concatenate((logit,logit0),axis=0)
        
logit = logit[:,0,:,:]

# --- generate video with masks
video3 = np.transpose(video,(1,0,2,3)) # [frames, channel, depth, width]
video3[:, 0, :, :] = np.maximum(255. * (logit > 0), video3[:, 0, :, :])

# --- Check for one frame ---
frame1 = video3[0,:,:,:].transpose(1,2,0)
frame1b = (frame1-frame1.min())/(frame1.max()-frame1.min())

plt.figure(1)
plt.imshow(frame1b, cmap='gray')

# --- Calculate area for the video ---
art_area = (logit > 0).sum((1, 2))
tseq = np.arange(art_area.shape[0]) / framerate
plt.figure(2)
plt.scatter(tseq, art_area, s=1) # s stands for "size of scatter"
plt.xlabel('Time (s)')
plt.ylabel('Area')

# # ----- Save video 1 ------
# video4 = video3.transpose(1, 0, 2, 3).astype(np.uint8)
# init1.savevideo(os.path.join(dir_res, 'Seg1-'+video_name), video4, framerate)

# # ---- Save video 2 (combine area curve) ----
# video5 = np.concatenate((video3, np.zeros(video3.shape[:3] + (video3.shape[0]+10,))), 3)

# tmp = (art_area-art_area.min())/(art_area.max()-art_area.min()); tmp = 1-tmp
# for (f, s) in enumerate(tmp):
#     yloc = int(round(s*350+50)); xloc = video3.shape[3]+f+1
#     video5[:,:,yloc,xloc] = 255.
#     r, c = skimage.draw.disk((yloc, xloc), 6)
#     video5[f, 0, r, c] = 255.

# video5 = video5.transpose(1, 0, 2, 3).astype(np.uint8)
# init1.savevideo(os.path.join(dir_res, 'Seg2-'+video_name), video5, framerate)

# --- Save data ---
mask=logit; mask[logit>0]=1; mask[logit<0]=0; mask=mask.astype(np.uint8)
output_name = 'Res-'+video_name[:video_name.find('.')]+'.mat'
mat1 = np.vstack((tseq, art_area))
result = {'name':video_name,'data':mat1,'mask':mask}
sio.savemat(file_name=os.path.join(dir_res,output_name), mdict=result, do_compression=True)
