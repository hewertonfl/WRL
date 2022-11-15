from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import eval2Furo as eval2
import time, datetime
import pickle 

cap = cv2.VideoCapture("12.jpg")
cap.release()
ximg = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
yimg = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))                              

width = ximg/100
height = yimg/100

aux = 1
######Initial Config########
   
eval2.parse_args()

if eval2.args.config is not None:
    set_cfg(eval2.args.config)

if eval2.args.trained_model == 'interrupt':
    eval2.args.trained_model = SavePath.get_interrupt('weights/')
elif eval2.args.trained_model == 'latest':
    eval2.args.trained_model = SavePath.get_latest('weights/', cfg.name)

if eval2.args.config is None:
    model_path = eval2.SavePath.from_str(eval2.args.trained_model)
    # TODO: Bad practice? Probably want to do a name lookup instead.
    eval2.args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % eval2.args.config)
    set_cfg(eval2.args.config)

if eval2.args.detect:
    cfg.eval_mask_branch = False

if eval2.args.dataset is not None:
    set_dataset(args.dataset)
    
def storeData(output, diametro, time): 
    # initializing data to be stored in db 
    global aux
    aux=aux*(-1)
    db = (output, aux)
    # Its important to use binary mode 
    dbfile = open('outputPickle', 'wb') 
    # source, destination 
    pickle.dump(db, dbfile)                   
    dbfile.close() 
    
    data = (diametro, time, aux)
    datafile = open('dataPickle', 'wb')
    pickle.dump(data, datafile)                   
    datafile.close() 
    
def detect(imgToSeg):
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if eval2.args.cuda:
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if eval2.args.resume and not eval2.args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if eval2.args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                            transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None        

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(eval2.args.trained_model)
        net.eval()
        print(' Done.')

        if eval2.args.cuda:
            net = net
            

        frame_times = MovingAverage(10)

        #####EVALUATE THE NETWORK######

        _, frame = cap.read()
        #cv2.imwrite("imgtosegFuro.jpg", frame)

        frame0 = imgToSeg[0]
        frame1 = imgToSeg[1]
        frame2 = imgToSeg[2]
        frame3 = imgToSeg[3]
        frame4 = imgToSeg[4]
        cv2.imwrite("imgtoseg0.jpg", frame0)
        cv2.imwrite("imgtoseg1.jpg", frame1)
        cv2.imwrite("imgtoseg2.jpg", frame2)
        cv2.imwrite("imgtoseg3.jpg", frame3)
        cv2.imwrite("imgtoseg4.jpg", frame4)
        str0 = 'imgtoseg0.jpg'
        str1 = 'imgtoseg1.jpg'
        str2 = 'imgtoseg2.jpg'
        str3 = 'imgtoseg3.jpg'
        str4 = 'imgtoseg4.jpg'
        
        inicio = time.time()
        timedate = datetime.datetime.now()
        img_numpy, boxes, centroFuros = eval2.evaluate(str0, str1, str2, str3, str4, net, dataset) ##Recebe a imagem segmentada e as boxes
        #print(centroFuros)
        #plt.imshow(img_numpy) 
        #plt.show()            
      
            #cv2.imshow("YOLACT", img_numpy)
        output = img_numpy
             
        
        dim=(1920, 1080)
        for x in range(len(output)):
        	output[x] = cv2.resize(output[x], dim, interpolation = cv2.INTER_AREA)      #Resize image to fit in gui
        

            
        fim = time.time()
        ftime = (fim-inicio)
            
        frame_times.add(ftime)
        video_fps = 1 / (frame_times.get_avg())

        fps_print = 'Processing Time: %.2f | FPS Average: %.2f' % (ftime, video_fps)
        print('\r' + fps_print + '    ', end='')

           # k = cv2.waitKey(10)
          #  if k & 0xFF == ord('q'):
         #        break

        #cv2.destroyAllWindows()
    return (img_numpy, boxes, centroFuros)
        
    
    
    



