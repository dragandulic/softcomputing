# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 02:07:14 2018

@author: dragan
"""

import numpy as np
import cv2
import os.path

import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16,12



from cnn import createSaveCnn
from keras.models import model_from_json

from scipy import ndimage
from vector import distance, pnt2line
import time






nameVideo = 'video/video-5.avi'

if not os.path.exists('model.h5'):
    print("Model nije kreiran, potrebno ga je kreirati!")
    nn,loaded_model = createSaveCnn()
    if nn == 1:
        print("uspesno kreiran model")
    else:
        print("nije kreiran model")
else:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")




def houghTrans(imageG,image_org):

    edges = cv2.Canny(imageG,50,150,apertureSize = 3)
    a=50
    b=10
    c=np.pi / 180
    linesP = cv2.HoughLinesP(edges, 1,c, 50, None, a, b)
    
    #for i in range(0, len(linesP)):
    #print('----%d' % len(linesP)) 
    
    if (len(linesP)>0):   
        l = linesP[1][0]
        cv2.line(image_org, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)
            
    return l

import math
def findPoints(lines):    
    dist=0
    Xmin=10000
    Ymin=10000
    Ymax=1
    Xmax=1
    for i in  range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            
            k1=x2-x1
            k2=y1-y2
            dist= math.sqrt(k1*k1 - k2*k2)
            print(dist)
            print("x1={x1}".format(x1=x1))
            print("y1={y1}".format(y1=y1))
            print("-----------------------")
            print("x2={x2}".format(x2=x2))
            print("y2={y2}".format(y2=y2))
            if x1<Xmin :
                Xmin=x1
                Ymin=y1
            if x2>Xmax: #and y2>60:
                Ymax=y2
                Xmax=x2
   
    return Xmin,Ymin,Xmax,Ymax





def selectReg(image_org,centar):        

    gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
    if nameVideo == 'video/video-5.avi':
        region = gray[centar[1]-13:centar[1]+13,centar[0]-9:centar[0]+9]
    else:
        region = gray[centar[1]-12:centar[1]+12,centar[0]-12:centar[0]+12]
    picturesRegion = []
    region1 = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    picturesRegion.append(region1)
           
   
    
        
    
    
    return image_org,picturesRegion




def vratiBroj(pictures,loaded_model):
 
    pictures = np.asarray(pictures)

    pictures = pictures.reshape(pictures.shape[0], 1, 28, 28)

    pictures = pictures.astype('float32')

    pictures /= 255
    
    result = loaded_model.predict(np.array(pictures[0:1], np.float32))
    
    
    broj = 0
    brojac = 0
    maxValue = np.max(result)
    for res in result[0]:
        if result[0][brojac] == maxValue:
            broj = brojac
        brojac+=1
    
    
    return broj





def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal




# color filter
kernel = np.ones((2,2),np.uint8)

lower = np.array([230, 230, 230])
upper = np.array([255, 255, 255])
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

elements = []
t =0
counter = 0
times = []


sumaBrojeva = 0

cc = -1
stap = 1
line = 0
def findObject(image):
    global stap
    global sumaBrojeva
    global line
    global cc
    global t
    img_org = image.copy()
    start_time = time.time()
    gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    
    if stap == 1:    
        line = houghTrans(gray,image)         
        stap = 2
    
     
        
    mask = cv2.inRange(image, lower, upper)    
    img0 = 1.0*mask

    img0 = cv2.dilate(img0,kernel) #cv2.erode(img0,kernel)
    img0 = cv2.dilate(img0,kernel)

    labeled, nr_objects = ndimage.label(img0)
    objects = ndimage.find_objects(labeled)
    
    for i in range(nr_objects):
        
        loc = objects[i]
        a1 = loc[1].stop
        b1 = loc[1].start
        xc = (a1 + b1)/2
        
        a2 = loc[0].stop
        b2 = loc[0].start
        yc = (a2 + b2)/2
        
        c1 = a1 - b1
        c2 = a2 - b2
        dxc = c1
        dyc = c2
        
    
        
        if(dxc>10 or dyc>10):
            
            cv2.circle(image, (xc,yc), 16, (25, 25, 255), 1)
            elem = {'center':(xc,yc), 'size':(dxc,dyc), 't':t}
            # find in range

            lst = inRange(20, elem, elements)
            
            nn = len(lst)

            if nn == 0:
                cc = cc + 1
                elem['id'] = cc
                elem['t'] = t
                elem['pass'] = False
                xc1 = xc
                yc1 = yc
                elem['history'] = [{'center':(xc1,yc1), 'size':(dxc,dyc), 't':t}]
                elem['future'] = [] 
                elem['number'] = None
                elements.append(elem)

            elif nn == 1:
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                xc1 = xc
                yc1 = yc
                lst[0]['history'].append({'center':(xc1,yc1), 'size':(dxc,dyc), 't':t}) 
                lst[0]['future'] = [] 
           
    for el in elements:
        tt = t - el['t']
        if(tt<3):
            if el['number'] is None:
                
                a,b=selectReg(img_org,el['center'])
            
                br = vratiBroj(b,loaded_model)

            if nameVideo == 'video/video-0.avi':
                a=line[0]
                b=line[1]
                c=line[2]+10
                d=line[3]-8
                dist, pnt, r = pnt2line(el['center'], (a,b), (c,d))
            else:
                dist, pnt, r = pnt2line(el['center'], (line[0],line[1]), (line[2],line[3]))
            
            passed = False
            er = r
            if er>0:
                
                passed = True
                
               
                if(dist<9):
                    
                    if el['pass'] == False:
                        el['pass'] = True
                        a,b=selectReg(img_org,el['center'])
                        sumaBrojeva += br
                        
                        #print('Suma %d' % sumaBrojeva)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    t +=1
    

framesG = []

cap = cv2.VideoCapture(nameVideo)



while(cap.isOpened):
    ret, frame = cap.read()
    #ret vraca false ako video nije uspesno otvoren ili nema vise frejmova
        
    
    if ret==False:
        break
    
    
    findObject(frame)
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    framesG.append(gray)
        
    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    
cap.release()
cv2.destroyAllWindows()



















