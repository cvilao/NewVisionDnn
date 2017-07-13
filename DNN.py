import argparse
import time
import cv2

# import the necessary packages
import imutils

import os
#import tarfile
import time
#import zipfile
from classify import *
import os
import cv2
import numpy as np
import sys

import Condensation



class objectDetect():

    kernel_perto = np.ones((39, 39), np.uint8)
    kernel_perto2 = np.ones((100, 100), np.uint8)
    kernel_medio = np.ones((22, 22), np.uint8)
    kernel_medio2 = np.ones((80, 80), np.uint8)
    kernel_longe = np.ones((12, 12), np.uint8)
    kernel_longe2 = np.ones((40, 40), np.uint8)
    kernel_muito_longe = np.ones((7, 7), np.uint8)
    kernel_muito_longe2 = np.ones((30, 30), np.uint8)
    mean_file = None
    labels = None
    net = None
    transformer = None

    def __init__(self, net, transformer, mean_file, labels):
        self.mean_file = mean_file
        self.labels = labels
        self.net = net
        self.transformer = transformer

    def searchball(self, image):

        frame, x, y, raio = Morphology(self,image,self.kernel_perto2)
        if (x==0 and y==0 and raio==0):
            frame, x, y, raio = Morphology(self,image,self.kernel_medio2)
            if (x==0 and y==0 and raio==0):
                frame, x, y, raio = Morphology(self,image,self.kernel_longe2)
                if (x==0 and y==0 and raio==0):
                    frame, x, y, raio = Morphology(self,image,self.kernel_muito_longe2)
                    if (x==0 and y==0 and raio==0):
#                        bkb.write_int(Mem,'VISION_LOST', 1)
			print("Ball not found")

        return frame, x, y, raio


def SearchLostBall(self):
    self.servo.writeWord(self.__SERVO_PAN, self.__SPEED, 200)
    self.servo.writeWord(self.__SERVO_PAN, self.__SPEED, 200)
    self.servo.writeWord(self.__SERVO_PAN, self.__SPEED, 200)





def Morphology(self, frame, kernel):

    YUV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    white_mask = cv2.inRange(YUV_frame[:,:,0], 200, 255)
#    cv2.imshow('mask',white_mask)
    mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, self.kernel_perto)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel_perto2,1)

    ret,th1 = cv2.threshold(mask,25,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
            #Passa para o classificador as imagens recortadas-----------------------
        type_label, results = classify(cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2RGB),
                                                           self.net, self.transformer,
                                                           mean_file=self.mean_file, labels=self.labels,
                                                           batch_size=None)
        #-----------------------------------------------------------------------
#            print results, type_label
    #       cv2.imshow('janela',images[0])
        if type_label == 'Ball':
            return frame, x+w/2, y+h/2, (w+h)/4
        #=================================================================================================
    return frame, 0, 0, 0




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification example using an archive - DIGITS')

###    # Positional arguments
    parser.add_argument('archive', help='Path to a DIGITS model archive')
###    #parser.add_argument('image_file', nargs='+', help='Path[s] to an image')
###    # Optional arguments
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--nogpu', action='store_true', help="Don't use the GPU")

    parser.add_argument('--ws', '--ws', action='store_true', help="no servo")

    args = vars(parser.parse_args())
    args2 = parser.parse_args()

    tmpdir = unzip_archive(args['archive'])
    caffemodel = None
    deploy_file = None
    mean_file = None
    labels_file = None
    for filename in os.listdir(tmpdir):
        full_path = os.path.join(tmpdir, filename)
        if filename.endswith('.caffemodel'):
            caffemodel = full_path
        elif filename == 'deploy.prototxt':
            deploy_file = full_path
        elif filename.endswith('.binaryproto'):
            mean_file = full_path
        elif filename == 'labels.txt':
            labels_file = full_path
        else:
            print 'Unknown file:', filename

    assert caffemodel is not None, 'Caffe model file not found'
    assert deploy_file is not None, 'Deploy file not found'

###    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu=False)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    labels = read_labels(labels_file)

###    #create index from label to use in decicion action
    number_label =  dict(zip(labels, range(len(labels))))
    print number_label

#    detectBall = objectDetect(net, transformer, mean_file, labels)

#    os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0 && v4l2-ctl -d /dev/video0 -c focus_absolute=0")
#    cap = cv2.VideoCapture(0)
#    cap.set(3,1280) #720 1280 1920
#    cap.set(4,720) #480 720 1080

#    while True:

#        script_start_time = time.time()

###        # Capture frame-by-frame
#        ret, frame = cap.read()
#        ret, frame = cap.read()
#        ret, frame = cap.read()
#        ret, frame = cap.read()
#        frame = frame[:,200:1100]

#        #===============================================================================
#        frame, x, y, raio = detectBall.searchball(frame)

#        cv2.circle(frame, (x, y), raio, (0, 255, 0), 4)
##        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#        cv2.imshow('frame',frame)
#        print "tempo de varredura = ", time.time() - script_start_time
#        #===============================================================================


#        print 'Script took %f seconds.' % (time.time() - script_start_time,)

#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

#    # When everything done, release the capture
#    cap.release()
#    cv2.destroyAllWindows()

