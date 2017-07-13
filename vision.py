import sys
sys.path.append("./src")
import numpy as np
import os
import cv2
import ctypes
import argparse
import time
from math import log,exp,tan,radians
import thread
import imutils

from BallVision import *
from DNN import *

import sys

""" Initiate the path to blackboard (Shared Memory)"""
sys.path.append('../../Blackboard/src/')
"""Import the library Shared memory """
from SharedMemory import SharedMemory 
""" Treatment exception: Try to import configparser from python. Write and Read from config.ini file"""
try:
    """There are differences in versions of the config parser
    For versions > 3.0 """
    from ConfigParser import ConfigParser
except ImportError:
    """For versions < 3.0 """
    from ConfigParser import ConfigParser 

""" Instantiate bkb as a shared memory """
bkb = SharedMemory()
""" Config is a new configparser """
config = ConfigParser()
""" Path for the file config.ini:"""
config.read('../../Control/Data/config.ini')
""" Mem_key is for all processes to know where the blackboard is. It is robot number times 100"""
mem_key = int(config.get('Communication', 'no_player_robofei'))*100 
"""Memory constructor in mem_key"""
Mem = bkb.shd_constructor(mem_key)


parser = argparse.ArgumentParser(description='Robot Vision', epilog= 'Responsavel pela deteccao dos objetos em campo / Responsible for detection of Field objects')
parser.add_argument('--visionball', '--vb', action="store_true", help = 'Calibra valor para a visao da bola')
parser.add_argument('--withoutservo', '--ws', action="store_true", help = 'Servos desligado')
parser.add_argument('--head', '--he', action="store_true", help = 'Configurando parametros do controle da cabeca')
parser.add_argument('archive', help='Path to a DIGITS model archive')
###    #parser.add_argument('image_file', nargs='+', help='Path[s] to an image')
###    # Optional arguments
parser.add_argument('--batch-size', type=int)
parser.add_argument('--nogpu', action='store_true', help="Don't use the GPU")


#----------------------------------------------------------------------------------------------------------------------------------

#x_limit01 = 200
#x_limit12 = 570
#x_limit23 = 660
#x_limit34 = 740
#x_limit45 = 1100

y_limit1 = 300
y_limit2 = 400

SERVO_PAN = 19
SERVO_TILT = 20

x = 0
y = 0
raio = 0


#----------------------------------------------------------------------------------------------------------------------------------

def statusBall(positionballframe):
	global lista
	if positionballframe[0] == 0:
		lista = []
		print "Campo nao encontrado"
		
	if positionballframe[0] == 1:
		lista = []
		mens = "Bola nao encontrada, campo "
		
		if positionballframe[1] == -1:
			mens += "a esquerda"
		elif positionballframe[1] == 1:
			mens += "a direita"
		else:
			mens += "esta no meio"
		
		if positionballframe[2] == -1:
			mens += " cima"
		elif positionballframe[2] == 1:
			mens += " baixo"
		else:
			mens += " centro"
		print mens
	if positionballframe[0] == 2:
		if bkb.read_float(Mem, 'VISION_TILT_DEG') < 50:
			lista.append(21.62629757*exp(0.042451235*bkb.read_float(Mem, 'VISION_TILT_DEG')))#8.48048735
			##bkb.write_float(Mem, 'VISION_BALL_DIST', 430*tan(radians(bkb.read_float(Mem, 'VISION_TILT_DEG'))))
			print 'Dist using tilt angle: ', bkb.read_float(Mem, 'VISION_BALL_DIST')
			#0.0848048735*exp(0.042451235*bkb.read_int('VISION_TILT_DEG')
			#print "Distancia da Bola func 1 em metros: " + str(0.0848048735*exp(0.042451235*bkb.read_int('VISION_MOTOR1_ANGLE')))
			#print "Bola encontrada na posicao x: " + str(round(positionballframe[1],2)) + " y: " + str(round(positionballframe[2],2)) + " e diametro de: " + str(round(positionballframe[3],2))
		else:
			#print "Bola encontrada na posicao x: " + str(round(positionballframe[1],2)) + " y: " + str(round(positionballframe[2],2)) + " e diametro de: " + str(round(positionballframe[3],2))
			#print "Distancia da Bola func 2 em metros: " + str(4.1813911146*pow(positionballframe[3],-1.0724682465))
			lista.append(418.13911146*pow(positionballframe[3],-1.0724682465))
			print 'Dist using pixel size: ', bkb.read_float(Mem, 'VISION_BALL_DIST')
		if len(lista) == 1:
			dist_media = lista[0]
		else:
			if len(lista) >= 1:
				lista.pop(0)
			dist_media = float(sum(lista)/len(lista))
		bkb.write_float(Mem, 'VISION_BALL_DIST', dist_media)
		bkb.write_int(Mem,'VISION_LOST', 0)
		
		
#		print "Bola encontrada = " + str(bkb.read_int('VISION_LOST_BALL'))
#		print "Posicao servo 1 tilt = " + str(bkb.read_int('VISION_MOTOR1_ANGLE'))
	else:
	    bkb.write_int(Mem,'VISION_LOST', 1)
#	    print "Bola Perdida = " + str(bkb.read_int('VISION_LOST_BALL'))

	    
#----------------------------------------------------------------------------------------------------------------------------------
x_limit01 = 0
x_limit12 = 420
x_limit23 = 450
x_limit34 = 470
x_limit45 = 900

def BallStatus(x,y):
	#Bola a esquerda
	if (x > x_limit01 and x < x_limit12):
		bkb.write_float(Mem,'VISION_PAN_DEG', -60) # Variavel da telemetria
		print ("Bola a Esquerda")

	#Bola ao centro
	if (x > x_limit12 and x < x_limit23):
		bkb.write_float(Mem,'VISION_PAN_DEG', -30) # Variavel da telemetria
		print ("Bola ao Centro Esquerda")

	#Bola a direita
	if (x > x_limit23 and x < x_limit34):
		bkb.write_float(Mem,'VISION_PAN_DEG', 30) # Variavel da telemetria
		print ("Bola ao Centro Direita")

	#Bola a direita
	if (x > x_limit34 and x < x_limit45):
		bkb.write_float(Mem,'VISION_PAN_DEG', 60) # Variavel da telemetria
		print ("Bola a Direita")

	#Bola abaixo
	if (y > 1 and y < 200):#y_limit1):
		bkb.write_float(Mem,'VISION_TILT_DEG', 0) # Variavel da telemetria
		print ("Bola acima")
	#Bola ao centro
	if (y > y_limit1 and y < y_limit2):
		bkb.write_float(Mem,'VISION_TILT_DEG', 45) # Variavel da telemetria
		print ("Bola Centralizada")
	#Bola acima
	if (y > y_limit2 and y < 720):
		bkb.write_float(Mem,'VISION_TILT_DEG', 70) # Variavel da telemetria
		print ("Bola abaixo")

def applyMask(frame):
	lower = np.array([23, 0,0])
	upper = np.array([57,255,255])
        kernel = np.ones((5,5),np.uint8)
#        mask = frame
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	mask = cv2.inRange(hsv, lower, upper)
	
	## erosion
	mask = cv2.erode(mask,kernel,iterations=2)
	
	## dilation
	mask = cv2.dilate(mask,kernel,iterations=2)
	#mostra = cv2.bitwise_and(frame,frame,mask=mask)
	return mask





def cutFrame(mask_verde):
##	#cima
	cima = -1
	for i in range(0,len(mask_verde),5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo pixels
			cima = i
			break
#	
#	#baixo
	baixo = -1
	for i in range(len(mask_verde)-1,-1,-5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo pixels
			baixo = i
			break
#	
#	# Girando mascara
	mask_verde=mask_verde.transpose()
#	
#	#esp p/ dir
	esquerda = -1
	for i in range(0,len(mask_verde),5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo 4 pixels
			esquerda = i
			break
#	
#	#dir p/ esp
	direita = -1
	for i in range(len(mask_verde)-1,0,-5):
		if np.any(sum(mask_verde[i]))>int(255*500): #minimo 4 pixels
			direita = i
			break
	
	return np.array([esquerda,direita,cima,baixo])


def thread_DNN():
	time.sleep(1)
	while True:
		script_start_time = time.time()

		print "FRAME = ", time.time() - script_start_time
		start = time.time()
#===============================================================================

		frame_b, x, y, raio = detectBall.searchball(frame)
#		BallStatus(x,y)
		#if args2.visionball:
		cv2.circle(frame_b, (x, y), raio, (0, 255, 0), 4)
		cv2.imshow('frame',frame_b)
#===============================================================================
		print "tempo de varredura = ", time.time() - start
	cap.release()
	cv2.destroyAllWindows()

#frame = 0







#----------------------------------------------------------------------------------------------------------------------------------
#Inicio programa

if __name__ == '__main__':


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
###    #
	detectBall = objectDetect(net, transformer, mean_file, labels)


	cap = cv2.VideoCapture(0) #Abrindo camera
        cap.set(3,1280) #720 1280 1920
        cap.set(4,720) #480 720 1080
	os.system("v4l2-ctl -d /dev/video0 -c focus_auto=0 && v4l2-ctl -d /dev/video0 -c focus_absolute=0")

	try:
            thread.start_new_thread(thread_DNN, ())
	except:
            print "Error Thread"



	while True:

		bkb.write_int(Mem,'VISION_WORKING', 1) # Variavel da telemetria

		#Salva o frame

		script_start_time = time.time()

#                ret, frame = cap.read()
#                ret, frame = cap.read()
#                ret, frame = cap.read()
                ret, frame = cap.read()
                frame = frame[:,200:1100]
		
		print "FRAME = ", time.time() - script_start_time
		start = time.time()
#===============================================================================
		cv2.imshow('Original',frame)
#		mask_verde = applyMask(frame)
#		#if mask_verde is not 0:
#		cut = cutFrame(mask_verde)
#		#frame_campo = mask_verde
#		frame_campo = frame[cut[2]:cut[3], cut[0]:cut[1]]
#		mostra = cv2.bitwise_and(frame,frame,mask=mask_verde)
#		cv2.imshow('Frame Cortado Grama',mostra)
#		frame_b, x, y, raio = detectBall.searchball(frame)
		BallStatus(x,y)
		if args2.visionball:
			cv2.circle(frame, (x, y), raio, (0, 255, 0), 4)
			cv2.imshow('Frame Deteccao',frame)
#===============================================================================
		print "tempo de varredura = ", time.time() - start


#===============================================================================

#		if args2.withoutservo == False:
#			posheadball = head.mov(positionballframe,posheadball,Mem, bkb)
	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
#	raw_input("Pressione enter pra continuar")

#	if args2.withoutservo == False:
#		head.finalize()
#	ball.finalize()
	cv2.destroyAllWindows()
	cap.release()
