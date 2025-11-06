import numpy as np
import pandas as pd
import cv2
from skimage import measure
from skimage.segmentation import find_boundaries
from Program import features_module

##########################################################################################
#                               DETECT
##########################################################################################
# Indica mediante contornos las partes del cometa
# Contorno azul: Cometa
# Contorno rojo: Cabeza
# Linea verde: Distancia de la cola
# Numero amarillo: Numero unico asociado al cometa
##########################################################################################
	
def detect(img, label, num_comet_free,classes,feature_comet):

	#Hace la imagen mas grande
	img = cv2.copyMakeBorder(img, 30, 0, 0, 30, cv2.BORDER_CONSTANT)
	label = cv2.copyMakeBorder(label,30, 0, 0, 30, cv2.BORDER_CONSTANT)
	classes = cv2.copyMakeBorder(classes, 30, 0, 0, 30, cv2.BORDER_CONSTANT)
	detect_comets = img

	#Detecta bordes de cabezas y cometas libres
	label_comets_free = np.zeros(label.shape, dtype=np.int8)
	label_comets_free[label<=num_comet_free] = label[label<=num_comet_free]
	boundary_heads = find_boundaries(np.multiply(classes==2,label_comets_free>0))
	boundary_comets = find_boundaries(label_comets_free)
	detect_comets[boundary_heads == 1,:] = [255,0,0]
	detect_comets[boundary_comets == 1,:] = [0,0,255]
	
	#Detecta bordes de cabezas y cometas traslapados
	label_comets_overlap = np.zeros(label.shape, dtype=np.int8)
	label_comets_overlap[label>num_comet_free] = label[label>num_comet_free]
	boundary_heads = find_boundaries(np.multiply(classes==2,label_comets_overlap>0))
	boundary_comets = find_boundaries(label_comets_overlap)
	detect_comets[boundary_heads == 1,:] = [255,0,0]
	detect_comets[boundary_comets == 1,:] = [90,90,90] 
	
	
	#Senala el largo de la cola del cometa y el radio de la cabeza
	start_tail = feature_comet[1]
	end_tail = feature_comet[2]
	start_head = feature_comet[3]
	end_head = feature_comet[4]
	color_tail = (0,255,0)
	color_head = (255,0,128)
	thickness = 2
	for i in range(len(start_tail)):
		start = tuple(map(int,tuple(map(sum, zip(start_tail[i],(0,30))))))
		end = tuple(map(int,tuple(map(sum, zip(end_tail[i],(0,30))))))
		detect_comets = cv2.line(detect_comets,start,end, color_tail, thickness)
		start = tuple(map(int,tuple(map(sum, zip(start_head[i],(0,30))))))
		end = tuple(map(int,tuple(map(sum, zip(end_head[i],(0,30))))))
		detect_comets = cv2.line(detect_comets,start,end, color_head, thickness)
	
	#Asigna centroides
	color_head = (255,0,0)
	color_comet = (0,0,255)
	for i in range(1,label.max()+1):
		xc,yc = features_module.centroid(label==i)
		detect_comets = cv2.circle(detect_comets, (int(xc),int(yc)), radius=2, color=color_comet, thickness=-1)
		if(np.sum(np.multiply(classes==2,label==i)) > 0):
			xh,yh = features_module.centroid(np.multiply(classes==2,label==i))
			detect_comets = cv2.circle(detect_comets, (int(xh),int(yh)), radius=2, color=color_head, thickness=-1)
			
	#Indica grado de daño
	degree_damage = classify_comets(feature_comet[0])
	comets = measure.regionprops(label)
	color = (255,128,0)
	font = cv2.FONT_HERSHEY_SIMPLEX 
	size = .75
	thickness = 2
	for i in range(label.max()):
		org = (comets[i].bbox[3],comets[i].bbox[0])	
		detect_comets = cv2.putText(detect_comets,str(degree_damage[i]),org,font,size,color,thickness)

	#Enumera cometas libre
	label = label_comets_free
	comets = measure.regionprops(label) 
	color = (255,255,0)
	font = cv2.FONT_HERSHEY_SIMPLEX 
	size = .75
	thickness = 2
	for i in range(label.max()):
		org = (comets[i].bbox[3],comets[i].bbox[2])	
		detect_comets = cv2.putText(detect_comets,str(i+1),org,font,size,color,thickness)
		
	#Enumera cometas traslapados
	label = label_comets_overlap-num_comet_free
	label[label<0] = 0
	comets = measure.regionprops(label) 
	color = (90,90,90)
	font = cv2.FONT_HERSHEY_SIMPLEX 
	size = .75
	thickness = 2
	for i in range(label.max()):
		org = (comets[i].bbox[3],comets[i].bbox[2])	
		detect_comets = cv2.putText(detect_comets,str(i+num_comet_free+1),org,font,size,color,thickness)
			
		

	return detect_comets
		
			
##########################################################################################
#                               CLASSIFY COMETS
##########################################################################################
# Calcula grado de daño
##########################################################################################

def classify_comets(feature_comet):

	diameter_head = feature_comet["Head diameter"].values
	tail_lenght = feature_comet["Tail length"].values
	
	proportion = diameter_head / tail_lenght
	
	
	degree_damage = np.zeros(proportion.shape)
	degree_damage[proportion < 0.05] = 5
	degree_damage[np.logical_and(proportion >= 0.05 , proportion < 0.4)] = 4
	degree_damage[np.logical_and(proportion >= 0.4 , proportion < 0.65)] = 3
	degree_damage[np.logical_and(proportion >= 0.65 , proportion < 0.9)] = 2
	degree_damage[proportion >= 0.9]  = 1
	degree_damage = np.uint8(degree_damage)
	
	
	return degree_damage
	

