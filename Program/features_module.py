import numpy as np
import pandas as pd 
import math
import cv2
from scipy import ndimage
from scipy.spatial import distance
from skimage.measure import find_contours
#import matplotlib.pyplot as plt

##########################################################################################
#                               GET INFO
##########################################################################################
# Se definen los nombres y cantidad de las caracteristicas que calcularan
##########################################################################################

def get_info():

	names_features = ["Comet area",
							"Comet length",
							"Comet DNA content",
							"Comet average intensity",
							"Head area",
							"Head diameter",
							"Head DNA content",
							"Head average intensity",
							"Head DNA%",
							"Tail area",
							"Tail length",
							"Tail DNA content",
							"Tail average intensity",
							"Tail DNA%",
							"Tail moment",
							"Olive moment",
							]
						    
	num_features = len(names_features)
	
	return names_features, num_features;

##########################################################################################
#                                EXTRACT
##########################################################################################
# Obtiene las caracteristicas definidas para cada cometa indicado en label y las guarda
# en un dataframe.
# Obtiene puntos de cominezo y final de las colas de los cometas.
##########################################################################################
	
def extract(img,label, classes,name_image):
	
	dna = img
	comets = label
	tails = (classes == 1)
	heads = (classes == 2)
	num_comets = comets.max()
	angles, angles_median = get_orientation(comets,classes)
	
	#Se obtiene el nombre y cantidad de caracteristicas
	names_features, num_features = get_info()
	
	#Se obtienen los angulos que corrigen la orientacion de los cometas
	angles,angles_median = get_orientation(comets,classes)
	
	#Se obtiene las caracteristicas para cada cometa
	values = np.zeros((num_comets,num_features))
	start_tail = []
	end_tail = []
	start_head = []
	end_head = []
	for i in range(1,num_comets+1):
		comet = (comets==i)
		head = np.logical_and(comet,heads)
		tail = np.logical_and(comet,tails)
		angle = angles[i-1]
		angle_median = angles_median[i-1]
		values[i-1,:],st,et,sh,eh = calculate(dna,comet,head,tail,angle,angle_median)
		start_tail.append(st)
		end_tail.append(et)
		start_head.append(sh)
		end_head.append(eh)
	
	#Se crea dataframe con las caracteristicas de los comeras	
	values = np.round(values,4)	
	rows = np.arange(1,num_comets+1)
	df_comets_features = pd.DataFrame(values, index=rows,columns = names_features) 
	df_comets_features.insert(0, "Image name", name_image, True)
	
	return df_comets_features,start_tail,end_tail,start_head,end_head
	
##########################################################################################
#                               CALCULATE
##########################################################################################
# Calcula cada una de las caracteristicas para un solo cometa de entrada.
# Retorna todas las caracteristicas de un cometa y el inicio y final de su cola.
##########################################################################################

def calculate(dna,comet, head, tail, angle, angle_median):

	names_features, num_features = get_info()
	values= np.zeros((num_features))
	
	############# COMET ##################
	
	#Comet Area
	values[0] = area(comet)
	
	#Comet Length
	values[1],dummy,dummy2 = comet_length(comet,head,angle)
	
	#Comet DNA content
	values[2] = DNA_content(dna,comet)
	
	#Comet average intensity
	values[3] = values[2] / values[0]
	
	############# HEAD ####################
	
	start_head = (0,0)
	end_head = (0,0)
	
	if(np.sum(head)>0):
	
		#Head Area
		values[4] = area(head)
		
		#Head diameter
		values[5],start_head,end_head = head_diameter(head,angle)
		
		#Head DNA content
		values[6] = DNA_content(dna,head)
		
		#Head average intensity
		values[7] = values[6] / values[4]
		
		#Head DNA%
		values[8] = (values[6] / values[2])*100
		
		#Points start end Head
		#dummy,start_head2,end_head2 = head_diameter(head,angle_median)
	
	############# TAIL ##################
	
	start_tail = (0,0)
	end_tail = (0,0)
	
	if(np.sum(tail)>0):
	
		#Tail Area
		values[9] = area(tail)
		
		#Tail length
		values[10],start_tail,end_tail = tail_length(head,tail,angle)
		
		#Tail DNA content
		values[11] = DNA_content(dna,tail)
		
		#Tail average intensity
		values[12] = values[11] / values[9]
		
		#Tail DNA%
		values[13] = (values[11] / values[2])*100
		
		#Points start end Tail
		#dummy, start_tail2, end_tail2 = tail_length(head,tail,angle_median)
		
	############# MOMENT ##################
	
	#Tail moment
	values[14] = values[10]*(values[13]/100)
	
	#Olive moment
	values[15] = olive_moment(values[13],head,tail)


	return values,start_tail,end_tail,start_head,end_head
	
##########################################################################################
#                               AREA
##########################################################################################
# Calcula el area de una region
##########################################################################################

def area(data):
	
	return np.sum(data)
	
##########################################################################################
#                               DNA CONTENT
##########################################################################################
# Suma de las intensidades del cometa, rango de 0-255 por pixel
##########################################################################################

def DNA_content(dna,obj):

	data = dna*(1*obj)
	return np.sum(data)
	
##########################################################################################
#                               HEAD DIAMETER
##########################################################################################
# Calcula el diametro de la cabeza del cometa con su orientacion corregida
##########################################################################################

def head_diameter(head,angle):

	#Rota cabeza del cometa, si es necesario
	if(angle == 0):
		head_r = head
	else:
		head_r = ndimage.rotate(head*10,angle,reshape=True)
		head_r = head_r > 0
	
	#Obtiene fila que pasa por el centroide de la cabeza
	x,y = centroid(head_r)
	signal = head_r[round(y),:]
	
	#Obtiene inicio y fin de la cabeza
	idx_head = np.where(signal)
	start = np.array([idx_head[0][0],round(y)])
	end = np.array([idx_head[0][-1],round(y)])
	
	#Obtiene el diametro
	diameter = distance.euclidean(start,end)
	
	#Se hallan las coordenadas del centroide y final de la cabeza
	org_center = (np.array(head.shape[::-1])-1)/2.
	rot_center = (np.array(head_r.shape[::-1])-1)/2.
	a = np.deg2rad(-angle) 
	start = np.array([x,y]) - rot_center
	start = np.array([start[0]*np.cos(a) + start[1]*np.sin(a),-start[0]*np.sin(a) + start[1]*np.cos(a)])
	start = tuple(np.uint(start+org_center))        
	end = end - rot_center
	end = np.array([end[0]*np.cos(a) + end[1]*np.sin(a),-end[0]*np.sin(a) + end[1]*np.cos(a) ])
	end = tuple(np.uint(end+org_center))

	return diameter,start,end

##########################################################################################
#                               COMET LENGTH
##########################################################################################
# Cacula el largo del cometa y obtiene sus puntos de inicio y fin.
##########################################################################################

def comet_length(comet,head,angle):

	#Rota cometa de acuerdo a su orientacion
	if(angle == 0):
		comet_r = comet
	else:
		comet_r = ndimage.rotate(comet*10,angle,reshape=True)
		comet_r = comet_r > 0
	
	if(np.sum(head)==0):
		#Obtiene fila que pasa por el centroide del cometa
		x,y = centroid(comet_r)
	else:
		#Obtiene fila que pasa por el centroide de la cabeza
		head_r = ndimage.rotate(head*10,angle,reshape=True)
		x,y = centroid(head_r)
		
	signal = comet_r[round(y),:]
	
	#Se halla comienzo y final del cometa
	idx_comet = np.where(signal)
	start = np.array([idx_comet[0][0],round(y)])
	end = np.array([idx_comet[0][-1],round(y)])

	#Se calcula la distancia del cometa
	length = distance.euclidean(start,end)

	#Se hallan las coordenadas del comienzo y final del cometa
	org_center = (np.array(comet.shape[::-1])-1)/2.
	rot_center = (np.array(comet_r.shape[::-1])-1)/2.
	a = np.deg2rad(-angle) 
	start = start - rot_center
	start = np.array([start[0]*np.cos(a) + start[1]*np.sin(a),-start[0]*np.sin(a) + start[1]*np.cos(a)])
	start = tuple(np.uint(start+org_center))        
	end = end - rot_center
	end = np.array([end[0]*np.cos(a) + end[1]*np.sin(a),-end[0]*np.sin(a) + end[1]*np.cos(a) ])
	end = tuple(np.uint(end+org_center))
            
	return length,start,end
	
##########################################################################################
#                               TAIL LENGTH
##########################################################################################
# Cacula el largo de la cola del cometa y obtiene sus puntos de inicio y fin.
##########################################################################################

def tail_length(head,tail,angle):

	if(area(tail)==0):
		x,y = centroid(head)
		return 0,x,y
	
	#Si no tiene cabeza, se usa la funcion de largo del cometa
	comet = 2*head + 1*tail
	if(area(head)==0): 
		lenght,start,end = comet_length(comet>0,head,angle)
		return lenght,start,end
		
	#Se rota el cometa de acuerdo a su orientacion
	if(angle == 0):
		comet_r = comet
	else:
		comet_r = ndimage.rotate(comet,angle,reshape=True)
		
	#Se obtiene fila que pasa por el centroide del cometa
	x,y = centroid(comet_r>1)
	signal = comet_r[round(y),:]
	
	#Se halla comienzo y final del cometa
	idx_tail = np.where(signal == 1)
	idx_head = np.where(signal == 2)
	start = np.array([idx_head[0][-1],round(y)])
	if(np.size(idx_tail)==0):
		end = start
	else:
		end = np.array([idx_tail[0][-1],round(y)])
	
	#Se calcula la distancia del cometa
	length = distance.euclidean(start,end)

	#Se hallan las coordenadas del comienzo y final del cometa
	org_center = (np.array(comet.shape[::-1])-1)/2.
	rot_center = (np.array(comet_r.shape[::-1])-1)/2.
	a = np.deg2rad(-angle)   
	start = start - rot_center
	start = np.array([start[0]*np.cos(a) + start[1]*np.sin(a),-start[0]*np.sin(a) + start[1]*np.cos(a)])
	start = tuple(np.uint(start+org_center))       
	end = end - rot_center
	end = np.array([end[0]*np.cos(a) + end[1]*np.sin(a),-end[0]*np.sin(a) + end[1]*np.cos(a) ])
	end = tuple(np.uint(end+org_center))
            
	return length, start, end

##########################################################################################
#                               OLIVE MOMENT
##########################################################################################
# Cacula el momento de olive, definido como el producto de la distancia del centroide
# de la cabeza al centroide de la cola con el contenido de ADN de la cola.
##########################################################################################

def olive_moment(dna_tail,head,tail):

	if(area(head) == 0):
		return 0
		
	if(area(tail) == 0):
		return 0
		
	hx,hy = centroid(head)
	tx,ty = centroid(tail)
	
	d = distance.euclidean([hx,hy],[tx,ty])
	
	olive = (dna_tail/100)*d
	
	return olive

##########################################################################################
#                               CENTROID
##########################################################################################
# Calcula los centroides de una region
##########################################################################################
def centroid(data):

	idx_y,idx_x = np.where(data)
	x = np.mean(idx_x)
	y = np.mean(idx_y)

	return x,y

##########################################################################################
#                               GET ORIENTATION
##########################################################################################
# Primero halla los angulos que corrigen la orientacion de los cometas. Con base en esa
# informacion, se usa una tecnica para mejorar la orientacion de los cometas:
# 1. Promedio + desviacion estandar
# 2. Promedio
# 3. Mediana
##########################################################################################

def get_orientation(comets,classes):

	#Se selecciona metodo para encontrar angulo de cometas con cabeza
	#alignCentroids = False
	#farthestPoint = True

	#Se hallan los angulos que corrigen la orientacion del cometa
	angles = np.zeros(comets.max())
	#comets_w_head = np.zeros(comets.max())
	#for i in range(1,comets.max()+1):
	#	comet = (comets==i)
	#	head = np.logical_and(comet,classes==2)
	#	if(np.sum(head)>0):
	#		if(alignCentroids):
	#			angles[i-1] = align_centroids(head,comet)
	#		if(farthestPoint):
	#			angles[i-1] = farthest_point(head,comet)
	#		comets_w_head[i-1] = 1
	#	else:
	#		angles[i-1] = angle_no_head(comet)
	
	angles_median = np.copy(angles)	
			
	#if( (np.sum(comets_w_head)/comets.max()) >= 0.5) :
	#	angles_median[:] = np.median(angles[comets_w_head == 1])
	#else:
	#	angles_median[:] = np.median(angles[comets_w_head == 0])

	#Se usa la mediana para mejorar la orientacion de los cometas
	#angles_median = np.copy(angles)	
	#angles_median[comets_w_head == 1] =  np.median(angles[comets_w_head == 1])
				

	return angles,angles_median
	
	
##########################################################################################
#                               FARTHEST POINT
##########################################################################################
# Halla el angulo que alinea el centroide de la cabeza con el punto mas lejano contenido
# en el contorno del cometa
##########################################################################################

def farthest_point(head,comet):

	xh,yh = centroid(head)
		
	contour = find_contours(comet,0)
	y = contour[0][:,0]
	x = contour[0][:,1]

	max_d = -1

	for xi,yi in zip(x,y):
		d = distance.euclidean([xh,yh],[xi,yi])
		if d > max_d:
			max_d = d
			end_point = [xi,yi]
			
			
	angle = get_angle(xh,yh,end_point[0],end_point[1])
	
	return angle
	

##########################################################################################
#                               ALIGN CENTROIDS
##########################################################################################
# Halla el angulo que alinea el centroide de la cabeza con el centroide del cometa
##########################################################################################

def align_centroids(head,comet):

	xh,yh = centroid(head)
	xc,yc = centroid(comet)
	angle = get_angle(xh,yh,xc,yc)
	
	return angle
	
	  
##########################################################################################
#                               ANGLE WITHOUT HEAD
##########################################################################################
# Halla el angulo para corregir la orientacion de cometas sin cabeza
##########################################################################################
def angle_no_head(comet):

	xc,yc = centroid(comet)
		
	contour = find_contours(comet,0)
	y = contour[0][:,0]
	x = contour[0][:,1]

	max_d = -1

	for xi,yi in zip(x,y):
		d = distance.euclidean([xc,yc],[xi,yi])
		if d > max_d:
			max_d = d
			end_point = [xi,yi]
			
			
	angle = get_angle(xc,yc,end_point[0],end_point[1])
	
	return angle
	
##########################################################################################
#                               GET ANGLE
##########################################################################################
# Obtiene angulo entre de dos puntos
##########################################################################################
def get_angle(x1,y1,x2,y2):

	x = x2 - x1
	y = y2 - y1
	
	if(x == 0):
		return 90
		
	angle = math.degrees(math.atan(y/x))
	
	if(x>=0) & (y>=0):
		angle = angle
	elif(x<0) & (y>=0):
		angle = angle+180
	elif(x<0) & (y<0):
		angle = angle+180
	elif(x>=0) & (y<0):
		angle = angle

	return angle

