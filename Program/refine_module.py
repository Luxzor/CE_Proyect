import numpy as np
from skimage import measure
from scipy import ndimage
from skimage.morphology import convex_hull_image

##########################################################################################
#                               REFINE
##########################################################################################
# Mejora el la predicción del modelo de aprendizaje profundo. Técnicas usadas:
# - 1. Eliminancion de cometas con probabilidad menor a 90%
# - 2. Separa cometas con muy poco traslape
# - 3. Elimina cabezas con probabilidad menor a 70% en regiones con dos o mas cabezas
# - 4. Aplica convex hull a cometas que esten libres de traslape y a todas las cabezas
# - 5. Rellena posible huecos dentro de los cometas
# - 6. Elimina areas menores al promedio menos k veces la desviacion estandar
##########################################################################################
	
def refine(predict, classes, pc = 0.9, ph = 0.7):

	classes_refine = np.copy(classes)
	
	comets,k = measure.label(classes>0,return_num=True)
	if(k == 0):
		return classes_refine
		
	classes_refine = remove_comets_prob(classes_refine, predict, pc)
	classes_refine = separate_overlap(classes_refine, predict, pc)
	classes_refine = remove_heads_prob(classes_refine, predict, ph)
	classes_refine = convex_hull(classes_refine)
	classes_refine = fill_holes(classes_refine)
	classes_refine = remove_areas_min(classes_refine)
	
	return classes_refine
	
##########################################################################################
#                               TECNICA 1
##########################################################################################
	
def remove_comets_prob(classes, predict, pc):
	
	#Arreglo que contiene la probabilidad de ser cometa de cada pixel
	#Probabilidad de cola + probabilidad de cabeza
	comets_probability = predict[:,:,1] + predict[:,:,2]   
	
	#Se hallan los cometas contenidos en la imagen etiquetada
	comets,k = measure.label(classes>0,return_num=True)
	
	#Se eliminan los cometas con probabilidad global menor a pc
	prob_comets = []
	mask = np.zeros(classes.shape)
	for i in range(1,k+1):
		comet_probability = np.multiply(comets_probability, comets==i)
		overall_probability = (sum(sum(comet_probability))) / (sum(sum(comets==i)))
		prob_comets.append(overall_probability)
		if(round(overall_probability,2) < pc):
			mask = mask + (comets ==i)
	classes[mask==1] = 0
	
	return classes
	
##########################################################################################
#                               TECNICA 2
##########################################################################################

def separate_overlap(classes, predict, pc):	

	#Arreglo que contiene la probabilidad de ser cometa de cada pixel
	#Probabilidad de cola + probabilidad de cabeza
	comets_probability = predict[:,:,1] + predict[:,:,2]   
	
	#Separa cometas con traslape pequeño
	comets_free, comets_overlap = separate(classes)
	comets_overlap,k = measure.label(comets_overlap>0,return_num=True)
	mask = np.zeros(classes.shape)
	for i in range(1,k+1):
		comets_separate = np.logical_and(comets_overlap==i,comets_probability >= pc)
		label_separate, num_comets = measure.label(comets_separate,return_num=True)
		if(num_comets >= 2):
			mask = mask + comets_separate
		else:
			mask = mask + (comets_overlap == i)
	mask = mask + (comets_free > 0)
	classes = np.multiply(classes,mask)
	
	return classes
	

##########################################################################################
#                               TECNICA 3
##########################################################################################
	
def remove_heads_prob(classes, predict, ph):	
	
	#Arreglo que contiene la probabilidad de ser cabeza de cada pixel
	heads_probability = predict[:,:,2]
	
	#Se obtienen los cometas traslapados
	comets_free, comets_overlap = separate(classes)
	
	#Se hallan los cabezas de los cometas traslapados
	heads,k = measure.label(comets_overlap>1,return_num=True,background=0)
	
	#Se eliminan las cabezas con probabilidad global menor a ph
	#unicamente en los cometas traslapados
	prob_heads = []
	mask = np.zeros(classes.shape)
	for i in range(1,k+1):
		head_probability = np.multiply(heads_probability, heads==i)
		overall_probability = (sum(sum(head_probability))) / (sum(sum(heads==i)))
		prob_heads.append(overall_probability)
		if(overall_probability <= ph):
			mask = mask + (heads ==i)
	classes[mask==1] = 1	
	
	return classes

##########################################################################################
#                               TECNICA 4
##########################################################################################

def convex_hull(classes):
	
	#Se hallan los cometas libres
	comets_free, comets_overlap = separate(classes)
	comets,k = measure.label(comets_free>0,return_num=True)
	
	#Se aplica convex hull a cada cometa libre
	for i in range(1,k+1):
		comet = (comets == i)
		comet_hull = convex_hull_image(comet)
		diff_comet = 1*comet + 2*comet_hull
		classes[diff_comet==2] = 1
	
	#Se hallan las cabezas	
	heads,k = measure.label(classes==2,return_num=True)
	
	#Se aplica convex hull a cada cabeza
	for i in range(1,k+1):
		head = (heads==i)
		head_hull = convex_hull_image(head)
		diff_head = 1*head + 2*head_hull
		classes[diff_head==2] = 2
		
	return classes

##########################################################################################
#                               TECNICA 5
##########################################################################################

def fill_holes(classes):	
	
	comet_filled = ndimage.binary_fill_holes(classes>0).astype(int)
	head_filled = ndimage.binary_fill_holes(classes==2).astype(int)
	
	classes = comet_filled + head_filled
	
	return classes
	
	
##########################################################################################
#                               TECNICA 6
##########################################################################################

def remove_areas_min(classes):	
	
	#Se hallan todas las areas
	comets,k = measure.label(classes>0,return_num=True)
	area_comets = []
	for i in range(1,k+1):
		area = np.sum(comets==i)
		area_comets.append(area)
	
	#Se calcula promedio y std
	area_mean = np.mean(area_comets)
	area_std = np.std(area_comets) 
	
	#Se halla umbral
	for i in range(10,0,-1):
		area_trh = area_mean - i*area_std
		if area_trh >= 0:
			break
	
	#Se eliminan areas menores al umbral
	mask = np.zeros(classes.shape)
	for i in range(1,k+1):
		area = np.sum(comets==i)
		if(area < area_trh):
			mask = mask + (comets ==i)
	classes[mask==1] = 0
	
	return classes


##########################################################################################
#                               SEPARATE
##########################################################################################
# Separa cometas en cometas libres y cometas traslapados
# - Cometas libres - Regiones con una cabeza detectada
# - Cometas traslapados - Regiones con dos o mas cabezas detectadas
##########################################################################################

def separate(classes):
	
	#Se hallan los cometas contenidos en la imagen etiquetada
	comets,k = measure.label(classes>0,return_num=True)
	
	#Se hallan las cabezas de cometas contenidos en la imagen etiquetada
	comets_heads = (classes == 2)
	
	#Se obtienen los cometas traslapados, mas de dos cabezas
	mask = np.zeros(classes.shape)
	for i in range(1,k+1):
		heads = np.logical_and(comets_heads, (comets==i))
		heads,num_heads = measure.label(heads,return_num=True)
		if(num_heads > 1):
			mask = mask + 1*(comets==i)
	comets_overlap = np.multiply(classes,1*mask)
	
	#Se obtienen los cometas libres
	comets_free = classes - comets_overlap
	
	return comets_free, comets_overlap


