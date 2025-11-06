import numpy as np
from skimage import measure
from skimage.segmentation import watershed


##########################################################################################
#                               Partition and Label
##########################################################################################
# Particiona los cometas traslapados con el método de watershed.
# - Se seleccionan las cabezas como marcadores
# - Se selecciona la imagen original en escalas de grises como mapa de distancias
# Se etiquetan los cometas con un numero no repetido
##########################################################################################

def partition_and_label(classes,im):
	
	#Separa cometas en libres y traslapados
	comets_free, comets_overlap = separate(classes)
	
	#Se etiquetan los cometas libres
	label_comets_free,num_comets_free = measure.label(comets_free>0,return_num=True)
	
	#Se aplica el algortimo de watershed en cometas traslapados
	markers = measure.label(comets_overlap == 2)
	dis = -(im*(comets_overlap>0))
	label_comets_overlap = watershed(dis, markers, mask=comets_overlap>0)
	
	#Se mezclan las etiquetas de cometas libres y traslapados
	label_comets_overlap = label_comets_overlap + num_comets_free
	label_comets_overlap[label_comets_overlap==num_comets_free]=0
	label_comets = label_comets_free + label_comets_overlap
	

	return label_comets, num_comets_free

##########################################################################################
#                               Label
##########################################################################################
# Se etiquetan los cometas con un numero no repetido, solo para cometas libres
##########################################################################################

def label(classes):
	
	#Separa cometas en libres y traslapados
	comets_free, comets_overlap = separate(classes)
	
	#Se etiquetan los cometas libres
	label_comets_free, num_comets_free = measure.label(comets_free>0,return_num=True)
	label_comets = label_comets_free

	return label_comets,num_comets_free
	

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

