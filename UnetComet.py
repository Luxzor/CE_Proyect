import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from Program import segmentation_module
from Program import refine_module
from Program import partition_module
from Program import detect_module
from Program import features_module

class UnetCometAssay():

	feature_comets = pd.DataFrame()

##########################################################################################
#                                   INIT
##########################################################################################
# Al crear una instancia de la clase UNetCometAssay carga el modelo de aprendizaje 
# profundo UNet ya entrenado.
##########################################################################################	

	def __init__(self):
	
		self.model1 = load_model("Program/Model/Model_Unet1.h5")
		self.model2 = load_model("Program/Model/Model_Unet2.h5")
		self.model3 = load_model("Program/Model/Model_Unet3.h5")
		self.model4 = load_model("Program/Model/Model_Unet4.h5")
		self.model5 = load_model("Program/Model/Model_Unet5.h5")
		
##########################################################################################
#                               GET DETECTION
##########################################################################################
# De acuerdo con las opciones recibidas en los parametros, la funcion hace uso de las
# funciones de segmentation.py y el modelo de aprendizaje profundo, para obtener una
# imagen con los cometas detectados y los parametros calculados en un csv.
# string - path_image : Ruta de la imagen
# string - path_output_dir : Ruta de guardado de la imagen de salida+
# boleano - only_free_comets : Opcional, no detecta traslapados
# float - resize : Opcional, redimensiona imagen (factor mayor a cero y menor a uno)
##########################################################################################

	def get_detection(self,path_image,path_output_dir,only_free_comets,resize=1):
		
		#Lee imagen y la convierte en escala de grises
		img = cv2.imread(path_image)
		img = img[:, :, [2, 1, 0]]
		name_image_ext = os.path.basename(path_image)
		name_image,ext = os.path.splitext(name_image_ext)
		img_gray, img_gray_resize = self.get_channel_resize(img,resize)
		
		#Realiza modulo de segmentacion
		prediction, classes = segmentation_module.predict_ensamble(self.model1,self.model2,self.model3,
																			        self.model4,self.model5, img_gray_resize)
		
		#Realiza modulo de refinamiento
		classes  = refine_module.refine(prediction, classes)
		classes = self.restore_size(img_gray,classes,resize)
		
		#Realiza modulo de particion, si se desea
		if (only_free_comets == False):
			label_comets, k  = partition_module.partition_and_label(classes,img_gray)
		else:
			label_comets, k  = partition_module.label(classes)

		#Realiza modulo extraccion de caracteristicas
		feature_comet   = features_module.extract(img_gray,label_comets,classes,name_image)
		
		#Realiza modulo de deteccion
		detect_comets = detect_module.detect(img,label_comets, k, classes,feature_comet)
		
		#Se guardan resultados
		
		#np.save(path_output_dir+"/"+name_image+"_predict.npy",prediction)  #Predicion
		
		#head = np.float32(classes==2)*165
		#tail = np.float32(classes>0)*90
		#imgs=head+tail
		#cv2.imwrite(path_output_dir+"/"+name_image+"_segmentation.png",imgs)  #Segmentacion
		
		self.feature_comets = pd.concat([self.feature_comets,feature_comet[0]])
		self.feature_comets.to_csv(path_output_dir+"/UnetCometAssay_output.csv")  #Caracteristicas
		
		detect_comets = detect_comets[:, :, [2, 1, 0]]
		cv2.imwrite(path_output_dir+"/"+name_image+"_output.png",detect_comets)  #Imagen de salida
		

		return detect_comets
		
		
##########################################################################################
#                               GET CHANNEL RESIZE
##########################################################################################
# Obtiene el canal de imagen RGB con mas informacion
# Reescale si se desea
##########################################################################################
	
		
	def get_channel_resize(self,img,resize):
		
		data_channel = np.zeros(3)
		data_channel[0] = np.sum(img[:,:,0])
		data_channel[1] = np.sum(img[:,:,1])
		data_channel[2] = np.sum(img[:,:,2])
		
		index_channel = data_channel.argmax()
		
		img_gray = img[:,:,index_channel]
		img_gray_resize = np.copy(img_gray)
		
		if( (resize > 0) & (resize < 1) ):
			img_gray_resize = cv2.resize(img_gray, (0,0), fx=resize, fy=resize) 
		
		
		return img_gray,img_gray_resize
		
##########################################################################################
#                               RESTORE SIZE
##########################################################################################
# Restaura el tamaño de la imagen de clases
##########################################################################################

	def restore_size(self,img_gray,classes,resize):
	
		if( (resize > 0) & (resize < 1) ):
			classes =cv2.resize(classes,img_gray.T.shape,interpolation=cv2.INTER_NEAREST)
			
		return classes 
			
		

