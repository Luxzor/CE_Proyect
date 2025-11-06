import cv2
import numpy as np

##########################################################################################
#                               PREDICT ENSAMBLE
##########################################################################################
# Hace uso de 5 modelos para predecir una imagen completa
##########################################################################################

def predict_ensamble(model1,model2,model3,model4,model5,img):

	predict1, classes1 =  predict(model1,img)
	predict2, classes2 =  predict(model2,img)
	predict3, classes3 =  predict(model3,img)
	predict4, classes4 =  predict(model4,img)
	predict5, classes5 =  predict(model5,img)
	 
	predict_back = np.stack((predict1[:,:,0],predict2[:,:,0],predict3[:,:,0],predict4[:,:,0],predict5[:,:,0]),axis=2)
	predict_back = np.mean(predict_back,axis=2)
	 
	predict_comet = np.stack((predict1[:,:,1],predict2[:,:,1],predict3[:,:,1],predict4[:,:,1],predict5[:,:,1]),axis=2)
	predict_comet = np.mean(predict_comet,axis=2)
	 
	predict_head = np.stack((predict1[:,:,2],predict2[:,:,2],predict3[:,:,2],predict4[:,:,2],predict5[:,:,2]),axis=2)
	predict_head = np.mean(predict_head,axis=2)
	
	predict_ensamble = np.stack((predict_back,predict_comet,predict_head),axis=2)
	classes_ensamble = np.argmax(predict_ensamble,axis=2)
	
	return predict_ensamble, classes_ensamble

	

##########################################################################################
#                               PREDICT
##########################################################################################
# Barre la imagen completa con una ventana de 288*288 que entra al modelo de aprendizaje
# profundo.
##########################################################################################

def predict(model,img):

	window = 128
	add = int((288 - window)/2)
	
	#Crea imagen con bordes espejo
	h,w = img.shape
	add_h = window - h % window
	add_w = window - w % window
	img_mirror = cv2.copyMakeBorder(img, add, add+add_h, add, add+add_w, cv2.BORDER_REFLECT)
	
	#Predice resultados de la imagen con bordes espejo, se usa la estrategia de traslape
	predict = np.zeros((h+add_h,w+add_w,3)) 
	for i in range(int((h + add_h) / window)):
		for j in range(int((w + add_w) / window)):
			img_tmp = img_mirror[window*i:window*i+288,window*j:window*j+288]
			img_tmp = np.reshape(img_tmp,(1,288,288,1))
			img_tmp = img_tmp/255
			result =  model(img_tmp, training=False)
			result = np.reshape(result,(288,288,3))
			predict[window*i:window*(i+1),window*j:window*(j+1),:] = result[add:add+window,add:add+window,:]
	
	#Recorta el arreglo de predicciones a su tamaño original
	predict = predict[0:h,0:w]
	
	#Clasifica pixel de acuerdo a la probabilidad mas alta
	classes = np.argmax(predict,axis=2)
	classes = classes[0:h,0:w]
		
	return predict, classes
