import tkinter
import os
from tkinter import filedialog
from Program import UnetComet

#Lectura de rutas de imagenes
tkinter.Tk().withdraw()
images_list = filedialog.askopenfilenames()
total_images = len(images_list)

#Lectura de directorio de salida
Output_directory = filedialog.askdirectory()

#Parametros complemnetarios
ofc = False  #Solo cometas intactos (True), todos los cometas (False)
fs =  1      #Factor de escalamiento de imagen

#Creacion del objeto Unet Comet, un solo objeto juntara todas las
#caracteristicas de los cometas de todas las imagenes procesadas.
comet_unet = UnetComet.UnetCometAssay()

#Procesamiento de imagenes
os.system("clear")
print("Procesando imagenes...\n")
for n,image in enumerate(images_list):
  output_dummy = comet_unet.get_detection(image,Output_directory,ofc,fs)
  print("Imagen ", n+1, " de ", total_images)
  
print("\nProceso terminado")  

