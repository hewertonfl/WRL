import cv2
import time
import numpy as np
import gi
import time, datetime
import os
import subprocess 
import sys
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter
import matplotlib.dates as mdates
import eval2Furo
import detectToInterface
import detectToInterfaceFuro
import Perspective
import math
#import sensor
import sys
import statistics
#import RPi.GPIO as gpio
##gpio.setup(23, gpio.IN, pull_up_down = gpio.PUD_DOWN)

os.chdir("C:/Users/evirt/OneDrive/Área de Trabalho/MedidorBicoLanca-main")

SAVEDIR = "Python/yolactCPUFast/Images"
SAVEDIRORIG = "Python/yolactCPUFast/OriginalImages"
#os.chdir("Python/yolactGPU")
#/home/visiontech/Python/yolactCPUFast/OriginalImages

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
 
 
builder = Gtk.Builder()
builder.add_from_file("interface/interface1.glade")


fig = plt.figure()

plt.style.use('seaborn-dark')
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#353c4a'  # bluish dark grey
for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey

fig.patch.set_facecolor('#353c4a')
ax = fig.add_subplot()


colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41',  # matrix green
    '#353c4a',  # NordicDark
]



dim = (768, 432)
dimint = (768, 432)
ligado = False
detect = False
detectAux = False
largMedida = []
largMedidaMed = []
localtime = []
aAuxiliar = []
aux=400
aux2=0
soundAux = 0
i = 1
frame1=np.array([])
frame00=np.array([])
frame01=np.array([])
frame02=np.array([])
frame03=np.array([])

vid = cv2.VideoCapture(0)                         
vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
frame00 = vid.read()[1]
#vid.release()

frame00 = cv2.cvtColor(frame00, cv2.COLOR_BGR2RGB)
frame01 = frame00
frame02 = frame00

distancia_sensor = 400    

#processo2 = subprocess.Popen('start  python real_time_detection.py', shell=True)
#time.sleep(5)


class Handler():
    def __init__(self):
        self.usar_estilo()
        self.label_diametro = builder.get_object('label_diametro')       
        self.label_diametro1 = builder.get_object('label_diametro1')
        self.label_diametro2 = builder.get_object('label_diametro2')
        self.label_diametro3 = builder.get_object('label_diametro3')
        self.label_diametro4 = builder.get_object('label_diametro4') 
        self.label_diametro5 = builder.get_object('label_diametro5')
        self.label_diametro6 = builder.get_object('label_diametro6')      
        self.label_area = builder.get_object('label_area')
        self.label_area1 = builder.get_object('label_area1')
        self.label_area2 = builder.get_object('label_area2')
        self.label_area3 = builder.get_object('label_area3') 
        self.label_area4 = builder.get_object('label_area4')
        self.label_area5 = builder.get_object('label_area5')
        self.label_distancia = builder.get_object('label_distancia')
        self.adjustment1 = builder.get_object('adjustment1') 
        
        self.segTab = builder.get_object('segTab') ########################################
        self.camTab = builder.get_object('camTab') ########################################
        self.tabPai = builder.get_object('tabPai') ########################################
        
        
    def onDeleteWindow(self, *args):
        vid = cv2.VideoCapture(0)  
        vid.release()
        Gtk.main_quit(*args)

    def on_PlayButton_clicked(self, *args):
        global ligado
        global detect
        detect = True
        ligado = True
        
       
    def on_adjustment1_value_changed(self, *args):
        global dim
        adjustvalue = self.adjustment1.get_value()
        dim = (1920*((adjustvalue)/100), 1080*((adjustvalue)/100)) 
        
    def usar_estilo(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path('Interface/cssStyle.css')
        screen = Gdk.Screen()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(screen.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        

builder.connect_signals(Handler())
handler = Handler()
window1 = builder.get_object("window1")
image = builder.get_object("image")
image1 = builder.get_object("image1")
image2 = builder.get_object("image2")
image3 = builder.get_object("image3")
plot1 = builder.get_object("plot1")
window1.show_all()


def detectframe(imgToSeg, distancia_sensor):
    global detect
    global frame00
    global frame01
    global frame02
    global detectAux
    imgSeg = [0] 
    boxes = [0]
    diamFuros = [0]
    sizeCentroFuros = [0]
    sizeDiamFuros = [0]
    
    if detectAux:
            detectAux = False 
            angles=[]
            img_numpy0, boxes0, centroFuros = detectToInterfaceFuro.detect(imgToSeg)
#VERIFICA SE O TAMANHO DOS ARRAYS É IGUAL, CASO NÃO SEJA RETIRAR O DIFERENTE. (Verifica se em todas as imagens a mesma quantidade de objetos foi encontrada)
            for y in range(len(img_numpy0)):
                if not isinstance(boxes0[y], int):
                    sizeCentroFuros[y]=len(centroFuros[y])
                    if(sizeCentroFuros[y]<=4):
                        return
            moda=statistics.mode(sizeCentroFuros)
            y = 0
            auxy = len(img_numpy0)
            while (y < auxy):
                if not isinstance(boxes0[y], int):
                    if(sizeCentroFuros[y]!=moda):
                        print("Deletando imagem")
                        img_numpy0.pop(y)
                        boxes0.pop(y)
                        boxes.pop(y)
                        centroFuros.pop(y)
                        diamFuros.pop(y)
                        sizeCentroFuros.pop(y)
                        sizeDiamFuros.pop(y)
                        imgSeg.pop(y)
                        auxy-=1
                        y-=1
                y+=1
            print("Quantidade de imagens segmentadas: ", len(sizeDiamFuros))
            if(len(centroFuros)==0):
                return

#Ordena as pontos centrais das detecções dos furos
            for x in range(len(img_numpy0)):

                if not isinstance(boxes0[x], int):
                    shapeImg = img_numpy0[x].shape
                    centroImagem=(shapeImg[1]/2, shapeImg[0]/2)
                    for j in range(len(centroFuros[x])):
                        deltaY = centroFuros[x][j][1] - centroImagem[1]
                        deltaX = centroFuros[x][j][0] - centroImagem[0]

                        angleInDegrees = math.atan(deltaY / deltaX)  * 180 / 3.14
                        if(deltaY>0 and deltaX>0):
                            angleInDegrees = angleInDegrees+0
                        if(deltaY>0 and deltaX<0):
                            angleInDegrees = 180+angleInDegrees
                        if(deltaY<0 and deltaX<0):
                            angleInDegrees = 180+angleInDegrees
                        if(deltaY<0 and deltaX>0):
                            angleInDegrees = angleInDegrees+360
                            
                        angles.append(angleInDegrees)

                    
                    for passnum in range(len(centroFuros[x])-1,0,-1):
                            for i in range(passnum):
                                if angles[i]>angles[i+1]:
                                    temp = angles[i]
                                    temp2 = centroFuros[x][i]
                                    angles[i] = angles[i+1]
                                    centroFuros[x][i] = centroFuros[x][i+1]
                                    angles[i+1] = temp
                                    centroFuros[x][i+1] = temp2

#TMODIFICAR PARA RODAR O EVAL APENAS UMA VEZ, ASSIM  COMO FEITO NO EVALFURO.
                    #Perspectiva e segmentação da imagem
                    imgToSeg2 = Perspective.do(img_numpy0[x], centroFuros[0])
                    print(x)
                    imgSeg[x], boxes[x], diamFuros[x] = detectToInterface.detect(imgToSeg2, distancia_sensor)
                    print(centroFuros[x])

                    if not isinstance(boxes, int):

                        frame00 = cv2.cvtColor(imgSeg[0], cv2.COLOR_BGR2RGB)
                        frame01 = cv2.cvtColor(imgToSeg2, cv2.COLOR_BGR2RGB)
                        frame02 = cv2.cvtColor(img_numpy0[x], cv2.COLOR_BGR2RGB)
                        
                        
                        #Verifica se a segmentação encontrou um mínimo de 6 objetos (5 furos + diametro externo)
                        lenDiamet=(len(diamFuros))
                        
                        #Ordena em ordem crescente.
                        diamFuros[x].sort(reverse = True)
                        #if((lenDiamet)>=6):
                            
                            #Leva o maior valor de diâmetro (externo) para a ultima posição do array
                            #for i in range(len(diamFuros)-1):
                            #    if(diamFuros[x][i] > diamFuros[x][i+1]):
                            #        diamFuros[x][i], diamFuros[x][i+1] = diamFuros[x][i+1], diamFuros[x][i]
                        
#VERIFICA SE O TAMANHO DOS ARRAYS É IGUAL, CASO NÃO SEJA RETIRAR O DIFERENTE. (Verifica se em todas as imagens a mesma quantidade de objetos foi encontrada)
            for y in range(len(imgSeg)):
                if not isinstance(boxes[y], int):
                    sizeDiamFuros[y]=len(diamFuros[y])
            moda=statistics.mode(sizeDiamFuros)
            y = 0
            auxy = len(imgSeg)
            while (y < auxy):
                if not isinstance(boxes[y], int):
                    print(sizeDiamFuros[y])
                    if(sizeDiamFuros[y]!=moda):
                        print("Deletando imagem")
                        imgSeg.pop(y)
                        boxes.pop(y)
                        diamFuros.pop(y)
                        sizeDiamFuros.pop(y)
                        auxy-=1
                        y-=1
                y+=1
            print("Quantidade de imagens segmentadas: ", len(diamFuros))
            if(len(diamFuros)==0):
                return
                
                
            print(diamFuros)
            arr = np.array(diamFuros)
            diamFurosMed = np.average(arr, axis=0)
            print(diamFurosMed)
            if(diamFuros[0]==0):
                return
            #TROCAR ABA PARA "IMAGEM SEGMENTADA"
            
            handler.tabPai.set_current_page(1)
            
#TODO#>MODIFICAR PARA IMPRIMIR COMO DIAMETRO NA INTERFACE APENAS A QUANTIDADE DE FUROS DO DISCO ATUAL.
            #Diametro dos furos
            
            if(len(diamFurosMed)>1):
                handler.label_diametro.set_text(str(diamFurosMed[1]))
                handler.label_area.set_text(str(3.14159*(diamFurosMed[1]/2)**2))
            else:
                return
            if(len(diamFurosMed)>2):
                handler.label_diametro1.set_text(str(diamFurosMed[2]))
                handler.label_area1.set_text(str(3.14159*(diamFurosMed[2]/2)**2))
            if(len(diamFurosMed)>3):
                handler.label_diametro2.set_text(str(diamFurosMed[3]))
                handler.label_area2.set_text(str(3.14159*(diamFurosMed[3]/2)**2))
            if(len(diamFurosMed)>4):
                handler.label_diametro3.set_text(str(diamFurosMed[4]))
                handler.label_area3.set_text(str(3.14159*(diamFurosMed[4]/2)**2))
            
            if(len(diamFurosMed)>5):
                handler.label_diametro4.set_text(str(diamFurosMed[5]))
                handler.label_area4.set_text(str(3.14159*(diamFurosMed[5]/2)**2))
            if(len(diamFurosMed)>6):
                handler.label_diametro5.set_text(str(diamFurosMed[6]))
                handler.label_area5.set_text(str(3.14159*(diamFurosMed[6]/2)**2)) 
            
           
            #Diametro total do bico
            handler.label_diametro6.set_text(str(diamFurosMed[0]))
                        
                        
            timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
             
            filename = "%s.jpg" % (timestamp)
            #Save image
            cv2.imwrite("Images/"+filename, imgSeg[0])
            #Save image Original
            #cv2.imwrite(folder+"/"+"OriginalImage"+filename, img_numpy0[0])
            #Save txt
            f = open("Images/"+filename+'.txt',"w+")
            f.write(str(diamFurosMed))
            f.close()
              

                        
def show_frame(*args):
    global detect
    global ligado
    global frame00
    global frame01
    global frame02
    global detectAux
    global vid
    t0=(time.time())
    auxa=dim[0]
    auxb=dim[1]
    auxa=int(auxa)
    auxb=int(auxb)
    dimint=(auxa, auxb)
    
    page = handler.tabPai.get_current_page()
    if(gpio.input(23)==1 and page==0):
        detect = True
        ligado = True
        print("Click Seg.")
        time.sleep(1)
    elif(gpio.input(23)==1 and page!=0):
        print("Click trocar aba") 
        handler.tabPai.next_page()
        time.sleep(1)
        if(page==3):
            handler.tabPai.set_current_page(0)
 
    #vid = cv2.VideoCapture(0)
    #vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
    #vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    #vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #vid.set(cv2.CAP_PROP_FPS, 30)
    
    frameArray=[0,0,0,0,0]
    frameArray[0] = vid.read()[1]

    frame03 = frameArray[0]
    #vid.release()
    
    frame03 = cv2.cvtColor(frame03, cv2.COLOR_BGR2RGB)
    shapeImage = frame03.shape
    shapeInvert = (shapeImage[1], shapeImage[0])
    centroImag=(int(shapeInvert[0]/2), int(shapeInvert[1]/2))
    cor = (100, 255, 100)
    cv2.circle(frame03, centroImag, 200, cor, 3)
    cv2.line(frame03, (centroImag[0], 0), (centroImag[0], shapeInvert[1]), cor, 3)
    cv2.line(frame03, (0, centroImag[1]), (shapeInvert[0], centroImag[1]), cor, 3)
    
    
    frame3 = cv2.resize(frame03, dimint, interpolation = cv2.INTER_AREA)      #Resize image to fit in gui as adjustment1 set
    pb3 = GdkPixbuf.Pixbuf.new_from_data(frame3.tobytes(),
                                                            GdkPixbuf.Colorspace.RGB,
                                                            False,
                                                            8,
                                                            frame3.shape[1],
                                                            frame3.shape[0],
                                                            frame3.shape[2]*frame3.shape[1])
    image3.set_from_pixbuf(pb3.copy())


    if ligado:
        if detectAux:
            frameArray[1] = vid.read()[1]
            frameArray[2] = vid.read()[1]
            frameArray[3] = vid.read()[1]
            frameArray[4] = vid.read()[1]
            imgToSeg = frameArray
            vid.release()
            
            
            timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
            #folder = os.path.join(SAVEDIRORIG)
            #os.mkdir(folder)
            filename = "%s.jpg" % (timestamp)
            #Save image
            cv2.imwrite("OriginalImages/"+filename, frameArray[0])
            
            detectframe(imgToSeg, distancia_sensor)
            vid = cv2.VideoCapture(0)                        
            vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
            vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
        if(detect):
            detect = False
            detectAux = True
            frame = cv2.imread("loading.jpg")
    
            loadingImg = GdkPixbuf.Pixbuf.new_from_data(frame.tobytes(),
                                                                GdkPixbuf.Colorspace.RGB,
                                                                False,
                                                                8,
                                                                frame.shape[1],
                                                                frame.shape[0],
                                                                frame.shape[2]*frame.shape[1])
            image3.set_from_pixbuf(loadingImg.copy())
        
        
            
        frame = cv2.resize(frame00, dimint, interpolation = cv2.INTER_AREA)      #Resize image to fit in gui as adjustment1 set
        frame1 = cv2.resize(frame01, dimint, interpolation = cv2.INTER_AREA)      #Resize image to fit in gui as adjustment1 set
        frame2 = cv2.resize(frame02, dimint, interpolation = cv2.INTER_AREA)      #Resize image to fit in gui as adjustment1 set

        pb = GdkPixbuf.Pixbuf.new_from_data(frame.tobytes(),
                                                            GdkPixbuf.Colorspace.RGB,
                                                            False,
                                                            8,
                                                            frame.shape[1],
                                                            frame.shape[0],
                                                            frame.shape[2]*frame.shape[1])
        image.set_from_pixbuf(pb.copy())
        
        pb1 = GdkPixbuf.Pixbuf.new_from_data(frame1.tobytes(),
                                                            GdkPixbuf.Colorspace.RGB,
                                                            False,
                                                            8,
                                                            frame1.shape[1],
                                                            frame1.shape[0],
                                                            frame1.shape[2]*frame1.shape[1])
        image1.set_from_pixbuf(pb1.copy())
        
        pb2 = GdkPixbuf.Pixbuf.new_from_data(frame2.tobytes(),
                                                            GdkPixbuf.Colorspace.RGB,
                                                            False,
                                                            8,
                                                            frame2.shape[1],
                                                            frame2.shape[0],
                                                            frame2.shape[2]*frame2.shape[1])
        image2.set_from_pixbuf(pb2.copy())

    return True    
    



GLib.idle_add(show_frame)

Gtk.main()

