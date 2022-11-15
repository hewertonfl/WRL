#from curses import def_prog_mode
from socket import TIPC_CONN_TIMEOUT
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
#import eval2Furo
from PIL import Image
import Perspective
import math
from CompassSensor import py_qmc5883l
import sys
import statistics
from banco_lanca import *
from subprocess import call
import argparse
#import RPi.GPIO as gpio
# gpio.setmode(gpio.BCM)
# gpio.setup(23, gpio.IN, pull_up_down = gpio.PUD_DOWN)

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--cod', type=str, required=True)
parser.add_argument('--usi', type=str, required=True)
parser.add_argument('--vida', type=str, required=True)
parser.add_argument('--site', type=str, required=True)
parser.add_argument('--pais', type=str, required=True)
parser.add_argument('--tipo', type=str, required=True)
# Parse the argument
args = parser.parse_args()
# Print "Hello" + the user input argument

global codigo
codigo = args.cod
global usina
usina = args.usi
global vida
vida = args.vida
global site
site = args.site
global pais
pais = args.pais
global tipo
tipo = args.tipo

print(tipo)

SAVEDIR = "./Images"
SAVEDIRORIG = "./OriginalImages"
#os.chdir("Python/yolactGPU")
#/home/visiontech/Python/yolactCPUFast/OriginalImages

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf

import detectToInterface
#import detectToInterfaceFuro


builder = Gtk.Builder()
builder.add_from_file("/home/visiontech/Python/WRL/Interface/interface1.glade")


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

global N_angleGlob
N_angleGlob = 0
global diametros 
diametros = np.array([0,0,0,0,0,0,0])
dim = (576, 576)
dimint = (576, 576)
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
vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
frame00 = vid.read()[1]

frame00 = cv2.cvtColor(frame00, cv2.COLOR_BGR2RGB)
frame01 = frame00
frame02 = frame00

distancia_sensor = 400

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
        self.mensagem_save = builder.get_object('mensagem_save')


        self.segTab = builder.get_object('segTab') 
        self.camTab = builder.get_object('camTab') 
        self.tabPai = builder.get_object('tabPai') 


    def onDeleteWindow(self, *args):
        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        vid.release()

        process = subprocess.Popen(['python3', '/home/visiontech/Python/WRL/main.py'], stdout=None, stderr=None)
        time.sleep(3)
        Gtk.main_quit(*args)

    def on_PlayButton_clicked(self, *args):
        global ligado
        global detect
        detect = True
        ligado = True


    def on_adjustment1_value_changed(self, *args):
        global dim
        adjustvalue = self.adjustment1.get_value()
        dim = (1920*((adjustvalue-20)/100), 1920*((adjustvalue-20)/100))
    
    def on_button_salvar (self,widget): 
        preencher_diametro("dados",codigo,vida,site,pais,diametros[1],diametros[2],diametros[3],diametros[4],diametros[5],diametros[6],diametros[0],N_angleGlob)
        self.mensagem_save.run()


    def on_button_save_clicked (self,widget):
        self.mensagem_save.hide()
        time.sleep(3)
        window1.destroy()

    def on_button_sair_clicked(self,widget):
        window1.destroy()

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
window1.fullscreen()



sensor = py_qmc5883l.QMC5883L()

sensor.calibration = [[1.035719514715836, 0.05190498196718226, 70.01671545997182], [0.05190498196718229, 1.0754245172266885, 2123.9444832218346], [0.0, 0.0, 1.0]]     


def CompassSensor():
    auxAngle = 0
    for i in range(1):
        sensor.declination = -110
        try:
            auxAngle = auxAngle + sensor.get_bearing()
        except:
            return 505
    N_angle = auxAngle/1
    print(N_angle)
    return N_angle
    
def normalizeAngle(num, lower=0.0, upper=360.0, b=False):
    from math import floor, ceil
    res = num
    if not b:
        if lower >= upper:
            raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                             (lower, upper))

        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if res == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num * 1.0  # Make all numbers float, to be consistent

    return res

def PerspectiveMasks(mascaras, centroFuros):
    mascaras2 = []
    for i in range(len(mascaras)):
        mascarasAux = Perspective.do(mascaras[i], centroFuros)
        mascaras2.append(mascarasAux)
    return mascaras2

def get_mask_diameter(mascaras):
    diamMasks = []
    
    for j in range(len(mascaras)):
        areaPixel=np.count_nonzero(mascaras[j])
        diam=2*(np.sqrt(areaPixel/3.141592653589793))
        diamMasks.append(diam)
        
    return diamMasks

def detectframe(imgToSeg, distancia_sensor):
    global detect
    global frame00
    global frame01
    global frame02
    global detectAux
    global N_angleGlob

    N_angleGlob = CompassSensor()
    if(N_angleGlob==505):
        return

    if detectAux:
            detectAux = False
            angles=[]
            imgSeg, centroFuros, diamMasks, mascaras = detectToInterface.detect(imgToSeg)

# Envia mascara do diam externo para primeira posição do vetor
            result = np.where(diamMasks == np.amax(diamMasks))
            
            diamMasksArray = np.array(diamMasks)
            mascarasArray = np.array(mascaras)
            
            diamExt = diamMasksArray[result[0][0]]
            ExtMask = mascarasArray[result[0][0]]
            
            diamMasksArray = np.delete(diamMasksArray, result[0][0])
            mascarasArray = np.delete(mascarasArray, result[0][0], 0)
            
            diamMasksArray = np.insert(diamMasksArray, 0, diamExt)
            mascarasArray = np.insert(mascarasArray, 0, ExtMask, 0)
            
            mascarasArrayFuro = (np.delete(mascarasArray, 0, 0)).tolist()
            
#Ordena as pontos centrais das detecções dos furos a partir do norte
            if not isinstance(mascaras, int):
                shapeImg = imgSeg.shape
                centroImagem=(shapeImg[1]/2, shapeImg[0]/2)               
                for j in range(len(centroFuros)):
                    deltaY = centroFuros[j][1] - centroImagem[1]
                    deltaX = centroFuros[j][0] - centroImagem[0]

                    angleInDegrees = math.atan(deltaY / deltaX)  * 180 / 3.14
                    if(deltaY>0 and deltaX>0):
                        angleInDegrees = angleInDegrees+0
                    if(deltaY>0 and deltaX<0):
                        angleInDegrees = 180+angleInDegrees
                    if(deltaY<0 and deltaX<0):
                        angleInDegrees = 180+angleInDegrees
                    if(deltaY<0 and deltaX>0):
                        angleInDegrees = angleInDegrees+360
                        
                    angleInDegrees = angleInDegrees*(-1)
                    print(centroFuros[j])
                    print(angleInDegrees)
                    angleInDegrees = angleInDegrees+N_angleGlob-90
                    angleInDegrees = (normalizeAngle(angleInDegrees, 0, 360))
                    angles.append(angleInDegrees)
                    

                for passnum in range(len(centroFuros)-1,0,-1):
                        for i in range(passnum):
                            if angles[i]>angles[i+1]:
                                temp = angles[i]
                                temp2, temp3 = centroFuros[i], mascarasArrayFuro[i]
                                angles[i] = angles[i+1]
                                centroFuros[i],mascarasArrayFuro[i] = centroFuros[i+1], mascarasArrayFuro[i+1]
                                angles[i+1] = temp
                                centroFuros[i+1], mascarasArrayFuro[i+1] = temp2, temp3

                mascarasArray = np.array(mascarasArrayFuro)

                mascarasArray = np.insert(mascarasArray, 0, ExtMask, 0)
                
                imgSeg2 = Perspective.do(imgSeg, centroFuros)
                
                mascaras2 = PerspectiveMasks(mascarasArray, centroFuros)
                
                diamMasks2 = get_mask_diameter(mascaras2)
                
                
                #Adicionar à imagem numero do furo e valor da segmentação
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.5
                font_thickness = 1
                text_color = [255, 255, 255]
                
                diamFurosMed = np.array(diamMasks2)
                
                #Verifica se a segmentação encontrou 5 ou 7 objetos (4-6 furos + diametro externo)
                lenDiamet=(len(diamMasks2))
                if lenDiamet == 7:
                    print(diamMasks2)
                    diamFurosMed = diamFurosMed*0.330718
                elif lenDiamet == 5:
                    print(diamMasks2)
                    diamFurosMed = diamFurosMed*0.19857
                else:
                    return
                print(diamFurosMed)
                for i in range(len(centroFuros)):
                    text_str = '%d' % (i+1)
                    text_str2 = '%.3f' % diamFurosMed[i+1]
                    text_pt = centroFuros[i]
                    text_pt2 = (text_pt[0]-20, text_pt[1]-20)
                    cv2.putText(imgSeg, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(imgSeg, text_str2, text_pt2, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)


                frame00 = cv2.cvtColor(imgSeg, cv2.COLOR_BGR2RGB)
                frame01 = cv2.cvtColor(imgSeg2, cv2.COLOR_BGR2RGB)
                frame02 = cv2.cvtColor(imgToSeg, cv2.COLOR_BGR2RGB)

            
 
            
            #TROCA ABA PARA "IMAGEM SEGMENTADA"
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

            ################################SAVE HERE
            global diametros
            for i in range(len(diamFurosMed)):
                diametros[i]=round(diamFurosMed[i])

            #Save image
            cv2.imwrite("/home/visiontech/Python/WRL/assets/"+str(site)+'-'+str(codigo)+'-'+str(vida)+'-B'+".jpg", imgSeg)
            cv2.imwrite("/home/visiontech/Python/WRL/assets/"+str(site)+'-'+str(codigo)+'-'+str(vida)+'-A'+".jpg", imgToSeg)




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

    frameArray =vid.read()[1]

    #frame03 = cv2.flip(frameArray,1)
    frame03 = frameArray
    #vid.release()

    frame03 = cv2.cvtColor(frame03, cv2.COLOR_BGR2RGB)
    shapeImage = frame03.shape
    shapeInvert = (shapeImage[1], shapeImage[0])
    centroImag=(int(shapeInvert[0]/2), int(shapeInvert[1]/2))
    cor = (100, 255, 100)
    cv2.circle(frame03, centroImag, 350, cor, 3)
    cv2.line(frame03, (centroImag[0], 0), (centroImag[0], shapeInvert[1]), cor, 3)
    cv2.line(frame03, (0, centroImag[1]), (shapeInvert[0], centroImag[1]), cor, 3)
    
    #Imprime o norte na imagem com base no angulo obtido pelo sensor
    N_angle = CompassSensor()
    if(N_angle != 505):
        x1 = int(350 * math.cos(math.radians(N_angle+270)))
        #print(x1)
        y1 = int(350 * math.sin(math.radians(N_angle+90)))
        text_position = (x1+centroImag[0], centroImag[1]-y1)
        cv2.putText(frame03, 'N', text_position, cv2.FONT_HERSHEY_DUPLEX, 2, [255, 255, 255], 2, cv2.LINE_AA)


    frame3 = cv2.resize(frame03, (768, 432), interpolation = cv2.INTER_AREA)      #Resize image to fit in gui as adjustment1 set
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
            imgToSeg = vid.read()[1] 
            vid.release()
            #imgToSeg = cv2.flip(imgToSeg,1)
            #Crop image to ROI
            imgToSeg = imgToSeg[170:910, 590:1330]

            timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
            #folder = os.path.join(SAVEDIRORIG)
            #os.mkdir(folder)
            filename = "%s.jpg" % (timestamp)
            #Save image
            cv2.imwrite("OriginalImages/"+filename, imgToSeg)
            try:
                detectframe(imgToSeg, distancia_sensor)
            except:
                pass
            vid = cv2.VideoCapture(0)
            vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
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
