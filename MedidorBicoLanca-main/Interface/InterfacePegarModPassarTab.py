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
import sensor
import sys
from statistics import mean


SAVEDIR = "./Images/"

os.chdir("Python/yolactCPU")



gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
 
 
builder = Gtk.Builder()
builder.add_from_file("Interface/interface1.glade")


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
    imgSeg=[0,0,0,0,0]
    boxes=[0,0,0,0,0]
    diamFuros=[0,0,0,0,0]
                        
    if detectAux:
                        detectAux = False 
                        angles=[]
                        
                        imgSeg[0], boxes[0], diamFuros[0] = detectToInterface.detect(imgToSeg[0], distancia_sensor)
                        imgSeg[1], boxes[1], diamFuros[1] = detectToInterface.detect(imgToSeg[1], distancia_sensor)
                        imgSeg[2], boxes[2], diamFuros[2] = detectToInterface.detect(imgToSeg[2], distancia_sensor)
                        imgSeg[3], boxes[3], diamFuros[3] = detectToInterface.detect(imgToSeg[3], distancia_sensor)
                        imgSeg[4], boxes[4], diamFuros[4] = detectToInterface.detect(imgToSeg[4], distancia_sensor)
                        if not isinstance(boxes, int):
                            #Cálculo pixel para mm

                            print("Distancia = %.1f mm" % distancia_sensor)
                                        
                            #for i in range(len(diamFuros)):
                            
                                #diamFuros[i] = (0.0009*distancia_sensor+0.0568)*diamFuros[i]

                                

                            frame00 = cv2.cvtColor(imgSeg[0], cv2.COLOR_BGR2RGB)
                            frame01 = cv2.cvtColor(imgSeg[0], cv2.COLOR_BGR2RGB)
                            frame02 = cv2.cvtColor(imgToSeg[0], cv2.COLOR_BGR2RGB)
                            
                            
                            #Verifica se a segmentação encontrou um mínimo de 7 objetos (6 furos + diametro externo)
                            
                            if(len(diamFuros[0][:])>=7):
                                for x in range(len(diamFuros)):
                                    #Leva o maior valor de diâmetro (externo) para a ultima posição do array
                                    for i in range(len(diamFuros[x][:])-1):
                                        if(diamFuros[x][i] > diamFuros[x][i+1]):
                                            diamFuros[x][i], diamFuros[x][i+1] = diamFuros[x][i+1], diamFuros[x][i]
                                
                                diamFuros = [mean(values) for values in zip(*diamFuros)]
                                print(diamFuros)
                                
                                #Diametro dos furos
                                handler.label_diametro.set_text(str(diamFuros[0]))
                                handler.label_diametro1.set_text(str(diamFuros[1]))
                                handler.label_diametro2.set_text(str(diamFuros[2]))
                                handler.label_diametro3.set_text(str(diamFuros[3]))
                                handler.label_diametro4.set_text(str(diamFuros[4]))
                                handler.label_diametro5.set_text(str(diamFuros[5]))
                                
                                handler.label_area.set_text(str(3.14159*(diamFuros[0]/2)**2))
                                handler.label_area1.set_text(str(3.14159*(diamFuros[1]/2)**2))
                                handler.label_area2.set_text(str(3.14159*(diamFuros[2]/2)**2))
                                handler.label_area3.set_text(str(3.14159*(diamFuros[3]/2)**2))
                                handler.label_area4.set_text(str(3.14159*(diamFuros[4]/2)**2))
                                handler.label_area5.set_text(str(3.14159*(diamFuros[5]/2)**2))
                                
                                
                                
                                #Diametro total do bico
                                handler.label_diametro6.set_text(str(diamFuros[len(diamFuros)-1]))
                                
                                
                                timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
                                folder = os.path.join(SAVEDIR, timestamp)
                                os.mkdir(folder)
                                filename = "%s.jpg" % (timestamp)
                                #Save image
                                cv2.imwrite(folder+"/"+filename, imgSeg)
                                #Save image Original
                                cv2.imwrite(folder+"/"+"OriginalImage"+filename, imgToSeg[0])
                                #Save txt
                                f = open(folder+"/"+filename+'.txt',"w+")
                                f.write(str(diamFuros))
                                f.close()
                        
def sensorMean(): 
    VarAux = True
    while (VarAux == True):
        auxSensor = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        index = np.array([])
        #print(index)
        distancia = 0
        for i in range(20):
            auxSensor[i] = sensor.distancia()
            
        distancia = np.mean(auxSensor)
        print(auxSensor)
        print(distancia)
        for i in range(len(auxSensor)):
            if math.sqrt((auxSensor[i] - distancia)**2) > 20:
                
                index = np.insert(index, 0, i)  
                #print(index)
        if np.any(index): 
            index = index.astype(int)
        #    print(index)       
            auxSensor = np.delete(auxSensor, index)
        distancia = np.mean(auxSensor)
        VarAux = math.isnan(distancia)
        distancia=1.0158*distancia+5.1606
        print(distancia)
        print(auxSensor)
    return round(distancia, 2)
                        
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
    print("FrameInit")
    
    handler.tabPai.next_page()  ########################################
    
    #vid = cv2.VideoCapture(0)
    #vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
    #vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    #vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #vid.set(cv2.CAP_PROP_FPS, 30)
    frameArray=[0,0,0,0,0]
    frameArray[0] = vid.read()[1]
    frameArray[1] = vid.read()[1]
    frameArray[2] = vid.read()[1]
    frameArray[3] = vid.read()[1]
    frameArray[4] = vid.read()[1]
    imgToSeg = frameArray
    frame03 = frameArray[0]
    #vid.release()
    
    frame03 = cv2.cvtColor(frame03, cv2.COLOR_BGR2RGB)
    shapeImage = frame03.shape
    shapeInvert = (shapeImage[1], shapeImage[0])
    centroImag=(int(shapeInvert[0]/2), int(shapeInvert[1]/2))
    cor = (100, 255, 100)
    cv2.circle(frame03, centroImag, 300, cor, 3)
    cv2.line(frame03, (centroImag[0], 0), (centroImag[0], shapeInvert[1]), cor, 3)
    cv2.line(frame03, (0, centroImag[1]), (shapeInvert[0], centroImag[1]), cor, 3)
    
    cv2.line(frame03, (shapeInvert[0]-200, shapeInvert[1]-100), (shapeInvert[0]-100, shapeInvert[1]-100), cor, 3)    
    cv2.line(frame03, (shapeInvert[0]-200, shapeInvert[1]-100), (shapeInvert[0]-200, shapeInvert[1]-120), cor, 3)
    cv2.line(frame03, (shapeInvert[0]-100, shapeInvert[1]-100), (shapeInvert[0]-100, shapeInvert[1]-120), cor, 3)
    
    distancia_sensor = sensorMean()
    medLine = ((0.0009*distancia_sensor+0.0568))*(100)
    cv2.putText(frame03, str(round(medLine, 1)), (shapeInvert[0]-190, shapeInvert[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
    
    frame3 = cv2.resize(frame03, dimint, interpolation = cv2.INTER_AREA)      #Resize image to fit in gui as adjustment1 set
    pb3 = GdkPixbuf.Pixbuf.new_from_data(frame3.tobytes(),
                                                            GdkPixbuf.Colorspace.RGB,
                                                            False,
                                                            8,
                                                            frame3.shape[1],
                                                            frame3.shape[0],
                                                            frame3.shape[2]*frame3.shape[1])
    image3.set_from_pixbuf(pb3.copy())


    #distancia_sensor = distancia_sensor-80
    handler.label_distancia.set_text(str(distancia_sensor))
    
    if ligado:
        if detectAux:
            vid.release()
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

