import time
import os
import pygame
from .sound_manager import SoundManger
from natsort import natsorted
global resx, resy, yoffset, imp, initPygameOnce
resx=1280
resy=800
yoffset=70
initPygameOnce=0

pygame.init()
display = pygame.display.set_mode((1024, 768))
#display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
#imp = pygame.image.load('/home/pi/MiniMax/powerdown/001.jpg').convert()
def load_images(path):
    store =[]
    q=0
    for file_name in os.listdir(path):
        temp=file_name
        q=q+1
        if q>=2:
            store.append(temp)
            q=0
    store=natsorted(store)  
    images = []
    image_array=[]
    for xyz in store:
        pic=path+xyz
        image_array.append(pic)
    for names in image_array:
        imagine = pygame.image.load(names)
        images.append(imagine)
    store=[]
    imagine=[]
    image_array=[]
    return images



class AnimationManager:
    def __init__(self, config) -> None:
        self.display = pygame.display.set_mode((1024, 768))
        #self.display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.background = pygame.Surface(self.display.get_size())
        #self.clock = pygame.time.Clock()
        self.animate_delay = config.animate_delay
        #self.sound_manager = SoundManger
        self.channel = pygame.mixer.Channel(1)

    def animate(self, images):
        closedmouth = pygame.image.load('/home/pi/MiniMax/Animations/advised/outputFile_001.jpg')
        loops = 0
        run = True
        length=len(images)
        print('starting the loop',length)
        while loops < length:      
            #frt=show_fps()
            #print(loops)
            intloops = int(loops)
            image = images[intloops]
            self.display.blit(image, (0, 0))
            pygame.display.flip()
            loops = loops + 0.75
        #time.sleep(0.05)
        self.display.blit(closedmouth, (0, 0)) # end animation with closed mouth .jpg 
        pygame.display.flip()
        
    def load_animations(self, animations):
        animations=[]
        print('Load Animations')
        temp=load_images('/home/pi/MiniMax/Animations/greetings/')#0
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/seeyou/')#1
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/powerdown/')#2
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/advised/')#3
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/agegender/')#4
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/disagegender/')#5
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/disemotional/')#6
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/emotional/')#7
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/found-person/')#8
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/objective/')#9
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/searchmode-off/')#10
        animations.append(temp)
        print('10 Loaded')
        temp=load_images('/home/pi/MiniMax/Animations/searchterminated/')#11
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/backwards/')#12
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/cautionmovingbackwards/')#13
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/cautionmovingforward/')#14
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/found-you/')#15
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/helpme/')#16
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/inmyway/')#17
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/search-on/')#18
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/search-person/')#19
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/left/')#20
        animations.append(temp)
        print('20 Loaded')
        temp=load_images('/home/pi/MiniMax/Animations/movingback/')#21
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/movingforward/')#22
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/movingleft/')#23
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/movingright/')#24
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/neutral/')#25
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/right/')#26
        animations.append(temp)
        temp=load_images('/home/pi/MiniMax/Animations/forwards/')#27
        animations.append(temp)
        print('27 Loaded')
        return animations
       
