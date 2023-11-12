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

#imp = pygame.image.load('/home/pi/MiniMax/powerdown/001.jpg').convert()
def load_images(path):
    store =[]
    for file_name in os.listdir(path):
        temp=file_name
        store.append(temp)
    store=natsorted(store)  
    images = []
    image_array=[]
    for xyz in store:
        pic=path+xyz
        image_array.append(pic)
    for names in image_array:
        imagine = pygame.image.load(names).convert()
        images.append(imagine)
    return images
    
class AnimationManager:
    def __init__(self, config) -> None:
        pygame.init()
        self.display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.background = pygame.Surface(self.display.get_size())
        #self.clock = pygame.time.Clock()
        self.animate_delay = config.animate_delay
        #self.sound_manager = SoundManger
        self.channel = pygame.mixer.Channel(1)

    def animate(self, images):
        closedmouth = pygame.image.load('/home/pi/MiniMax/Animations/advised/outputFile_001.jpg').convert()
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
            loops = loops +1.25
        #time.sleep(0.05)
        self.display.blit(closedmouth, (0, 0)) # end animation with closed mouth .jpg 
        pygame.display.flip()
       