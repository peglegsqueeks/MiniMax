import time
import os
import pygame
from typing import List, Optional
import pathlib
from natsort import natsorted
global images, folder_dir

pygame.init()
display = pygame.display.set_mode((1280, 720))
thesound = pygame.mixer.Sound("/home/pi/MiniMax/DefaultSayings/greetings.wav")
folder_dir = "/home/pi/MiniMax/Results/greetings"



#display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
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

def show_fps():
    "shows the frame rate on the screen"
    fr = str(int(clock.get_fps()))
    frt = font.render(fr, 1, pygame.Color("red"))
    return frt

def play_sequence(images):
    loops = 0
    run = True
    length=len(images)
    print('starting the loop',length)
    while loops < length:      
        #frt=show_fps()
        #print(loops)
        image = images[loops]
        display.blit(image, (0, 0))
        pygame.display.flip()
        loops = loops +2
      
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 30)
animation=load_images('/home/pi/MiniMax/Results/greetings/')
pygame.mixer.Sound.play(thesound)
play_sequence(animation)
display.blit(images[22], (x, y))
pygame.display.update()
pygame.quit()