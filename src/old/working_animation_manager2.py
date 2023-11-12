import time
import os
import pygame
global resx, resy, yoffset, images
resx=1280
resy=800
yoffset=70
images=[]
def load_images(path):
    
    for file_name in os.listdir(path):
        image = pygame.image.load(path + os.sep + file_name).convert()
        images.append(image)
    return images

class AnimationManager:
    def __init__(self, config) -> None:
        pygame.init()

        self.display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.background = pygame.Surface(self.display.get_size())
        self.clock = pygame.time.Clock()
        self.animate_delay = config.animate_delay
    
    def animate(self, ticks: int):
        for _ in range(ticks):
            clock = pygame.time.Clock()
            value = 0
            run = True
            for loops in range(0,len(images)):      
                clock.tick(13) 
                if value >= len(images):
                    value = 0
                    break
                image = images[value]
                y = 20
                x = 20
                # Displaying the image in our game window
                self.display.blit(image, (x, y))
                # Updating the display surface
                pygame.display.update()
                # Filling the window with black color
                #self.display.fill((0, 0, 0))
                self.display.blit(self.background, (0, 0)) 
                # Increasing the value of value variable by 1
                # after every iteration
                value += 4

    def draw_robot_big_mouth(self):
        color=(200,50,50)
        black=(0,0,0)
        
        size1=(int(resx*0.33), int(resy*0.5)+yoffset, int(resx*0.35), int(resy*0.16)+yoffset)
        size2=(int(resx*0.345), int(resy*0.52)+yoffset, int(resx*0.32), int(resy*0.12)+yoffset)
        pygame.draw.ellipse(self.display, color, size1)
        pygame.draw.ellipse(self.display, black, size2)

        pygame.draw.rect(self.display, (50,50,150), pygame.Rect(resx*0.1, resy*0.04+yoffset, resx*0.8, resy*0.70+yoffset),  8) #head outline

        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.3), int(resy*0.22)+yoffset], 80, 4) #outer eye
        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.7), int(resy*0.22)+yoffset], 80, 4)
        pygame.draw.circle(self.display,(5,5,200),[int(resx*0.3), int(resy*0.22)+yoffset], 70, 0) #blue eye
        pygame.draw.circle(self.display,(5,5,200),[int(resx*0.7), int(resy*0.22)+yoffset], 70, 0)
        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.49), int(resy*0.37)+yoffset], 9, 5) #nothrals
        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.51), int(resy*0.37)+yoffset], 9, 5)
        pygame.draw.circle(self.display,(0,0,0),[int(resx*0.3), int(resy*0.22)+yoffset], 29, 0) #iner eye
        pygame.draw.circle(self.display,(0,0,0),[int(resx*0.7), int(resy*0.22)+yoffset], 29, 0)
    
    def draw_robot_small_mouth(self):
        color=(200,50,50)
        black=(0,0,0)
        
        size1 = (int(resx*0.3), int(resy*0.5)+yoffset, int(resx*0.4), int(resy*0.067)+yoffset)
        size2 = (int(resx*0.35), int(resy*0.51+yoffset), int(resx*0.3), int(resy*0.05)+yoffset)

        pygame.draw.ellipse(self.display, color, size1)
        pygame.draw.ellipse(self.display, black, size2)
        pygame.draw.rect(self.display, (50,50,150), pygame.Rect(resx*0.1, resy*0.04+yoffset, resx*0.8, resy*0.70+yoffset),  8) # head outline

        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.3), int(resy*0.22)+yoffset], 80, 4) #outer eye
        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.7), int(resy*0.22)+yoffset], 80, 4)
        pygame.draw.circle(self.display,(5,5,200),[int(resx*0.3), int(resy*0.22)+yoffset], 50, 0) #blue eye
        pygame.draw.circle(self.display,(5,5,200),[int(resx*0.7), int(resy*0.22)+yoffset], 50, 0)
        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.49), int(resy*0.37)+yoffset], 9, 5) #nothrals
        pygame.draw.circle(self.display,(255,0,0),[int(resx*0.51), int(resy*0.37)+yoffset], 9, 5)
        pygame.draw.circle(self.display,(0,0,0),[int(resx*0.3), int(resy*0.22)+yoffset], 22, 0) #inner eye
        pygame.draw.circle(self.display,(0,0,0),[int(resx*0.7), int(resy*0.22)+yoffset], 22, 0)