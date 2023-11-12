import time
import pygame
import os
global resx, resy, yoffset
resx=1280
resy=800
yoffset=70

def load_images(path):
    images = []
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
            run = Tru
            for loops in range(0,len(images)):      
                clock.tick(11) 
                if value >= len(images):
                    value = 0
                    break
                image = images[value]
                y = 20
                # Displaying the image in our game window
                window.blit(image, (x, y))
                # Updating the display surface
                pygame.display.update()
                # Filling the window with black color
                window.fill((0, 0, 0))
                # Increasing the value of value variable by 1
                # after every iteration
                value += 4
                
window = pygame.display.set_mode((1280, 800))
pygame.init()
print("set mode")
images=load_images('/home/pi/MiniMax/AnimatedFace')
print("loaded")
time.sleep(1)
print('go go go go')

self.animate(1)
pygame.quit()

