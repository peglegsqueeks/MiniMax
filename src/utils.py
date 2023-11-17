from dataclasses import dataclass
from enum import Enum
import logging
from .animation_manager import AnimationManager, load_images
import numpy as np
import pygame

# All the configs for the robot will be stored in this RobotConfig dataclass. Dataclasses are an 
# easy way to store configuration in Python - expecially useful when we have lots of named parameters
# that we will be using all over our code. They a) make code much more readable, b) make the code
# much easier to modify, and c) make it much easier to pass in arguments from the commandline or
# other places if we want to e.g. modify a single parameter for a single run.
@dataclass
class RobotConfig:
    # serial port params
    # prefix: port_
    port_baudrate: int = 115200

    # navigation params
    # prefix: nav_
    nav_xres: int = 800
    nav_yres: int = 600
    nav_x_deviation: int = 0
    nav_ymax: int = 0

    # TTS params
    # prefix: tts_
    tts_rate: int = 145
    tts_voice: str = 'english+f4'
    tts_volume: float = 1

    # Biscuit Operation Mode params
    # prefic: biscuit_
    biscuit_wait_time: float = 4

    # Animation params
    # prefix: animate_
    animate_delay: float = 0.08

    # Person search params
    # prefix: ps_
    ps_nn_path: str = '/home/pi/depthai-python/examples/models/mobilenet-ssd_openvino_2021.4_6shave.blob'
    ps_full_frame_tracking: bool = True
    ps_xres: int = 800
    ps_yres: int = 600
    ps_tolerance: int = 95
    ps_bottom_buffer: int = 5
    ps_give_biscuit_on_success: bool = True
    
    animations=[]
    print('starting Robot')
    print('starting to load animations into memory')
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
    print('Finished loading animations into memory')


# Enum's are a way to create a type with statically set values. They're an idea stolen from more
# strongly-type languages (e.g. Java) but even though their implementation in Python is reasonably
# bare-bones, they do make help us make code more readable when we have lots of different states
# or types that an object could exist in.
class RobotState(Enum):
    # Special Cases
    EXIT = 0
    IDLE = 1

    # Operational Modes
    PERSON_SEARCH_GIVE_BISCUIT = 2
    PERSON_SEARCH_NO_GIVE_BISCUIT = 3
    CUP_SEARCH_GIVE_BISCUIT = 4
    CUP_SEARCH_NO_GIVE_BISCUIT = 5
    AGEGENDER = 6
    EMOTIONS = 7

lower_case_letters = [
    pygame.K_a,
    pygame.K_b,
    pygame.K_c,
    pygame.K_d,
    pygame.K_e,
    pygame.K_f,
    pygame.K_g,
    pygame.K_h,
    pygame.K_i,
    pygame.K_j,
    pygame.K_k,
    pygame.K_l,
    pygame.K_m,
    pygame.K_n,
    pygame.K_o,
    pygame.K_p,
    pygame.K_q,
    pygame.K_r,
    pygame.K_s,
    pygame.K_t,
    pygame.K_u,
    pygame.K_v,
    pygame.K_w,
    pygame.K_x,
    pygame.K_y,
    pygame.K_z,
]

# InputObject is currently only used to detect keystrokes - if more user input
# is required in the future, it should be added here.
class InputObject:
    def __init__(self) -> None:
        self.pressed_keys = self._get_pressed_keys() 
    
    def _get_pressed_keys(self):
        pressed = []

        #TODO: is this a bottleneck??
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in lower_case_letters:
                    pressed.append(pygame.key.name(event.key))
        
        # logging.debug(pressed)
        
        return pressed


def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)
