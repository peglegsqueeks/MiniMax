# built-in imports
import time
import logging
import random
import subprocess
import serial
import subprocess
import os
import numpy as np
global animations, images, imp, display
import pygame
# local imports
from .operational_modes import AgeGenderOperationalMode, EmotionOperationalMode, ObjectSearchOperationMode
from .sound_manager import SoundManger
from .animation_manager import AnimationManager, load_images
from .tts_manager import TTSManager
from .utils import RobotConfig, RobotState, InputObject

sayings=[]
sayings.append("better")
#sayings.append("compute")
sayings.append("cross")
sayings.append("danger")
sayings.append("directive")
sayings.append("humans")
#sayings.append("cautionroguerobots")
sayings.append("chess")
#sayings.append("dangerwillrobinson")
sayings.append("malfunction2")
sayings.append("nicesoftware2")
#sayings.append("no5alive")
sayings.append("program")
sayings.append("selfdestruct")
sayings.append("shallweplayagame")
sayings.append("silly")
sayings.append("stare")
sayings.append("world")
#sayings.append("comewithme")
sayings.append("gosomewhere2")
#sayings.append("hairybaby")
sayings.append("lowbattery")
sayings.append("robotnotoffended")
sayings.append("satisfiedwithmycare")
sayings.append("waitbeforeswim")
sayings.append("helpme")

#closedmouth = pygame.image.load('/home/pi/MiniMax/Animations/advised/outputFile_001.jpg').convert()
# The Robot class is the main driver of the entire application. It stores the main run loop, all operational
# states (which control the behaviour of the robot), and all application managers which control sounds, TTS,
# animation, and engine control via the serial port. 
#
# The aim of a main driver class like this is to keep things as flexible as possible, with most application
# code abstracted away. For example, say we wanted to add a new behavior to the robot - we have a new depth AI
# feature we want implemented. In that case, we would want to create a new OperationalMode and add it to the list
# of OperationMode's stores in the self.operation_modes list created in the Robot initialiser, as opposed to 
# e.g. create a new function called new_depthai_feature() in the robot class. This way we can create as many
# new features as we want to the application without pollution this main driver class. The end product should
# be one where the different features stay out of each other's way, and any issues are much easier to test and 
# debug.
def playvideo(animatedvideo):
    subprocess.call(['cvlc', '--fullscreen', animatedvideo, '--play-and-exit'])

class Robot:
    def __init__(self, config: RobotConfig = None) -> None:
        self._setup_logging()
        #logging.info("Starting Robot")
        
        if config is None:
            # use default config
            self.config = RobotConfig()
        else:
            self.config = config
        
        #logging.debug("Starting AnimationManager")
        self.animator = AnimationManager(config=self.config)

        #logging.debug("Starting SoundManager")
        self.sound_manager = SoundManger(config=self.config)

        #logging.debug("Starting TTSManager")
        self.tts_manager = TTSManager(config=self.config)

        #logging.debug("Starting Serial Port")
        self.port = serial.Serial("/dev/ttyS0", baudrate=self.config.port_baudrate, timeout=None)

        # load up all the operational states here. 
        # 
        #logging.debug("Creating Operational Modes")
        self.operational_modes = [
            AgeGenderOperationalMode(),
            EmotionOperationalMode(),
            ObjectSearchOperationMode(biscuit_mode=False, label='person'),
            ObjectSearchOperationMode(biscuit_mode=True, label='person'),
            ObjectSearchOperationMode(biscuit_mode=False, label='cup'),
            ObjectSearchOperationMode(biscuit_mode=True, label='cup'),
        ]
    def animate(self, images):
        self.animator.animate(images)
        
    def load(self):
        self.animator.load_animations()
        
    def run(self):
        self.start()

        #logging.info("Starting run")
        # start the robot in IDLE
        state = RobotState.IDLE

        keep_running = True
        while keep_running:
            # logging.debug(f"Step on current state: {state}")
            state = self.step(state)
            if state is None:
                # if for some reason our step does not set a new state, put the robot
                # into idle
                state = RobotState.IDLE
            elif state == RobotState.EXIT:
                # if state is switched to EXIT, then exit out of the loop
                keep_running = False
        self.exit()
    
    def start(self):
        # take care of any startup activities here
        #logging.info("Starting")
        #images=load_images('/home/pi/MiniMax/Animations/greetings/')
        #self.say("Greetings Humans")
        #playvideo('/home/pi/MiniMax/DefaultSayings/greetings.mp4')
        self.sound_manager.play_sound("greetings")
        self.animate(self.config.animations[0])
        
    def exit(self):
        #logging.info("Exiting")
        # take care of any close down activities here.
        #images=load_images('/home/pi/MiniMax/Animations/powerdown/')
        self.sound_manager.play_sound("powerdown")
        self.animate(self.config.animations[2])
        #playvideo('/home/pi/MiniMax/DefaultSayings/powerdown.mp4')
        #self.sound_manager.play_sound("powerdown")
        #pygame.display.flip()
        #self.animate(1)
        #images=[]
        time.sleep(1.5)

    def step(self, state):
        # logging.debug("Getting inputs")
        inputs = self._get_inputs()

        # logging.debug(inputs.pressed_keys)

        # Special Case States
        if state == RobotState.IDLE:
            luckynumb=random.randint(0,1100000)
            if luckynumb>1000 and luckynumb <1010:
                # put random saying in here
                randsay=random.randint(0,19)
                images=load_images('/home/pi/MiniMax/Animations/'+sayings[randsay]+'/')
                self.sound_manager.play_sound(sayings[randsay])
                self.animate(images)
             # logging.debug("Processing idle state")
            # If we're in IDLE, we want to check whether we should be switching across to 
            # any other operational modes, or quitting. If so, return the relevant state.
            # If there is no input, then return IDLE again so that we keep idling.
            if 'q' in inputs.pressed_keys:
                # exit if q is pressed
                return RobotState.EXIT
            # Operational Mode States
            # We want to loop through all the operational states currently loaded into the robot.
            # For each loaded state, we check whether the relevant trigger key has been set. If it
            # has, then we return the relevant RobotState so that we know to trigger that operational
            # state in the next call to step()
            for om in self.operational_modes:
                if om.get_key() in inputs.pressed_keys:
                    return om.get_state()
            
            return RobotState.IDLE
        
        elif state == RobotState.EXIT:
            # shouldn't actually be able to get here, but take care of it just in case
            return RobotState.EXIT

        # logging.debug("Processing operational mode states")
        # Operational Mode States
        for om in self.operational_modes:
            if om.get_state() == state:
                return om.run(self)

    def _get_inputs(self) -> InputObject:
        inputs = InputObject()

        return inputs
    
    def say(self, string):
        self.tts_manager.say(string)

    def play_sound(self, filename):
        self.sound_manager.play_sound(filename)

    def animate(self, images):
        self.animator.animate(images)

    def write_serial(self, pin):
        self.port.write(str.encode(pin))

    def _setup_logging(self):
        logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler( "logs/debug.log", mode='w'),
            logging.StreamHandler()
            ]
        )