from typing import List, Optional
import pygame
#import moviepy.editor
#from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import pygame  #pip install pygame
from pygame import mixer
import pathlib
mixer.init()
# I don't think sounds in pygame block, so should be fine to run in the main process
# If the sounds do block, then will probably want to split this all into a seperate
# process

#TODO: want to change all the filenames to be more descriptive, given that we will be
# playing sounds by filename

_default_filepaths = [
   "Sound/Default/greetings.wav",
    "Sound/Default/powerdown.wav",
    "Sound/Default/advised.wav",
    "Sound/Default/agegender.wav",
    "Sound/Default/dis-agegender.wav",
    "Sound/Default/dis-emotional.wav",
    "Sound/Default/emotional.wav",
    "Sound/Default/found-person.wav",
    "Sound/Default/objective.wav",
    "Sound/Default/searchmode-off.wav",
    "Sound/Default/searchterminated.wav",
    
    "Sound/Sayings/better.wav",
    "Sound/Sayings/compute.wav",
    "Sound/Sayings/cross.wav",
    "Sound/Sayings/danger.wav",
    "Sound/Sayings/directive.wav",
    "Sound/Sayings/humans.wav",

    "Sound/Sayings/cautionroguerobots.mp3",
    "Sound/Sayings/chess.mp3",
    
    "Sound/Sayings/dangerwillrobinson.mp3",
    "Sound/Sayings/malfunction.mp3",
    "Sound/Sayings/nicesoftware.mp3",
    "Sound/Sayings/no5alive.mp3",
    "Sound/Sayings/program.wav",
    "Sound/Sayings/selfdestruct.wav",
    "Sound/Sayings/shallweplayagame.mp3",
    
    "Sound/Sayings/comewithme.mp3",
    "Sound/Sayings/gosomewhere.mp3",
    "Sound/Sayings/hairybaby.mp3",
    "Sound/Sayings/lowbattery.mp3",
    "Sound/Sayings/robotnotoffended.mp3",
    "Sound/Sayings/satisfiedwithmycare.wav",
    "Sound/Sayings/waitbeforeswim.mp3",
    
    "Sound/Sayings/silly.wav",
    "Sound/Sayings/stare.wav",
    "Sound/Sayings/world.wav",
    
    
       
    "Sound/2.mp3",
    "Sound/Powerup/Powerup_chirp2.mp3",
    "Sound/Randombeeps/Questioning_computer_chirp.mp3",
    "Sound/Randombeeps/Double_beep2.mp3",
    "Sound/Powerdown/Long_power_down.mp3",
    "Sound/Radarscanning/Radar_bleep_chirp.mp3",
    "Sound/Radarscanning/Radar_scanning_chirp.mp3",
    "Sound/celebrate1.mp3",
    "Sound/Randombeeps/Da_de_la.mp3",
]

class SoundManger:
    def __init__(self, config, file_paths: Optional[List[str]] = None) -> None:
        pygame.mixer.pre_init(48000, -16, 8, 8192)# initialise music,sound mixer
        pygame.mixer.init()
        if file_paths is not None:
            file_paths += _default_filepaths
        else:
            file_paths = _default_filepaths
        #self.sounds = {
           # pathlib.Path(f).stem: pygame.mixer.Sound(f) for f in file_paths
       # }
        self.channel = pygame.mixer.Channel(1)

    def play_sound(self, sound):
        video_path='DefaultSayings/'+sound+'.avi'
        sound_path='DefaultSayings/'+sound+'.wav'
        file_name = video_path
        window_name = "RobotFace"
        interframe_wait_ms = 1

        cap = cv2.VideoCapture(file_name)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
            
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(window_name, 800, 600)
        mixer.music.load(sound_path)
        mixer.music.play()

        while (True):
            ret, frame = cap.read()
            key = cv2.waitKey(1)
            if not ret:
                print("Reached end of video, exiting.")
                break

            cv2.imshow(window_name, frame)
            if key & 0x7F == ord('q'):
                print("Exit requested.")
                break

        cap.release()
        cv2.destroyAllWindows()
    

#clip = VideoFileClip('DefaultSayings/'+sound+'.avi')
        #clipresized = clip.resize (height=100)
        #clipresized.preview(fullscreen=True)
        #video = moviepy.editor.VideoFileClip("")
        #clip.ipython_display()
        #self.channel.play(self.sounds[sound])
        #self.sounds[sound].play()