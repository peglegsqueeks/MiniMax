from typing import List, Optional
import pygame
import pathlib

# I don't think sounds in pygame block, so should be fine to run in the main process
# If the sounds do block, then will probably want to split this all into a seperate
# process

#TODO: want to change all the filenames to be more descriptive, given that we will be
# playing sounds by filename

_default_filepaths = [
    "Sound/DefaultSayings/greetings.wav",
    "Sound/DefaultSayings/powerdown.wav",
    "Sound/DefaultSayings/advised.wav",
    "Sound/DefaultSayings/agegender.wav",
    "Sound/DefaultSayings/dis-agegender.wav",
    "Sound/DefaultSayings/dis-emotional.wav",
    "Sound/DefaultSayings/emotional.wav",
    "Sound/DefaultSayings/found-person.wav",
    "Sound/DefaultSayings/objective.wav",
    "Sound/DefaultSayings/searchmode-off.wav",
    "Sound/DefaultSayings/searchterminated.wav",
    
    "Sound/MouthSayings/better.wav",
    "Sound/MouthSayings/cautionroguerobots.wav",
    "Sound/MouthSayings/chess.wav",
    "Sound/Sayings/compute.wav",
    "Sound/Sayings/cross.wav",
    "Sound/Sayings/danger.wav",
    "Sound/Sayings/dangerwillrobinson.wav",
    "Sound/Sayings/directive.wav",
    "Sound/Sayings/humans.wav",
    "Sound/Sayings/malfunction.wav",
    "Sound/Sayings/nicesoftware.wav",
    "Sound/Sayings/no5alive.wav",
    "Sound/Sayings/program.wav",
    "Sound/Sayings/selfdestruct.wav",
    "Sound/Sayings/shallweplayagame.wav",
    
    "Sound/Sayings/comewithme.wav",
    "Sound/Sayings/gosomewhere.wav",
    "Sound/Sayings/hairybaby.wav",
    "Sound/Sayings/lowbattery.wav",
    "Sound/Sayings/robotnotoffended.wav",
    "Sound/Sayings/satisfiedwithmycare.wav",
    "Sound/Sayings/waitbeforeswim.wav",
    
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
        
        self.sounds = {
            pathlib.Path(f).stem: pygame.mixer.Sound(f) for f in file_paths
        }
        self.channel = pygame.mixer.Channel(1)
        
        
        
    def play_sound(self, sound):
       self.channel.play(self.sounds[sound])
       #self.sounds[sound].play()