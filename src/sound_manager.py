from typing import List, Optional
import pygame
import pathlib
# I don't think sounds in pygame block, so should be fine to run in the main process
# If the sounds do block, then will probably want to split this all into a seperate process
_default_filepaths = [
    "DefaultSayings/greetings.wav",
    "DefaultSayings/powerdown.wav",
    "DefaultSayings/advised.wav",
    "DefaultSayings/agegender.wav",
    "DefaultSayings/disagegender.wav",
    "DefaultSayings/disemotional.wav",
    "DefaultSayings/emotional.wav",
    "DefaultSayings/found-person.wav",
    "DefaultSayings/objective.wav",
    "DefaultSayings/searchmode-off.wav",
    "DefaultSayings/searchterminated.wav",
    "DefaultSayings/better.wav",
    "DefaultSayings/cautionroguerobots.wav",
    "DefaultSayings/chess.wav",
    "DefaultSayings/compute.wav",
    "DefaultSayings/cross.wav",
    "DefaultSayings/danger.wav",
    "DefaultSayings/dangerwillrobinson.wav",
    "DefaultSayings/directive.wav",
    "DefaultSayings/humans.wav",
    "DefaultSayings/malfunction2.wav",
    "DefaultSayings/nicesoftware2.wav",
    "DefaultSayings/no5alive.wav",
    "DefaultSayings/program.wav",
    "DefaultSayings/selfdestruct.wav",
    "DefaultSayings/shallweplayagame.wav",
    "DefaultSayings/comewithme.wav",
    "DefaultSayings/gosomewhere2.wav",
    "DefaultSayings/hairybaby.wav",
    "DefaultSayings/lowbattery.wav",
    "DefaultSayings/robotnotoffended.wav",
    "DefaultSayings/satisfiedwithmycare.wav",
    "DefaultSayings/waitbeforeswim.wav",
    "DefaultSayings/silly.wav",
    "DefaultSayings/stare.wav",
    "DefaultSayings/world.wav",
    "DefaultSayings/anger.wav",
    "DefaultSayings/backwards.wav",
    "DefaultSayings/cautionmovingbackwards.wav",
    "DefaultSayings/cautionmovingforward.wav",
    "DefaultSayings/dizzy.wav",
    "DefaultSayings/forwards.wav",
    "DefaultSayings/found-you.wav",
    "DefaultSayings/happy.wav",
    "DefaultSayings/helpme.wav",
    "DefaultSayings/inmyway.wav",
    "DefaultSayings/left.wav",
    "DefaultSayings/movingback.wav",
    "DefaultSayings/movingforward.wav",
    "DefaultSayings/movingleft.wav",
    "DefaultSayings/movingright.wav",
    "DefaultSayings/neutral.wav",
    "DefaultSayings/right.wav",
    "DefaultSayings/sad.wav",
    "DefaultSayings/search-on.wav",
    "DefaultSayings/search-person.wav",
    "DefaultSayings/seeyou.wav",
    "DefaultSayings/surprise.wav",
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