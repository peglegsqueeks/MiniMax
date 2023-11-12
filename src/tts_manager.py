import threading
import queue

import pyttsx3


class TTSThread(threading.Thread):
    def __init__(self, queue, config):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.config = config

        self.start()

    def run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.config.tts_rate)
        engine.setProperty('voice', self.config.tts_voice)
        engine.setProperty("volume", self.config.tts_volume)
        engine.startLoop(False)
        t_running = True
        while t_running:
            if self.queue.empty():
                engine.iterate()
                continue
            else:
                data = self.queue.get()
                if data == "exit":
                    t_running = False
                else:
                    engine.say(data)
                    continue
        engine.endLoop()


class TTSManager:
    def __init__(self, config) -> None:
        self.queue = queue.Queue()
        self.thread = TTSThread(self.queue, config)

    def say(self, string):
        self.queue.put(string)