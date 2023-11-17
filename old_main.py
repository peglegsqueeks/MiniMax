import os
import argparse
import sys
import time
import cv2
import operator
import serial
import pyttsx3
import pygame
import numpy as np
from pathlib import Path
import depthai as dai
from MultiMsgSync import TwoStageHostSeqSync
import blobconverter
import warnings
import threading
import queue
frames = 0
warnings.filterwarnings("ignore", category=DeprecationWarning) 

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
# wmctrl -a Brains pi@Brains: ~/examples/lite/examples/object_detection/raspberry_pi

port = serial.Serial("/dev/ttyS0", baudrate=115200, timeout=None)
pygame.mixer.pre_init(48000, -16, 8, 8192)# initialise music,sound mixer
pygame.mixer.init()
#############################################################################
### set up pygame    and load sounds ready to play from Sound folder      ###
#############################################################################
FPS = 60
pygame.init()
display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
background = pygame.Surface(display.get_size())
clock = pygame.time.Clock()
# create a queue to send commands from the main thread

frames = 0
paused = False
running = True
char=0
# define robot sounds
startup_chirp = pygame.mixer.Sound("Sound/2.mp3")
startup_powerup = pygame.mixer.Sound("Sound/Powerup/Powerup_chirp2.mp3")  
questioning1 = pygame.mixer.Sound("Sound/Randombeeps/Questioning_computer_chirp.mp3")
double_beep = pygame.mixer.Sound("Sound/Randombeeps/Double_beep2.mp3")
long_powerdown = pygame.mixer.Sound("Sound/Powerdown/Long_power_down.mp3")  
Radar_bleep_chirp = pygame.mixer.Sound("Sound/Radarscanning/Radar_bleep_chirp.mp3")
Radar_scanning_chirp = pygame.mixer.Sound("Sound/Radarscanning/Radar_scanning_chirp.mp3")
celebrate1 = pygame.mixer.Sound("Sound/celebrate1.mp3")
Da_de_la = pygame.mixer.Sound("Sound/Randombeeps/Da_de_la.mp3")

# set navigation variables 
global tolerance, ymax
xres, yres, x_deviation, ymax =800, 600, 0, 0
keyword_engine_shutdown=False
# setup physical switch gpio 
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(17,GPIO.IN)

_MARGIN, _ROW_SIZE, _FONT_SIZE, _FONT_THICKNESS  = 10, 10, 1, 1   # pixels
_TEXT_COLOR = (0, 240, 0)  # green
x, y, r_person = 0, 0, 0
person_reached, pin ='False', '2z'

# define more variables
old_value = ""
input_value = ""
global q
global frame1
global frame_rate_calc
global quitting
q = queue.Queue()
quitting=False
frame_rate_calc = 1
input_value=False
new_object='person'
key_pressed=0
stop = False
########################
###   Functions      ###
########################

# Seperate Thread for Text to Speech Engine. Just push text to speak to the queue via "q.put"
class TTSThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True
        self.start()

    def run(self):
        #tts_engine = pyttsx3.init()
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
        
def reached_person(thing):
    time.sleep(0.1)
    saying='I think I found a ' +str(thing)
    q.put(saying)
    animate(2)
    #engine.runAndWait()
    print('well, hello there, I think I found a ' +str(thing))
    time.sleep(0.2)
    
def biscuit():
    
    time.sleep(1)
    saying='If you want a biscuit, take one from my tray'
    #engine.runAndWait()
    q.put(saying)
    animate(4)
    time.sleep(0.1)
    #print('if you want a biscuit, take one from my tray')
    start=time.time()
    elapsed=4
    engine.endLoop()
    #time.sleep(0.01)
    while int(elapsed)>0:
        time.sleep(0.01)
        end=time.time()
        elapsed=4-int(end-start)
        print(4-int(elapsed))
        countdown=int(elapsed)
        engine.say(str(countdown)+ ' seconds')
        #saying=(str(countdown)+ ' seconds')
        #q.put(saying)
        #animate(1)
        engine.runAndWait()
        #time.sleep(0.01)
    #time.sleep(0.1)
    engine.say('The biscuits are leaving now bye bye')
    engine.runAndWait()
    #q.put(saying)
    animate(4)
    #engine.runAndWait()
    #time.sleep(0.1)
    pin='9z' #do a 180 degree turn
    port.write(str.encode(pin))
    pin='2z'
    port.write(str.encode(pin))
    time.sleep(1)
        
def your_quitting():
    time.sleep(0.1)
    saying="Obviously you don't need me any more. I will power down."
    #engine.say("Obviously you don't need me any more. I'll power down.")
    #engine.runAndWait()
    q.put(saying)
    animate(4)
    #time.sleep(0.1)
    print('before sleep')
    #long_powerdown.play()
    print('after sound')
    #pygame.display.quit()
    #cv2.destroyAllWindows()
    #s.videostream.stop()
    
def search_on(thing):
    time.sleep(0.1)
    saying="Search mode enabled. Searching for " + thing
    q.put(saying)
    animate(4)
    #engine.runAndWait()
    #time.sleep(0.1)
    Radar_bleep_chirp.play()
    
def emotion_detect():
    time.sleep(0.1)
    saying="Detection of human emotional state enabled."
    q.put(saying)
    animate(4)
    #engine.runAndWait()
    #time.sleep(0.1)
    Radar_scanning_chirp.play()

def emotion_end():
    time.sleep(0.1)
    saying="Emotion Detection state disabled."
    q.put(saying)
    animate(3)
    #engine.runAndWait()
    #time.sleep(0.1)
    Radar_bleep_chirp.play()
    
def age_gender_detect():
    time.sleep(0.1)
    saying="Detecting human age and gender"
    q.put(saying)
    animate(3)
    #engine.runAndWait()
    #time.sleep(0.1)
    Radar_scanning_chirp.play()    
    
def age_gender_end():
    time.sleep(0.1)
    saying="Age and Gender Detection disabled."
    q.put(saying)
    animate(3)
    #engine.runAndWait()
    #time.sleep(0.1)
    Radar_scanning_chirp.play()

def search_on_biscuit():
    time.sleep(0.1)
    saying="Searching for people who would like biscuits"
    q.put(saying)
    animate(4)
    #engine.runAndWait()
    time.sleep(0.1)
    Radar_bleep_chirp.play()
    time.sleep(0.1)
    #pygame.display.quit()
    #pygame.init()
    #display = pygame.display.set_mode((300, 300)) 
 
def search_off():
    saying="Search mode Terminated."
    q.put(saying)
    animate(2)
    #engine.runAndWait()
    time.sleep(0.1)
    Radar_scanning_chirp.play()
    #pygame.display.quit()
    time.sleep(0.1)
    #pygame.init()
    #display = pygame.display.set_mode((300, 300))
    #cap.release()

def celebrate():
    time.sleep(0.1)
    saying="Objective reached."
    print('focus on terminal')
    q.put(saying)
    animate(3)
    time.sleep(0.5)
    saying="Woo Hoo, Yay."
    #print('focus on terminal')
    q.put(saying)
    animate(3)
    #engine.runAndWait()
    time.sleep(0.2)
    celebrate1.play()
    time.sleep(0.2)
    Da_de_la.play()
    time.sleep(0.2)
    celebrate1.play()
    print('focus on terminal')
    time.sleep(0.1)
    
def search_360(thing):
    saying= thing + " can't be located, circle search initiated"
    q.put(saying)
    animate(4)
    #engine.runAndWait()
    time.sleep(0.1)
    Radar_bleep_chirp.play()

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def create_pipeline(stereo):
    pipeline = dai.Pipeline()
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300,300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    #SensorResolution.THE_1080_P
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    cam.preview.link(cam_xout.input)
    # Workaround: remove in 2.18, use `cam.setPreviewNumFramesPool(10)`
    # This manip uses 15*3.5 MB => 52 MB of RAM.
    copy_manip = pipeline.create(dai.node.ImageManip)
    copy_manip.setNumFramesPool(15)
    copy_manip.setMaxOutputFrameSize(3499200)
    cam.preview.link(copy_manip.inputImage)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    copy_manip.out.link(face_det_manip.inputImage)

    '''if stereo:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        # Spatial Detection network if OAK-D
        print("OAK-D detected, app will display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        face_det_nn.setBoundingBoxScaleFactor(0.8)
        face_det_nn.setDepthLowerThreshold(100)
        face_det_nn.setDepthUpperThreshold(5000)
        stereo.depth.link(face_det_nn.inputDepth)
    else: # Detection network if OAK-1'''
    print("OAK-1 detected, app won't display spatial coordiantes")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    face_det_manip.out.link(face_det_nn.input)
    # Send face detections to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    face_det_nn.out.link(face_det_xout.input)
    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'age_gender_manip' to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(image_manip_script.inputs['face_det_in'])
    # Only send metadata, we are only interested in timestamp, so we can sync
    # depth frames with NN output
    face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])
    copy_manip.out.link(image_manip_script.inputs['preview'])
    image_manip_script.setScript("""
    import time
    msgs = dict()

    def add_msg(msg, name, seq = None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        # node.warn(f"New msg {name}, seq {seq}")

        # Each seq number has it's own dict of msgs
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg

        # To avoid freezing (not necessary for this ObjDet model)
        if 15 < len(msgs):
            #node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() # Remove first element

    def get_msgs():
        global msgs
        seq_remove = [] # Arr of sequence numbers to get deleted
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
            # node.warn(f"Checking sync {seq}")

            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == 2: # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                return syncMsgs # Returned synced msgs
        return None

    def correct_bb(bb):
        if bb.xmin < 0: bb.xmin = 0.001
        if bb.ymin < 0: bb.ymin = 0.001
        if bb.xmax > 1: bb.xmax = 0.999
        if bb.ymax > 1: bb.ymax = 0.999
        return bb

    while True:
        time.sleep(0.001) # Avoid lazy looping

        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')

        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
            # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
            passthrough = node.io['passthrough'].get()
            seq = passthrough.getSequenceNum()
            add_msg(face_dets, 'dets', seq)

        sync_msgs = get_msgs()
        if sync_msgs is not None:
            img = sync_msgs['preview']
            dets = sync_msgs['dets']
            for i, det in enumerate(dets.detections):
                cfg = ImageManipConfig()
                correct_bb(det)
                cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
                # node.warn(f"Sending {i + 1}. age/gender det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
                cfg.setResize(64, 64)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(img)
    """)
    manip_manip = pipeline.create(dai.node.ImageManip)
    manip_manip.initialConfig.setResize(64, 64)
    manip_manip.setWaitForConfigInput(True)
    image_manip_script.outputs['manip_cfg'].link(manip_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(manip_manip.inputImage)
    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    emotions_nn = pipeline.create(dai.node.NeuralNetwork)
    emotions_nn.setBlobPath(blobconverter.from_zoo(name="emotions-recognition-retail-0003", shaves=6))
    manip_manip.out.link(emotions_nn.input)
    recognition_xout = pipeline.create(dai.node.XLinkOut)
    recognition_xout.setStreamName("recognition")
    emotions_nn.out.link(recognition_xout.input)
    return pipeline

def emote():
    with dai.Device() as device:
        device.setLogLevel(dai.LogLevel.CRITICAL)
        device.setLogOutputLevel(dai.LogLevel.CRITICAL)
        stereo = 1 < len(device.getConnectedCameras())
        device.startPipeline(create_pipeline(stereo))
        sync = TwoStageHostSeqSync()
        queues = {}
        # Create output queues
        for name in ["color", "detection", "recognition"]:
            queues[name] = device.getOutputQueue(name)
        while True:
            for name, q in queues.items():
                # Add all msgs (color frames, object detections and age/gender recognitions) to the Sync class.
                if q.has():
                    sync.add_msg(q.get(), name)
            msgs = sync.get_msgs()
            if msgs is not None:
                frame = msgs["color"].getCvFrame()
                detections = msgs["detection"].detections
                recognitions = msgs["recognition"]

                for i, detection in enumerate(detections):
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    rec = recognitions[i]
                    emotion_results = np.array(rec.getFirstLayerFp16())
                    emotion_name = emotions[np.argmax(emotion_results)]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                    y = (bbox[1] + bbox[3]) // 2
                    cv2.putText(frame, emotion_name, (bbox[0]+20, y-100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                    cv2.putText(frame, emotion_name, (bbox[0]+20, y-100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                    #if stereo:
                        # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                        #coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                        #cv2.putText(frame, coords, (bbox[0], y + 80), cv2.FONT_HERSHEY_TRIPLEX, .7, (0, 0, 0), 8)
                        #cv2.putText(frame, coords, (bbox[0], y + 80), cv2.FONT_HERSHEY_TRIPLEX, .7, (255, 255, 255), 2)
                cv2.imshow("Camera", frame)
            key_pressed=cv2.waitKey(1)
            if key_pressed == ord('z'):
                print('Terminating Emotion Detection')
                cv2.destroyAllWindows()
                emotion_end()
                return key_pressed

def create_pipeline2(stereo):
    pipeline = dai.Pipeline()
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    # Workaround: remove in 2.18, use `cam.setPreviewNumFramesPool(10)`
    # This manip uses 15*3.5 MB => 52 MB of RAM.
    copy_manip = pipeline.create(dai.node.ImageManip)
    copy_manip.setNumFramesPool(15)
    copy_manip.setMaxOutputFrameSize(3499200)
    cam.preview.link(copy_manip.inputImage)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    copy_manip.out.link(cam_xout.input)

    # ImageManip will resize the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    copy_manip.out.link(face_det_manip.inputImage)

    '''#if stereo:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Spatial Detection network if OAK-D
        print("OAK-D detected, app will display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        face_det_nn.setBoundingBoxScaleFactor(0.8)
        face_det_nn.setDepthLowerThreshold(100)
        face_det_nn.setDepthUpperThreshold(5000)
        stereo.depth.link(face_det_nn.inputDepth)
    else: # Detection network if OAK-1'''
    print("OAK-1 detected, app won't display spatial coordiantes")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    face_det_nn.input.setQueueSize(1)
    face_det_manip.out.link(face_det_nn.input)

    # Send face detections to the host (for bounding boxes)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    face_det_nn.out.link(face_det_xout.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'recognition_manip' to crop the initial frame
    image_manip_script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(image_manip_script.inputs['face_det_in'])

    # Remove in 2.18 and use `imgFrame.getSequenceNum()` in Script node
    face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])

    copy_manip.out.link(image_manip_script.inputs['preview'])

    image_manip_script.setScript("""
    import time
    msgs = dict()

    def add_msg(msg, name, seq = None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        # node.warn(f"New msg {name}, seq {seq}")

        # Each seq number has it's own dict of msgs
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg

        # To avoid freezing (not necessary for this ObjDet model)
        if 15 < len(msgs):
            node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() # Remove first element

    def get_msgs():
        global msgs
        seq_remove = [] # Arr of sequence numbers to get deleted
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
            # node.warn(f"Checking sync {seq}")

            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == 2: # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                return syncMsgs # Returned synced msgs
        return None

    def correct_bb(bb):
        if bb.xmin < 0: bb.xmin = 0.001
        if bb.ymin < 0: bb.ymin = 0.001
        if bb.xmax > 1: bb.xmax = 0.999
        if bb.ymax > 1: bb.ymax = 0.999
        return bb

    while True:
        time.sleep(0.001) # Avoid lazy looping

        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')

        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
            # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
            passthrough = node.io['passthrough'].get()
            seq = passthrough.getSequenceNum()
            add_msg(face_dets, 'dets', seq)

        sync_msgs = get_msgs()
        if sync_msgs is not None:
            img = sync_msgs['preview']
            dets = sync_msgs['dets']
            for i, det in enumerate(dets.detections):
                cfg = ImageManipConfig()
                correct_bb(det)
                cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
                # node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
                cfg.setResize(62, 62)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(img)
    """)

    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(62, 62)
    recognition_manip.setWaitForConfigInput(True)
    image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)

    # Second stange recognition NN
    print("Creating recognition Neural Network...")
    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setBlobPath(blobconverter.from_zoo(name="age-gender-recognition-retail-0013", shaves=6))
    recognition_manip.out.link(recognition_nn.input)

    recognition_xout = pipeline.create(dai.node.XLinkOut)
    recognition_xout.setStreamName("recognition")
    recognition_nn.out.link(recognition_xout.input)

    return pipeline

def age_gender():
    with dai.Device() as device:
        stereo = 1 < len(device.getConnectedCameras())
        device.startPipeline(create_pipeline2(stereo))

        sync = TwoStageHostSeqSync()
        queues = {}
        # Create output queues
        for name in ["color", "detection", "recognition"]:
            queues[name] = device.getOutputQueue(name)

        while True:
            for name, q in queues.items():
                # Add all msgs (color frames, object detections and recognitions) to the Sync class.
                if q.has():
                    sync.add_msg(q.get(), name)

            msgs = sync.get_msgs()
            if msgs is not None:
                frame = msgs["color"].getCvFrame()
                detections = msgs["detection"].detections
                recognitions = msgs["recognition"]

                for i, detection in enumerate(detections):
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                    # Decoding of recognition results
                    rec = recognitions[i]
                    age = int(float(np.squeeze(np.array(rec.getLayerFp16('age_conv3')))) * 100)
                    gender = np.squeeze(np.array(rec.getLayerFp16('prob')))
                    gender_str = "female" if gender[0] > gender[1] else "male"

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                    y = (bbox[1] + bbox[3]) // 2
                    
                    cv2.putText(frame, gender_str, (bbox[0], y - 102), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                    cv2.putText(frame, gender_str, (bbox[0], y - 102), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, str(age), (bbox[0]+120, y-102), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 8)
                    cv2.putText(frame, str(age), (bbox[0]+120, y-102), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
                    #if stereo:
                        # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                        #coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                        #cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                        #cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Camera", frame)
            key_pressed=cv2.waitKey(1)
            if key_pressed == ord('z'):
                print('Terminating Age and Gender Detection')
                cv2.destroyAllWindows()
                age_gender_end()
                return key_pressed 
            
#################################################################################
######         main VP loop  Search for People / send serial comms         ######
#################################################################################    
def testing(objective_reached,more_biscuit):
    global object_name, x_deviation, r_person, person_reached, pin, old_pin
    xres, yres,tolerance,frame_rate_calc,x_deviation,bottom_buffer,not_a_person =800,600,95,1,0,5,0
    x1,x2,y1,y2=0,0,0,0
    old_status='LOST'
    labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    nnPathDefault = str((Path(__file__).parent / Path('../models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
    parser = argparse.ArgumentParser()
    parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
    parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=True)
    args = parser.parse_args()
    fullFrameTracking = args.full_frame
    #fullFrameTracking=True
    # Create pipeline
    pipeline = dai.Pipeline()
    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    detectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
    objectTracker = pipeline.create(dai.node.ObjectTracker)
    xlinkOut = pipeline.create(dai.node.XLinkOut)
    trackerOut = pipeline.create(dai.node.XLinkOut)
    xlinkOut.setStreamName("preview")
    trackerOut.setStreamName("tracklets")
    # Properties
    camRgb.setPreviewSize(300, 300)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) #THE_1080_P 
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)
    # testing MobileNet DetectionNetwork
    detectionNetwork.setBlobPath(args.nnPath)
    detectionNetwork.setConfidenceThreshold(0.6)
    detectionNetwork.input.setBlocking(False)
    objectTracker.setDetectionLabelsToTrack([15])  # track only person
    # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
    objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
    # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
    objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
    # Linking
    camRgb.preview.link(detectionNetwork.input)
    objectTracker.passthroughTrackerFrame.link(xlinkOut.input)
    if fullFrameTracking:
        camRgb.setPreviewKeepAspectRatio(False)
        camRgb.video.link(objectTracker.inputTrackerFrame)
        #objectTracker.inputTrackerFrame.setBlocking(False)
        #objectTracker.inputTrackerFrame.setQueueSize(2)
    else:
        detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
    detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
    detectionNetwork.out.link(objectTracker.inputDetections)
    objectTracker.out.link(trackerOut.input)
    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        preview = device.getOutputQueue("preview", 4, False)
        tracklets = device.getOutputQueue("tracklets", 4, False)
        startTime = time.monotonic()
        counter = 0
        fps = 0
        frame = None
        found_people = {}
        sorted_tracked= {}
        while(True):
            imgFrame = preview.get()
            track = tracklets.get()
            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
            color = (255, 255, 255)
            frame = imgFrame.getCvFrame()
            frame=cv2.resize(frame,(800,600),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            trackletsData = track.tracklets
            old_pin=pin
            for t in trackletsData:
                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                xmin, ymin = x1, y1
                xmax, ymax = x2, y2
                x_diff, y_diff = (xmax-xmin), (ymax-ymin)
                obj_x_center = int(xmin+(x_diff/2))
                obj_y_center = int(ymin+(y_diff/2))
                center_coordinates = (obj_x_center, obj_y_center)
                # Put info in dictionary for detected / tracked object ONLY if detected object is being successfully tracked #
                found_people[t.id] = (t.id, obj_x_center, obj_y_center, ymax, t.status.name)
                if t.status.name!="TRACKED":
                    found_people.popitem()
                radius = 2
                #cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color)
                cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 45), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color)
                cv2.putText(frame, t.status.name, (x1 + 10, y1 + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                cv2.circle(frame, center_coordinates, radius, color, 1)
            sorted_x = sorted(found_people.items(), key=operator.itemgetter(0))# key sort only items that are tracked, lowest key first #
            if len(sorted_x)<1: 
                pin = "5z"
                port.write(str.encode(pin)) # send pin to motor controller
                #print('Old pin '+old_pin +' new pin '+pin)
                #print('There are ZERO people in current frame ! ')
                
            if len(sorted_x)>0:
                for q in sorted_x:
                    listy=(q[1])
                    index=listy[0]
                    status=listy[4]
                    if status=="TRACKED":
                        sorted_tracked[index]=listy
            if len(sorted_tracked)>0: # if the dictionary is not empty....
                for w in sorted_tracked:
                    newlist=(sorted_tracked[w])
                obj_x_center= newlist[1] # get x co-ordinate of lowest ID
                obj_y_center= newlist[2] # get y co-ordinate of lowest ID
                ymax= newlist[3] # get ymax of lowest ID
                x_deviation = int((xres/2)-obj_x_center)
                # calculate the deviation from the center of the screen
                if(abs(x_deviation)<tolerance): # is object in the middle of screen?
                    if abs(ymax>(yres-bottom_buffer)):     # is object close to the bottom of the frame?
                        pin = "2z"
                        r_person=r_person+1
                        port.write(str.encode(pin))
                        print('Old pin '+old_pin +' new pin '+pin)
                        if (r_person>2):
                            pin = "2z"
                            port.write(str.encode(pin))
                            print('Old pin '+old_pin +' new pin '+pin)
                            person_reached='True'
                            objective_reached='True'
                            print('........................... reached objective')
                            print('waiting at objective reached')
                            cv2.destroyAllWindows()
                            return objective_reached, more_biscuit
                    else:
                        objective_reached='False'
                        pin = "1z"
                        if old_pin != pin:
                            port.write(str.encode(pin))
                            print('Old pin '+old_pin +' new pin '+pin)
                            print("........................... moving robot FORWARD")
                            r_person=0
                else:
                    objective_reached='False'
                    if (x_deviation>tolerance):
                        if old_pin !="3z" and x_deviation<175:
                            pin = "3z"
                            port.write(str.encode(pin))
                            print('Old pin '+old_pin +'  '+pin+'........................... turning left' )
                            r_person=0
                        if old_pin !="7z" and x_deviation>=175:
                            pin="7z"
                            port.write(str.encode(pin))
                            print('Old pin '+old_pin +'  '+pin+'....... turning left on the spot' )
                            r_person=0
                    elif ((x_deviation*-1)>tolerance):
                        if old_pin !="4z" and abs(x_deviation)<175:
                            pin="4z"
                            port.write(str.encode(pin))
                            print('Old pin '+old_pin +'  '+pin+'........................... turning right' )
                            r_person=0
                        if old_pin !="8z" and abs(x_deviation)>=175:
                            pin="8z"
                            port.write(str.encode(pin))
                            print('Old pin '+old_pin +'  '+pin+'....... turning right on the spot' )
                            r_person=0
            cv2.putText(frame, "fps: {:.2f}".format(fps), (2, frame.shape[0] - 7), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color)
            key_pressed=cv2.waitKey(1)
            if key_pressed==ord('z'):
                print('Aborting search z ')
                cv2.destroyAllWindows()
                search_off()
                print('menu waiting for keyboard input')
                more_biscuit=0
                pin='5z'
                port.write(str.encode(pin))
                return key_pressed, more_biscuit
            sorted_tracked.clear()
            cv2.imshow("tracker", frame)
    return objective_reached, more_biscuit

def draw_robot_small_mouth():
    color=(200,50,50)
    black=(0,0,0)
    resx = 1024
    resy = 768
    size1 = (int(resx*0.3), int(resy*0.5), int(resx*0.4), int(resy*0.067))
    size2 = (int(resx*0.35), int(resy*0.51), int(resx*0.3), int(resy*0.05))
    #pygame.draw.rect(display, (50,50,150), pygame.Rect(30, 60, 160, 80),  2)
    pygame.draw.ellipse(display, color, size1)
    pygame.draw.ellipse(display, black, size2)
    pygame.draw.rect(display, (50,50,150), pygame.Rect(resx*0.1, resy*0.04, resx*0.8, resy*0.70),  8) # head outline
    #pygame.draw.rect(display, (200, 50, 50), pygame.Rect(int(resx*0.3), int(resy*0.6), int(resx*0.4), int(resy*0.05)),  4) #mouth
    pygame.draw.circle(display,(255,0,0),[int(resx*0.3), int(resy*0.22)], 80, 4) #outer eye
    pygame.draw.circle(display,(255,0,0),[int(resx*0.7), int(resy*0.22)], 80, 4)
    pygame.draw.circle(display,(5,5,200),[int(resx*0.3), int(resy*0.22)], 50, 0) #blue eye
    pygame.draw.circle(display,(5,5,200),[int(resx*0.7), int(resy*0.22)], 50, 0)
    pygame.draw.circle(display,(255,0,0),[int(resx*0.49), int(resy*0.37)], 9, 5) #nothrals
    pygame.draw.circle(display,(255,0,0),[int(resx*0.51), int(resy*0.37)], 9, 5)
    pygame.draw.circle(display,(0,0,0),[int(resx*0.3), int(resy*0.22)], 22, 0) #iner eye
    pygame.draw.circle(display,(0,0,0),[int(resx*0.7), int(resy*0.22)], 22, 0)
    #pygame.display.flip()

def draw_robot_big_mouth():
    color=(200,50,50)
    black=(0,0,0)
    resx = 1024
    resy = 768
    size1=(int(resx*0.33), int(resy*0.5), int(resx*0.35), int(resy*0.16))
    size2=(int(resx*0.345), int(resy*0.52), int(resx*0.32), int(resy*0.12))
    pygame.draw.ellipse(display, color, size1)
    pygame.draw.ellipse(display, black, size2)
    #pygame.draw.rect(display, (50,50,150), pygame.Rect(30, 60, 160, 80),  2)
    pygame.draw.rect(display, (50,50,150), pygame.Rect(resx*0.1, resy*0.04, resx*0.8, resy*0.70),  8) #head outline
    #pygame.draw.rect(display, (200,50,50), pygame.Rect(int(resx*0.3), int(resy*0.6), int(resx*0.4), int(resy*0.10)),  4) #mouth
    pygame.draw.circle(display,(255,0,0),[int(resx*0.3), int(resy*0.22)], 80, 4) #outer eye
    pygame.draw.circle(display,(255,0,0),[int(resx*0.7), int(resy*0.22)], 80, 4)
    pygame.draw.circle(display,(5,5,200),[int(resx*0.3), int(resy*0.22)], 70, 0) #blue eye
    pygame.draw.circle(display,(5,5,200),[int(resx*0.7), int(resy*0.22)], 70, 0)
    pygame.draw.circle(display,(255,0,0),[int(resx*0.49), int(resy*0.37)], 9, 5) #nothrals
    pygame.draw.circle(display,(255,0,0),[int(resx*0.51), int(resy*0.37)], 9, 5)
    pygame.draw.circle(display,(0,0,0),[int(resx*0.3), int(resy*0.22)], 29, 0) #iner eye
    pygame.draw.circle(display,(0,0,0),[int(resx*0.7), int(resy*0.22)], 29, 0)
    #pygame.display.flip()

def animate(loop):
    delay=0.08
    for x in range(loop-1):
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_big_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_small_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_big_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_small_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_big_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_small_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_big_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_small_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_big_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        draw_robot_small_mouth()
        pygame.display.update()
        time.sleep(delay)
        display.blit(background, (0, 0))
        #pygame.display.update()
        
###################################################################
###      START OF MAIN CODE                                     ###
###################################################################
if __name__ == '__main__':
  # start initializing voice and make chirping  robot sounds
  
  iterate=0
  display.blit(background, (0, 0))
  draw_robot_small_mouth()
  pygame.display.update()
  engine = pyttsx3.init()
  engine.setProperty("rate", 168)
  engine.setProperty('voice', 'english+f4')
  engine.setProperty("volume", 1)
  saying='Greetings Humans.' # Initializing warm up sequence
  char=len(saying)
  q = queue.Queue()
  tts_thread = TTSThread(q)  # note: thread is auto-starting
  q.put(saying)
  animate(2)
  #engine.runAndWait()
  # play some sounds at start
  #time.sleep(0.3)
  #startup_powerup.play()
  #time.sleep(1.4)
  #startup_chirp.play()
  #time.sleep(1.2)
  #questioning1.play()
  #time.sleep(1.1)
  
# initialize control variables 
  speaking=False
  global the_object, more_biscuit
  the_object='person'
  pin='5z'
  old_pin='6z'
  control='True'
  more_biscuit=0
  wait=True

  x = 'p'
  z=0
  char=0
  #start=0

  while wait==True:
      display.blit(background, (0, 0))
      draw_robot_small_mouth()
      pygame.display.update()
      clock.tick(FPS)
      frames += 1
      spek=engine.isBusy()
      if spek==True:
          iterate=iterate+1
          #if iterate < (int(char)):
          animate(1)
      spek=engine.isBusy()
      if spek==False:
          iterate=0
                       
      pygamekeypress=False
      if more_biscuit==0:
          #x=sys.stdin.read(1)[0]
          while pygamekeypress==False:
              #print('hello')
              for event in pygame.event.get():
                  if event.type == pygame.KEYDOWN:
                      print('keydown detected')
                      if event.key == pygame.K_p:
                          print('check left')
                          x="p"
                          pygamekeypress=True
                          break
                      if event.key == pygame.K_q:
                          x="q"
                          pygamekeypress=True
                          break
                      if event.key == pygame.K_b:
                          print('check left')
                          x="b"
                          pygamekeypress=True
                          break
                      if event.key == pygame.K_a:
                          break
                      if event.key == pygame.K_e:
                          print('Emotion Detector')
                          x="e"
                          pygamekeypress=True
                          break
                      if event.key == pygame.K_c:
                          x="c"
                          pygamekeypress=True
                          break
                  
      print("You pressed", x)
      if x == "p":# Select the Search for a person mode.Exit via "z" key.
          #print("you pressed p now start VP")
          new_object='person'
          search_on(new_object)
          control='False'
          more_biscuit=0
          z=1
      if x == "q": # Select Quit the program / Only works after other modes have been exited with "z" key
          #print("You pressed q Now QUIT")
          your_quitting()
          wait=False
          control='True'
          more_biscuit=0
      if x == "b":# Select the search for a person to Deliver a Biscuit mode.Exit via "z" key.
          #print("you pressed b now start VP")
          new_object='person'
          search_on_biscuit()
          control='False'
          z=1
      if x == "c":# Select search for a cup mode. Exit via "z" key.
          #print("you pressed c now start VP")
          new_object='cup'
          search_on(new_object)
          control='False'
          more_biscuit=0
          z=1
      if x == "a": # Select the Age / Gender recognizer. Exit via "z" key.
          #print("you pressed m now start VP")
          new_object='person'
          age_gender_detect()
          control='False'
          more_biscuit=0
          z=1
      if x == "e": # Select the Emotional state recognizer. Exit via "z" key.
          #print("you pressed k now start VP")
          new_object='person'
          emotion_detect()
          control='False'
          more_biscuit=0
          z=1
      while control=='False':
          #start=time.time()
          #print('before starting VP function ')
          if x=='e':
              control=emote()
              break
          if x=='a':
              control=age_gender()
              break  
          control, more_biscuit=testing(new_object,more_biscuit)   ########start vp function, send the name of the target object(new_object) return (control and more_biscuit) 
          #print('After starting run function ')
          if control=='z':  #Use z to break out of VP loop
              x=''
              more_biscuit=0
              z=0
              wait=False
              #print('we are exiting')
              sys.exit()
              sys.exit(1)
              sys.exit(0)
              break
          if control=='True' and z==1: # Visual Process run and returned Target reached
              reached_person(new_object)
              if x!='b': # if the we are NOT in deliver biscuit mode then perform celebrate function
                  celebrate()
              if x=='b':  # we are in biscuit mode perform biscuit function set more_biscuit=1 to BYPASS main keyboard menu and continue biscuit mode.
                  biscuit()
                  q = queue.Queue()
                  tts_thread = TTSThread(q)
                  more_biscuit=1
              #os.system('wmctrl -a Brains pi@Brains: ~/depthai-python/examples/ObjectTracker')
              #os.system('wmctrl -a bash')
              #set focus on terminal window
              z=0
              break
    #pygame.display.quit()
    #pygame.init()
  spek=engine.isBusy()
  animate(2)
  #termios.tcsetattr(sys.stdin, termios.TCSADRAIN, filedescriptors)  #set the terminal window input mode back to a standard mode that waits for return to be pressed   
  if keyword_engine_shutdown==True:
      keyword_engine_shutdown=False
      just_try_and_stop_me = r.listen_in_background(source, callback) 
  print(control)
  print('we are exiting')
  #animate(1)
  time.sleep(1)
  cv2.destroyAllWindows()
  sys.exit()
  sys.exit(1)
  sys.exit(0)

