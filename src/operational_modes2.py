from abc import ABC, abstractmethod
import time
import operator
import depthai as dai
import blobconverter
import cv2
import numpy as np
from .MultiMsgSync import TwoStageHostSeqSync
from .animation_manager import AnimationManager, load_images
from .sound_manager import *
from .utils import RobotState, frame_norm
from .utils import RobotConfig

class OperationalMode(ABC):
    @abstractmethod
    def run(self, robot):
        pass

    @abstractmethod
    def get_key(self) -> str:
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        pass

class AgeGenderOperationalMode(OperationalMode):
    def __init__(self, config: RobotConfig = None) -> None:
        super().__init__()
        
        if config is None:
            # use default config
            self.config = RobotConfig()
        else:
            self.config = config    
            #logging.debug("Starting AnimationManager")
        self.animator = AnimationManager(config=self.config)

        #logging.debug("Starting SoundManager")
        self.sound_manager = SoundManger(config=self.config)
        
    def play_sound(self, filename):
        self.sound_manager.play_sound(filename)
    
    def animate(self, images):
        self.animator.animate(images)
    
    def run(self, robot):
        self._start(robot)
        self._main(robot)
        self._end(robot)

        return RobotState.IDLE
    
    def _start(self, robot):
        time.sleep(0.1)
        images=load_images('/home/pi/MiniMax/Animations/agegender/')
        self.sound_manager.play_sound("agegender")
        self.animate(images)
        #robot.say("Detecting human age and gender")
        #robot.animate(1)
        #engine.runAndWait()
        #time.sleep(0.1)
        robot.play_sound("Radar_scanning_chirp")
    
    def _main(self, robot):
        with dai.Device() as device:
            stereo = 1 < len(device.getConnectedCameras())
            device.startPipeline(self._create_pipeline(stereo))

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
                        
                        #cv2.putText(frame, gender_str, (bbox[0], y - 102), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                        #cv2.putText(frame, gender_str, (bbox[0], y - 102), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, str(age), (bbox[0]+120, y-102), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 8)
                        cv2.putText(frame, str(age), (bbox[0]+120, y-102), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
                        #if stereo:
                        # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                        #coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                        #cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                        #cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                    #flippy=cv2.flip(frame,0)
                    cv2.imshow("Camera", frame)
                key_pressed=cv2.waitKey(1)

                if key_pressed == ord('z'):
                    break

    def _end(self, robot):
        cv2.destroyAllWindows()
        time.sleep(0.1)
        images=load_images('/home/pi/MiniMax/Animations/disagegender/')
        self.sound_manager.play_sound("disagegender")
        self.animate(images)
        #robot.say("Age and Gender Detection disabled.")

        #robot.animate(1)
        #engine.runAndWait()
        #time.sleep(0.1)
        robot.play_sound("Radar_scanning_chirp")
    
    def _create_pipeline(self, stereo):
        pipeline = dai.Pipeline()
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
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
        #print("OAK-1 detected, app won't display spatial coordiantes")
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

    def get_key(self) -> str:
        return 'a'
    
    def get_state(self) -> RobotState:
        return RobotState.AGEGENDER


class EmotionOperationalMode(OperationalMode):
    def __init__(self, config: RobotConfig = None) -> None:
        super().__init__()
        self.emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
        
        if config is None:
            # use default config
            self.config = RobotConfig()
        else:
            self.config = config    
        #logging.debug("Starting AnimationManager")
        self.animator = AnimationManager(config=self.config)

        #logging.debug("Starting SoundManager")
        self.sound_manager = SoundManger(config=self.config)
        
    def play_sound(self, filename):
        self.sound_manager.play_sound(filename)
    
    def animate(self, images):
        self.animator.animate(images)

    def run(self, robot):
        self._start(robot)
        self._main(robot)
        self._end(robot)
        return RobotState.IDLE
    
    def _start(self, robot):
        #time.sleep(0.1)
        images=load_images('/home/pi/MiniMax/Animations/emotional/')
        self.sound_manager.play_sound("emotional")
        self.animate(images)
        #robot.say("Detection of human emotional state enabled.")
        #robot.animate(1)
        robot.play_sound("Radar_scanning_chirp")
    
    def _main(self, robot):
        with dai.Device() as device:
            device.setLogLevel(dai.LogLevel.CRITICAL)
            device.setLogOutputLevel(dai.LogLevel.CRITICAL)
            stereo = 1 < len(device.getConnectedCameras())
            device.startPipeline(self._create_pipeline(stereo))
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
                        emotion_name = self.emotions[np.argmax(emotion_results)]
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                        y = (bbox[1] + bbox[3]) // 2
                        cv2.putText(frame, emotion_name, (bbox[0]+20, y-100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                        cv2.putText(frame, emotion_name, (bbox[0]+20, y-100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
                        #if stereo:
                            # You could also get detection.spatialCoordinates.x and detection.spatialCoordinates.y coordinates
                            #coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                            #cv2.putText(frame, coords, (bbox[0], y + 80), cv2.FONT_HERSHEY_TRIPLEX, .7, (0, 0, 0), 8)
                            #cv2.putText(frame, coords, (bbox[0], y + 80), cv2.FONT_HERSHEY_TRIPLEX, .7, (255, 255, 255), 2)
                    #flipped = cv2.flip(frame, 0)
                    cv2.imshow("Camera", frame)
                key_pressed=cv2.waitKey(1)
                if key_pressed == ord('z'):
                    break

    def _end(self, robot):
        cv2.destroyAllWindows()
        time.sleep(0.1)
        images=load_images('/home/pi/MiniMax/Animations/disemotional/')
        self.sound_manager.play_sound("disemotional")
        self.animate(images)
        #robot.say("Emotion Detection state disabled.")
        #robot.animate(1)
        #engine.runAndWait()
        #time.sleep(0.1)
        robot.play_sound("Radar_scanning_chirp")
    
    def _create_pipeline(self,robot):
        pipeline = dai.Pipeline()
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
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
        #print("OAK-1 detected, app won't display spatial coordiantes")
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

    def get_key(self) -> str:
        return 'e'
    
    def get_state(self) -> RobotState:
        return RobotState.EMOTIONS


class ObjectSearchOperationMode(OperationalMode):
    def __init__(self, label: str, biscuit_mode=True, config: RobotConfig = None) -> None:
        super().__init__()
        _supported_labels = [
            'person',
            'cup',
        ]
        assert label in _supported_labels, f"label '{label}' is not in supported labels {_supported_labels}"
        self.label = label
        self.biscuit_mode = biscuit_mode

        if self.label == 'person':
            self.model_id = 15
            self.model_threshold = 0.5
        elif self.label == 'cup':
            self.model_id = 3
            self.model_threshold = 0.3
            
        if config is None:
            # use default config
            self.config = RobotConfig()
        else:
            self.config = config    
        #logging.debug("Starting AnimationManager")
        self.animator = AnimationManager(config=self.config)

        #logging.debug("Starting SoundManager")
        self.sound_manager = SoundManger(config=self.config)  

    def play_sound(self, filename):
        self.sound_manager.play_sound(filename)
    
    
    def run(self, robot):
        self._start(robot)
        success = self._main(robot)
        return self._end(robot, success = success, give_biscuit_on_success=robot.config.ps_give_biscuit_on_success)
    
    def _start(self, robot):
        time.sleep(0.1)
        if self.biscuit_mode:
            #images=load_images('/home/pi/MiniMax/Results/searchterminated/')
            #self.sound_manager.play_sound("searchterminated")
            #self.animate(images)
            robot.say(f"Search mode enabled. Searching for {self.label} who like jellybeans")
        else:
            robot.say(f"Search mode enabled. Searching for {self.label}")
        #robot.animate(1)
        time.sleep(0.1)
        robot.play_sound('Radar_bleep_chirp')
        time.sleep(0.1)

    def _main(self, robot):
        pipeline = self._create_pipeline2(robot)

        pin = '2z'
        r_person = 0

        with dai.Device(pipeline) as device:
            preview = device.getOutputQueue("preview", 4, False)
            tracklets = device.getOutputQueue("tracklets", 4, False)
            startTime = time.monotonic()
            counter = 0
            fps = 0
            frame = None
            found_people = {}
            sorted_tracked= {}
            while True:
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
                    robot.write_serial("5z")
                    
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
                    x_deviation = (int(robot.config.ps_xres/2)-obj_x_center)
                    # calculate the deviation from the center of the screen
                    if(abs(x_deviation)<robot.config.ps_tolerance): # is object in the middle of screen?
                        if abs(ymax<(0+robot.config.ps_bottom_buffer)):     # is object close to the bottom of the frame?
                            robot.write_serial('2z')
                            r_person=r_person+1
                            print('Old pin '+old_pin +' new pin '+pin)
                            if (r_person>3):
                                pin="2z"
                                robot.write_serial(pin)
                                print('Old pin '+old_pin +' new pin '+pin)
                                print('........................... reached objective')
                                print('waiting at objective reached')
                                robot.say("stop")
                                cv2.destroyAllWindows()
                                return True
                        else:
                            #if old_pin != pin:
                            pin="1z"
                            robot.write_serial(pin)
                            print('Old pin '+old_pin +' new pin '+pin)
                            print("........................... moving robot FORWARD")
                            robot.say("go")
                            r_person=0
                    else:
                        if (x_deviation>robot.config.ps_tolerance):
                            if old_pin !="3z" and x_deviation<175:
                                pin="3z"
                                robot.write_serial(pin)
                                print('Old pin '+old_pin +'  '+pin+'........................... turning left' )
                                robot.say("left")
                                r_person=0
                            if old_pin !="7z" and x_deviation>=175:
                                pin="7z"
                                robot.write_serial(pin)
                                print('Old pin '+old_pin +'  '+pin+'....... turning left on the spot' )
                                robot.say("spot left")
                                r_person=0
                        elif ((x_deviation*-1)>robot.config.ps_tolerance):
                            if old_pin !="4z" and abs(x_deviation)<175:
                                pin="4z"
                                robot.write_serial(pin)
                                print('Old pin '+old_pin +'  '+pin+'........................... turning right' )
                                robot.say("right")
                                r_person=0
                            if old_pin !="8z" and abs(x_deviation)>=175:
                                pin="8z"
                                robot.write_serial(pin)
                                print('Old pin '+old_pin +'  '+pin+'....... turning right on the spot' )
                                robot.say("spot right")
                                r_person=0
                cv2.putText(frame, "fps: {:.2f}".format(fps), (2, frame.shape[0] - 7), cv2.FONT_HERSHEY_TRIPLEX, 0.6, color)
                key_pressed=cv2.waitKey(1)
                if key_pressed==ord('z'):
                    print('Aborting search z ')
                    cv2.destroyAllWindows()
                    print('menu waiting for keyboard input')
                    pin='5z'
                    robot.write_serial(pin)
                    return False
                sorted_tracked.clear()
                cv2.imshow("tracker", frame)

    def _create_pipeline2(self, robot):
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
        camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(15)
        
        # testing MobileNet DetectionNetwork
        detectionNetwork.setBlobPath(robot.config.ps_nn_path)
        detectionNetwork.setConfidenceThreshold(self.model_threshold)
        detectionNetwork.input.setBlocking(False)
        objectTracker.setDetectionLabelsToTrack([self.model_id])  # track only person
        
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
        
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        #objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        #objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        camRgb.preview.link(detectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xlinkOut.input)
        if robot.config.ps_full_frame_tracking:
            camRgb.setPreviewKeepAspectRatio(False)
            camRgb.video.link(objectTracker.inputTrackerFrame)
        else:
            detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        detectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.out.link(trackerOut.input)
        return pipeline

    def _end(self, robot, success=False, give_biscuit_on_success=True):
        if not self.biscuit_mode:
            # Person Search Mode
            time.sleep(0.1)
            if success:
                images=load_images('/home/pi/MiniMax/Animations/found-person/')
                self.sound_manager.play_sound("found-person")
                self.animator.animate(images)
                #robot.say(f'I think I found a {self.label}')
            else:
                images=load_images('/home/pi/MiniMax/Animations/searchterminated/')
                self.sound_manager.play_sound("searchterminated")
                self.animator.animate(images)
                #robot.say("Search mode terminated")
            
            #robot.animate(1)
            print(f'well, hello there, I think I found a {self.label}')
            time.sleep(0.2)

            return RobotState.IDLE
        else:
            # Biscuit giving mode
            if not success:
                time.sleep(0.1)
                images=load_images('/home/pi/MiniMax/Animations/searchterminated/')
                self.sound_manager.play_sound("searchterminated")
                self.animate(images)
                #robot.say('Search mode Terminated.')
                #robot.animate(1)
                robot.play_sound('Radar_bleep_chirp')
                time.sleep(0.2)

                return RobotState.IDLE

            elif success and give_biscuit_on_success:
                self._give_biscuit(robot)

                return RobotState.BISCUIT
            else:
                time.sleep(0.1)
                images=load_images('/home/pi/MiniMax/Animations/objective/')
                self.sound_manager.play_sound("objective")
                self.animate(images)
                #robot.say("Objective reached.")
                #robot.animate(1)
                time.sleep(0.5)
                #robot.say("Woo Hoo, Yay.")
                #robot.animate(1)
                #time.sleep(0.2)
                robot.play_sound('celebrate1')
                time.sleep(0.2)
                robot.play_sound('Da_de_la')
                time.sleep(0.2)
                robot.play_sound('celebrate1')
                print('focus on terminal')
                time.sleep(0.1)

                return RobotState.PERSON_SEARCH
    
    def _give_biscuit(self, robot):
        time.sleep(1)
        robot.say('If you want a jelly bean, take one from my tray')
        #robot.animate(1)
        time.sleep(0.1)

        start=time.time()
        elapsed = 0
        while elapsed < robot.config.biscuit_wait_time:
            end = time.time()
            elapsed = end - start
            robot.say(str(int(robot.config.biscuit_wait_time - elapsed)))

            time.sleep(1)

        robot.say('The jelly beans are leaving now bye bye')

        #robot.animate(1)

        robot.write_serial('9z')  #do a 180 degree turn
        robot.write_serial('2z')
        
        time.sleep(1)

    def get_key(self) -> str:
        if self.biscuit_mode and self.label == 'person':
            return 'b'
        elif not self.biscuit_mode and self.label == 'person':
            return 'p'
        elif self.biscuit_mode and self.label == 'cup':
            return 'v'
        elif not self.biscuit_mode and self.label == 'cup':
            return 'c'
    
    def get_state(self) -> RobotState:
        if self.biscuit_mode and self.label == 'person':
            return RobotState.PERSON_SEARCH_GIVE_BISCUIT
        elif not self.biscuit_mode and self.label == 'person':
            return RobotState.PERSON_SEARCH_NO_GIVE_BISCUIT
        elif self.biscuit_mode and self.label == 'cup':
            return RobotState.CUP_SEARCH_GIVE_BISCUIT
        elif not self.biscuit_mode and self.label == 'cup':
            return RobotState.CUP_SEARCH_NO_GIVE_BISCUIT
        

