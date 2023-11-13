# MiniMax
Raspberry pi project:-
uses 1 x Raspberry pi3B+ for the movement
uses 1 x Raspberry pi4 8gb for the GUI and running the ML models
comms between pi's is via serial connection Tx/Rx
A mobile Robot Platform 4 wheels / 2 motor control channels for small and large robots with LCD screen and a female Android GUI
Very much a work in progress WIP

to run:-
cd MiniMax
python run.py

Uses code from Depthai library / examples for Age /gender ML detection and Emotional state detection.
uses code from Depthai / examples to detect people and follow them.
uses code from Ultraborg library to allow the Ultaborg servos & sensor controller to work
Uses pygame for the main GUI
Uses code form the Thunderborg library to make the motor controller work
All animations for the female Android GUI are in their respective folders in the Animations folder

Operational Modes are started by pressing a key on the wireless keyboard connected to the pi4.

e = Emotion detection mode
a = Age and gender detection
p = Detect and find person mode
c = Detect and find a cup / mug

To exit out of a mode, type 'z'

To exit the program type 'q'

There is also an Idle mode when the robot isn't doing anything.
during Idle mode GUI may randomly animate and talk.

All feedback regarding opperational modes and results are via the GUI with ML generated speech.

dependencies:-
pyserial
pygame
opencv
tensorflow
depthai
pyttsx3
tflite_runtime

Will produce a dependency.txt file with exact versions using pip freeze.
