import cv2
import numpy as np
import pyautogui
import time
import os
import keyboard  
from datetime import datetime

# Auto-generated recording configuration
output_path = r"C:\Users\joshu\PythonSelenium\TestRecordings\TestingThing\v1.5.6\hello\Build_1\9\1\12_20250602_081536_recording.mp4"
print("="*60)
print("AZURE DEVOPS TEST CASE RECORDING")
print("="*60) 
print("Project: TestingThing")
print("Version: 1.5.6")
print("Branch: hello")
print("Build: 1")
print("Feature: 9")
print("Test Suite: 1")
print("Test Case: 12")
print("="*60)
print("Saving recording to:", output_path)
print("="*60)

fps = 15.0
screen_size = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, screen_size)

recording = False
paused = False
cancelled = False

print("""
Screen Recorder Controls:
Press 's' to Start Recording
Press 'p' to Pause
Press 'r' to Resume
Press 'q' to Stop and Save Recording
Press 'c' to Cancel (works anytime - before or during recording)
Press 'x' to Exit without recording
""")

while True:
    # Check for cancel or exit at any time (before or during recording)
    if keyboard.is_pressed('c'):
        print("Recording cancelled!")
        cancelled = True
        break
    
    if keyboard.is_pressed('x'):
        print("Exiting without recording.")
        cancelled = True
        break

    if keyboard.is_pressed('s') and not recording:
        print("Recording started.")
        recording = True
        paused = False
        time.sleep(0.5)

    elif keyboard.is_pressed('p') and recording and not paused:
        print("Paused.")
        paused = True
        time.sleep(0.5)

    elif keyboard.is_pressed('r') and recording and paused:
        print("Resumed.")
        paused = False
        time.sleep(0.5)

    elif keyboard.is_pressed('q') and recording:
        print("Stopped and saved.")
        break

    if recording and not paused:
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, screen_size)
        out.write(frame)

out.release()

if cancelled:
    if os.path.exists(output_path):
        os.remove(output_path)
    print("Recording cancelled and file deleted (if any).")
else:
    print(f"Recording saved to: {output_path}")
    
    # Create a summary file
    summary_path = output_path.replace('.mp4', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Test Case Recording Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Project: TestingThing\n")
        f.write(f"Version: 1.5.6\n")
        f.write(f"Branch: hello\n")
        f.write(f"Build: 1\n")
        f.write(f"Feature: 9\n")
        f.write(f"Test Suite: 1\n")
        f.write(f"Test Case: 12\n")
        f.write(f"Recording Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Recording File: {os.path.basename(output_path)}\n")
    
    print(f"Summary saved to: {summary_path}")
