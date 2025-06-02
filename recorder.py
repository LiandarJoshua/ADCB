import cv2
import numpy as np
import pyautogui
import time
import os
import keyboard  

def get_next_filename(folder, base_name="screen_recording", ext=".mp4"):
    i = 1
    while True:
        filename = f"{base_name}{i}{ext}"
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            return filename
        i += 1

output_folder = r"C:\Users\joshu\PythonSelenium\videotest"
output_filename = get_next_filename(output_folder)
output_path = os.path.join(output_folder, output_filename)

print("Saving recording to:", output_path)
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
Press 'c' to Cancel and Delete Recording
""")

while True:
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

    elif keyboard.is_pressed('c') and recording:
        print("Recording cancelled. Deleting file...")
        cancelled = True
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
    print("Recording cancelled and file deleted.")
else:
    print(f"Recording saved to: {output_path}")
