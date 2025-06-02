import time
import os
import psutil
import subprocess
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
from datetime import datetime

class ChromeRecorderLauncher:
    def __init__(self, recorder_script_path="recorder.py"):
        self.recorder_script_path = recorder_script_path
        self.chrome_driver = None
        self.monitoring = True
        self.recorder_process = None
        self.base_recordings_folder = 
        self.config_file = os.path.join(self.base_recordings_folder, "test_config.json")
        self.load_test_config()
        
    def load_test_config(self):
        """Load or create test configuration file"""
        default_config = {
            "current_version": "1.0.0",
            "project_name": "TestAutomation",
            "branch": "main", 
            "build_number": 1,
            "current_test_suite": "",
            "current_feature": "",
            "current_test_case": ""
        }
        
        os.makedirs(self.base_recordings_folder, exist_ok=True)
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except:
                self.config = default_config
        else:
            self.config = default_config
            self.save_test_config()
    
    def save_test_config(self):
        """Save current test configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def setup_test_case_info(self):
        """Interactive setup for test case information"""
        print("\n" + "="*50)
        print("TEST CASE SETUP")
        print("="*50)
        
        # Project info
        project = input(f"Project Name [{self.config['project_name']}]: ").strip()
        if project:
            self.config['project_name'] = project
            
        # Version info
        print(f"\nCurrent Version: {self.config['current_version']}")
        print("1. Keep current version")
        print("2. Increment patch (x.x.X)")
        print("3. Increment minor (x.X.0)")
        print("4. Increment major (X.0.0)")
        print("5. Set custom version")
        
        version_choice = input("Choose version option (1-5): ").strip()
        if version_choice == "2":
            self.increment_version("patch")
        elif version_choice == "3":
            self.increment_version("minor")
        elif version_choice == "4":
            self.increment_version("major")
        elif version_choice == "5":
            custom_version = input("Enter version (x.x.x format): ").strip()
            if self.validate_version(custom_version):
                self.config['current_version'] = custom_version
        
        # Test organization
        feature = input(f"Feature/Epic [{self.config.get('current_feature', '')}]: ").strip()
        if feature:
            self.config['current_feature'] = feature
            
        test_suite = input(f"Test Suite [{self.config.get('current_test_suite', '')}]: ").strip()
        if test_suite:
            self.config['current_test_suite'] = test_suite
            
        test_case = input(f"Test Case ID [{self.config.get('current_test_case', '')}]: ").strip()
        if test_case:
            self.config['current_test_case'] = test_case
        
        # Branch info
        branch = input(f"Branch [{self.config['branch']}]: ").strip()
        if branch:
            self.config['branch'] = branch
            
        # Build number
        build_input = input(f"Build Number [{self.config['build_number']}]: ").strip()
        if build_input and build_input.isdigit():
            self.config['build_number'] = int(build_input)
        
        self.save_test_config()
        self.display_current_config()
    
    def validate_version(self, version):
        """Validate version format (x.x.x)"""
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))
    
    def increment_version(self, version_type="patch"):
        """Increment version number (major.minor.patch)"""
        try:
            major, minor, patch = map(int, self.config["current_version"].split('.'))
            
            if version_type.lower() == "major":
                major += 1
                minor = 0
                patch = 0
            elif version_type.lower() == "minor":
                minor += 1
                patch = 0
            else:  # patch
                patch += 1
                
            self.config["current_version"] = f"{major}.{minor}.{patch}"
            self.save_test_config()
        except:
            print("Error incrementing version. Using current version.")
    
    def display_current_config(self):
        """Display current test configuration"""
        print("\n" + "="*50)
        print("CURRENT TEST CONFIGURATION")
        print("="*50)
        print(f"Project: {self.config['project_name']}")
        print(f"Version: {self.config['current_version']}")
        print(f"Branch: {self.config['branch']}")
        print(f"Build: {self.config['build_number']}")
        print(f"Feature: {self.config.get('current_feature', 'Not set')}")
        print(f"Test Suite: {self.config.get('current_test_suite', 'Not set')}")
        print(f"Test Case: {self.config.get('current_test_case', 'Not set')}")
        print("="*50)
    
    def get_recording_folder_path(self):
        """Generate Azure DevOps style folder structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build folder structure: Project/Version/Branch/Build/Feature/TestSuite/
        folder_parts = [
            self.base_recordings_folder,
            self.config['project_name'],
            f"v{self.config['current_version']}",
            self.config['branch'],
            f"Build_{self.config['build_number']}"
        ]
        
        if self.config.get('current_feature'):
            folder_parts.append(self.config['current_feature'])
            
        if self.config.get('current_test_suite'):
            folder_parts.append(self.config['current_test_suite'])
        
        recording_folder = os.path.join(*folder_parts)
        os.makedirs(recording_folder, exist_ok=True)
        
        return recording_folder
    
    def get_recording_filename(self):
        """Generate recording filename with test case info"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename_parts = []
        
        if self.config.get('current_test_case'):
            filename_parts.append(self.config['current_test_case'])
        else:
            filename_parts.append("TestCase")
            
        filename_parts.append(timestamp)
        filename_parts.append("recording")
        
        return "_".join(filename_parts) + ".mp4"
    
    def create_recorder_with_path(self):
        """Create a modified recorder script with the correct output path"""
        recording_folder = self.get_recording_folder_path()
        recording_filename = self.get_recording_filename()
        full_recording_path = os.path.join(recording_folder, recording_filename)
        
        # Create a temporary recorder script with the custom path
        temp_recorder_path = "temp_recorder.py"
        
        recorder_template = f'''import cv2
import numpy as np
import pyautogui
import time
import os
import keyboard  
from datetime import datetime

# Auto-generated recording configuration
output_path = r"{full_recording_path}"
print("="*60)
print("AZURE DEVOPS TEST CASE RECORDING")
print("="*60) 
print("Project: {self.config['project_name']}")
print("Version: {self.config['current_version']}")
print("Branch: {self.config['branch']}")
print("Build: {self.config['build_number']}")
print("Feature: {self.config.get('current_feature', 'Not set')}")
print("Test Suite: {self.config.get('current_test_suite', 'Not set')}")
print("Test Case: {self.config.get('current_test_case', 'Not set')}")
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
    print(f"Recording saved to: {{output_path}}")
    
    # Create a summary file
    summary_path = output_path.replace('.mp4', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Test Case Recording Summary\\n")
        f.write("="*40 + "\\n")
        f.write(f"Project: {self.config['project_name']}\\n")
        f.write(f"Version: {self.config['current_version']}\\n")
        f.write(f"Branch: {self.config['branch']}\\n")
        f.write(f"Build: {self.config['build_number']}\\n")
        f.write(f"Feature: {self.config.get('current_feature', 'Not set')}\\n")
        f.write(f"Test Suite: {self.config.get('current_test_suite', 'Not set')}\\n")
        f.write(f"Test Case: {self.config.get('current_test_case', 'Not set')}\\n")
        f.write(f"Recording Date: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}\\n")
        f.write(f"Recording File: {{os.path.basename(output_path)}}\\n")
    
    print(f"Summary saved to: {{summary_path}}")
'''
        
        with open(temp_recorder_path, 'w') as f:
            f.write(recorder_template)
        
        return temp_recorder_path, full_recording_path
        
    def is_chrome_running(self):
        """Check if Chrome process is running"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'chrome' in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    
    def launch_recorder(self):
        """Launch the recorder.py script with custom path"""
        try:
            if self.recorder_process and self.recorder_process.poll() is None:
                print("Recorder is already running.")
                return True
            
            # Create custom recorder with test case path
            temp_recorder_path, recording_path = self.create_recorder_with_path()
            
            print(f"Launching recorder for test case...")
            print(f"Recording will be saved to: {recording_path}")
            
            # Launch custom recorder script in a new process
            self.recorder_process = subprocess.Popen(
                ['python', temp_recorder_path],
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            print("Test case recorder launched successfully!")
            return True
            
        except Exception as e:
            print(f"Error launching recorder: {e}")
            return False
    
    def stop_recorder(self):
        """Stop the recorder script"""
        try:
            if self.recorder_process and self.recorder_process.poll() is None:
                print("Stopping recorder script...")
                self.recorder_process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    self.recorder_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Force killing recorder process...")
                    self.recorder_process.kill()
                
                self.recorder_process = None
                print("Recorder script stopped.")
                
                # Clean up temp recorder file
                if os.path.exists("temp_recorder.py"):
                    try:
                        os.remove("temp_recorder.py")
                    except:
                        pass
            else:
                print("Recorder is not running.")
                
        except Exception as e:
            print(f"Error stopping recorder: {e}")
    
    def launch_chrome_with_selenium(self):
        """Launch Chrome using Selenium WebDriver"""
        try:
            chrome_options = Options()
            # Remove headless mode so Chrome window is visible
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            print("Launching Chrome with Selenium...")
            self.chrome_driver = webdriver.Chrome(options=chrome_options)
            
            # Navigate to a default page (you can change this)
            self.chrome_driver.get("https://www.google.com")
            
            return True
            
        except Exception as e:
            print(f"Error launching Chrome: {e}")
            return False
    
    def monitor_chrome(self):
        """Monitor Chrome process and launch recorder when detected"""
        chrome_was_running = False
        
        print("Starting Chrome monitor for test case recording...")
        print("The script will automatically launch test case recorder when Chrome opens.")
        print("Press Ctrl+C to stop monitoring.")
        
        try:
            while self.monitoring:
                chrome_is_running = self.is_chrome_running()
                
                # Chrome just started
                if chrome_is_running and not chrome_was_running:
                    print("Chrome detected! Launching test case recorder...")
                    self.launch_recorder()
                    chrome_was_running = True
                
                # Chrome just closed
                elif not chrome_is_running and chrome_was_running:
                    print("Chrome closed! Stopping recorder...")
                    self.stop_recorder()
                    chrome_was_running = False
                
                time.sleep(2)  # Check every 2 seconds
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            self.stop_recorder()
            if self.chrome_driver:
                self.chrome_driver.quit()
    
    def auto_launch_and_record(self):
        """Automatically launch Chrome and start recorder"""
        print("Auto-launching Chrome and test case recorder...")
        
        if self.launch_chrome_with_selenium():
            # Launch recorder immediately after Chrome starts
            if self.launch_recorder():
                print(f"\nChrome launched and test case recorder started!")
                print("\nRecorder Controls (in the recorder window):")
                print("Press 's' to Start Recording")
                print("Press 'p' to Pause")
                print("Press 'r' to Resume") 
                print("Press 'q' to Stop and Save Recording")
                print("Press 'c' to Cancel and Delete Recording")
                print("\nPress Enter here to close Chrome and stop recorder...")
                
                # Wait for user input to stop
                input()
                
                # Stop recorder and close Chrome
                self.stop_recorder()
                if self.chrome_driver:
                    self.chrome_driver.quit()
                    print("Chrome closed.")
            else:
                print("Failed to launch recorder.")
                if self.chrome_driver:
                    self.chrome_driver.quit()
        else:
            print("Failed to launch Chrome.")
    
    def launch_chrome_only(self):
        """Just launch Chrome without recorder (for testing)"""
        print("Launching Chrome only...")
        
        if self.launch_chrome_with_selenium():
            print("Chrome launched successfully!")
            print("Press Enter to close Chrome...")
            input()
            
            if self.chrome_driver:
                self.chrome_driver.quit()
                print("Chrome closed.")
        else:
            print("Failed to launch Chrome.")

def main():
    launcher = ChromeRecorderLauncher()
    
    print("\nAZURE DEVOPS TEST CASE RECORDER")
    print("=" * 50)
    
    # Setup test case information first
    setup_choice = input("Do you want to setup/modify test case information? (y/n): ").strip().lower()
    if setup_choice == 'y':
        launcher.setup_test_case_info()
    else:
        launcher.display_current_config()
    
    print("\nChoose an option:")
    print("1. Monitor for Chrome (launch test recorder when Chrome opens)")
    print("2. Auto-launch Chrome and test recorder immediately")
    print("3. Launch Chrome only (for testing)")
    print("4. Setup test case information")
    print("5. View current configuration")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        launcher.monitor_chrome()
    elif choice == "2":
        launcher.auto_launch_and_record()
    elif choice == "3":
        launcher.launch_chrome_only()
    elif choice == "4":
        launcher.setup_test_case_info()
        main()  # Return to main menu
    elif choice == "5":
        launcher.display_current_config()
        main()  # Return to main menu
    elif choice == "6":
        print("Exiting...")
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()