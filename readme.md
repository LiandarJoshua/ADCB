# Chrome Test Case Recorder

A Python application that automatically launches Chrome and records screen activity for Azure DevOps test case documentation. This tool integrates Chrome browser automation with screen recording to streamline test case creation and documentation.

## Features

- **Automated Chrome Launch**: Uses Selenium WebDriver to launch Chrome automatically
- **Screen Recording**: Records screen activity with customizable controls
- **Azure DevOps Integration**: Organizes recordings with project structure (Project/Version/Branch/Build/Feature/TestSuite)
- **Test Case Management**: Tracks test case information with version control
- **Flexible Recording Controls**: Start, pause, resume, cancel, or exit recording at any time
- **Automatic File Organization**: Creates structured folder hierarchy for easy navigation
- **Summary Generation**: Automatically creates summary files with test case metadata

## Prerequisites

### Required Python Packages
```bash
pip install selenium
pip install opencv-python
pip install pyautogui
pip install psutil
pip install keyboard
pip install numpy
```

### Additional Requirements
- **ChromeDriver**: Download and install ChromeDriver that matches your Chrome browser version
  - Download from: https://chromedriver.chromium.org/
  - Ensure ChromeDriver is in your system PATH or project directory
- **Chrome Browser**: Google Chrome must be installed on your system

## Installation

1. **Clone or download** the script to your local machine
2. **Install required packages** using pip (see prerequisites above)
3. **Setup ChromeDriver** and ensure it's accessible
4. **Modify the base recordings folder** in the script if needed:
   ```python
   self.base_recordings_folder = r"C:\Users\joshu\PythonSelenium\TestRecordings"
   ```

## Usage

### Running the Application
```bash
python chrome_recorder_launcher.py
```

### Main Menu Options

1. **Monitor for Chrome** - Automatically launches recorder when Chrome is detected
2. **Auto-launch Chrome and test recorder immediately** - Starts both Chrome and recorder
3. **Launch Chrome only** - Opens Chrome without recording (for testing)
4. **Setup test case information** - Configure project and test details
5. **View current configuration** - Display current test case settings
6. **Exit** - Close the application

### Test Case Configuration

The application will prompt you to setup test case information including:

- **Project Name**: Your project identifier
- **Version**: Semantic versioning (x.x.x format)
  - Keep current version
  - Increment patch (x.x.X)
  - Increment minor (x.X.0)
  - Increment major (X.0.0)
  - Set custom version
- **Feature/Epic**: Feature or epic name
- **Test Suite**: Test suite identifier
- **Test Case ID**: Specific test case identifier
- **Branch**: Git branch name
- **Build Number**: Build number for tracking

### Recording Controls

Once the recorder window opens, use these keyboard shortcuts:

| Key | Function |
|-----|----------|
| `s` | Start Recording |
| `p` | Pause Recording |
| `r` | Resume Recording |
| `q` | Stop and Save Recording |
| `c` | Cancel (works anytime - before or during recording) |
| `x` | Exit without recording |

## File Structure

The application creates an organized folder structure:

```
TestRecordings/
├── ProjectName/
│   ├── v1.0.0/
│   │   ├── main/
│   │   │   ├── Build_1/
│   │   │   │   ├── FeatureName/
│   │   │   │   │   ├── TestSuiteName/
│   │   │   │   │   │   ├── TestCaseID_20241201_143022_recording.mp4
│   │   │   │   │   │   └── TestCaseID_20241201_143022_recording_summary.txt
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── test_config.json
```

## Generated Files

### Recording Files
- **Video File**: `.mp4` format with timestamp and test case information
- **Summary File**: `.txt` file containing test case metadata and recording details

### Configuration File
- **test_config.json**: Stores current test case configuration for persistence between sessions

## Recording Specifications

- **Resolution**: 1280 x 720 pixels
- **Frame Rate**: 15 FPS
- **Format**: MP4 (H.264)
- **Codec**: mp4v

## Troubleshooting

### Common Issues

**ChromeDriver not found**
- Ensure ChromeDriver is downloaded and in your PATH
- Verify ChromeDriver version matches your Chrome browser version

**Permission errors**
- Run the script with appropriate permissions
- Ensure the recordings folder is writable

**Keyboard shortcuts not working**
- Make sure the recorder window has focus
- Try running as administrator if keyboard hooks fail

**Recording quality issues**
- Adjust screen resolution before recording
- Close unnecessary applications to improve performance

### Error Messages

**"Error launching Chrome"**
- Check ChromeDriver installation and PATH
- Verify Chrome browser is properly installed

**"Error launching recorder"**
- Check if required Python packages are installed
- Verify file permissions in the recordings directory

## Configuration

### Customizing Recording Settings

You can modify these settings in the `create_recorder_with_path` method:

```python
fps = 15.0                    # Frame rate
screen_size = (1280, 720)     # Resolution
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
```

### Changing Default Paths

Modify the base recordings folder:
```python
self.base_recordings_folder = r"your\custom\path\here"
```

## Advanced Usage

### Monitoring Mode
- Automatically detects when Chrome starts and launches recorder
- Useful for automated testing workflows
- Stops recording when Chrome closes

### Integration with CI/CD
- Can be integrated into automated test pipelines
- Configuration file allows for programmatic setup
- Structured output suitable for test reporting tools

## Limitations

- Windows-specific keyboard shortcuts (can be adapted for other OS)
- Requires Chrome browser and ChromeDriver
- Screen recording captures entire screen (not just Chrome window)
- Single recording session per launch

## Version History

- **v1.0.0**: Initial release with basic recording functionality
- **v1.1.0**: Added Azure DevOps integration and structured file organization
- **v1.2.0**: Enhanced recording controls and cancel-before-start functionality

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed correctly
3. Review error messages for specific guidance

