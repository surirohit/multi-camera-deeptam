# Rosbag Extractor

## Usage


```bash
python rosbag_extractor.py --rosbag_path <path/to/file.bag> --output_path <path/to/output/dir>
```

Output directory will be created if it doesn't exist.

Currently, only cameras 0, 2, 4 and 6 are being used. Camera 8 is not synchronized with the rest. If you need to change the cameras to use, look at line 13 of the `rosbag_extractor.py`
