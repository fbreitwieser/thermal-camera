# Thermal Camera Viewer

A Python application for viewing and recording thermal camera feeds from Infiray P2 and Topdon TC001 USB thermal cameras.

## Features

- Real-time thermal image display with temperature readings
- Multiple colormaps (turbo, jet, inferno, bone, viridis)
- Temperature markers for min/max/center temperatures
- **Colormap range locking** - lock the color scale to a fixed temperature range for comparison
- Video recording (AVI format), PNG snapshots
- Adjustable zoom (1x-5x) and contrast, median-filter, full-screen

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy

## Installation

```bash
pip install opencv-python numpy
```

## Usage

```bash
# Auto-detect camera on /dev/video2 (default)
python thermal-camera.py

# Auto-detect camera on specific device
python thermal-camera.py --device 0

# Explicitly specify camera type
python thermal-camera.py --camera infiray --device 2
python thermal-camera.py --camera topdon --device 0

# Specify output directory for recordings
python thermal-camera.py --output-dir ./my-recordings
```

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--device` | `-d` | Video device number (default: 2, check `v4l2-ctl --list-devices`) |
| `--camera` | `-c` | Camera type: `infiray` or `topdon` (default: auto-detect) |
| `--median-filter` | `-m` | Apply 5x5 spatial median filter to reduce noise |
| `--output-dir` | `-o` | Directory for recordings/snapshots (default: recordings) |
| `--no-cuda` | | Disable CUDA acceleration even if available |

## Key Bindings

| Key | Action |
|-----|--------|
| `+` / `-` | Zoom in / out |
| `c` / `d` | Increase / decrease contrast |
| `n` | Toggle median filter |
| `l` | Lock/unlock colormap range to current min/max |
| `[` / `]` | Adjust locked range minimum (when locked) |
| `;` / `'` | Adjust locked range maximum (when locked) |
| `r` / `t` | Start / stop recording |
| `f` | Toggle fullscreen |
| `p` | Take snapshot |
| `m` / `,` | Cycle colormaps forward / reverse |
| `h` | Cycle HUD: Normal → Extended → Help → Off |
| `Space` | Pause |
| `q` / `ESC` | Quit |

## Colormap Range Locking

Press `l` to lock the colormap to the current temperature range. This is useful when:
- Comparing temperatures across different scenes
- Monitoring a specific temperature range
- Avoiding color shifts as objects enter/leave the frame

When locked, use `[`/`]` to adjust the minimum and `;`/`'` to adjust the maximum.

## Finding Your Camera Device

Use `v4l2-ctl` to list available video devices:

```bash
v4l2-ctl --list-devices
```

Thermal cameras typically appear as UVC devices. The auto-detection will probe the device to determine the camera type automatically.

## Display

The viewer shows:
- **Center crosshairs** with temperature at center point
- **Red marker** at hottest point with temperature
- **Blue marker** at coldest point with temperature
- **HUD** with temperature range, average, and current settings

Extended HUD (`h` key) shows additional info:
- Current colormap
- Zoom level
- Contrast setting
- Median filter status
- Range lock status
- FPS (with CUDA indicator if enabled)

## Output Files

- **Recordings**: `recordings/thermal-camera-YYYY-MM-DD-HH_MM_SS.avi`
- **Snapshots**: `recordings/thermal-camera-YYYY-MM-DD-HH_MM_SS.png`

## Technical Details: Infiray P2 Stream Encoding

The Infiray P2 Pro presents itself as a standard UVC (USB Video Class) device, but embeds thermal radiometry data within the video stream. This encoding was reverse-engineered by the community.

### Frame Format

```
┌─────────────────────────┐
│                         │
│   Visual Image (YUYV)   │  256 x 192 pixels
│                         │
├─────────────────────────┤
│                         │
│   Thermal Data (raw)    │  256 x 192 pixels
│                         │
└─────────────────────────┘
        256 pixels wide
```

- **Resolution**: 256 x 384 total (256 x 192 image + 256 x 192 thermal)
- **Frame rate**: 25 Hz
- **Color format**: YUYV (YUV 4:2:2)

### Thermal Data Encoding

Each pixel in the thermal data region contains a 16-bit raw temperature value stored in little-endian format:

- **Channel 0 (Y)**: Low byte of temperature
- **Channel 1 (U/V)**: High byte of temperature
- **Combined**: `raw_value = low_byte + (high_byte << 8)`

The raw 16-bit value represents temperature in units of 1/64 Kelvin:
```
temperature_celsius = (raw_value / 64) - 273.15
```

### V4L2 Configuration
To access the raw thermal data, RGB conversion must be disabled:

```python
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
cap.set(cv2.CAP_PROP_FORMAT, -1)  # Raw format
```

## Acknowledgments

This project is based on [PyThermalCamera](https://github.com/leswright1977/PyThermalCamera) by Les Wright. The Infiray P2 stream format was reverse-engineered by [LeoDJ](https://chaos.social/@LeoDJ/109633033381602083), with community contributions from the [Infiray P2 Pro discussion thread](https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/?all) on the EEVblog forum.

## License

Apache 2.0
