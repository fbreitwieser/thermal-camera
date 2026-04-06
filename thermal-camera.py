#!/usr/bin/env python3
"""
Thermal Camera Viewer for Infiray P2 & Topdon TC001

Usage:
    python thermal_camera.py --device 2                    # Auto-detect camera on /dev/video2
    python thermal_camera.py --camera infiray --device 2   # Infiray P2 on /dev/video2
    python thermal_camera.py --camera topdon --device 0    # Topdon TC001 on /dev/video0

Key Bindings:
    +/- : Zoom In/Out
    c/d : Increase/Decrease Contrast
    n   : Toggle Median Filter
    r/t : Start/Stop Recording
    f   : Toggle Fullscreen
    p   : Snapshot
    m/,  : Cycle through ColorMaps (forward/reverse)
    h   : Cycle HUD: Help → Off → Normal
    l   : Lock/unlock colormap range to current min/max
    [/] : Adjust locked range minimum
    ;/' : Adjust locked range maximum
    Space : Pause
    q/ESC : Quit
"""

import argparse
import io
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CameraConfig:
    """Camera and display configuration."""
    # Camera type: 'infiray' or 'topdon'
    camera_type: str = 'infiray'

    # Sensor dimensions
    sensor_width: int = 256
    sensor_height: int = 192

    # Display settings
    scale: int = 3
    alpha: float = 1.0  # Contrast (0.0 - 3.0)
    colormap_index: int = 0

    # Colormap range locking (None = auto-scale)
    range_locked: bool = False
    locked_min_temp: Optional[float] = None
    locked_max_temp: Optional[float] = None

    # State flags
    hud_mode: int = 0  # 0=normal HUD, 1=help page, 2=no HUD
    fullscreen: bool = False
    recording: bool = False
    paused: bool = False
    use_median_filter: bool = False

    @property
    def display_width(self) -> int:
        return self.sensor_width * self.scale

    @property
    def display_height(self) -> int:
        return self.sensor_height * self.scale


# Colormap definitions: (OpenCV colormap constant, name, needs RGB conversion)
# See https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html
COLORMAPS = [
    (cv2.COLORMAP_TURBO, "turbo", False),
    (cv2.COLORMAP_JET, "jet", False),
    #(cv2.COLORMAP_HOT, "hot", False),
    #(cv2.COLORMAP_MAGMA, "magma", False),
    (cv2.COLORMAP_INFERNO, "inferno", False),
    #(cv2.COLORMAP_PLASMA, "plasma", False),
    (cv2.COLORMAP_BONE, "bone", False),
    #(cv2.COLORMAP_SPRING, "spring", False),
    #(cv2.COLORMAP_AUTUMN, "autumn", False),
    (cv2.COLORMAP_VIRIDIS, "viridis", False),
    #(cv2.COLORMAP_PARULA, "parula", False),
    #(cv2.COLORMAP_RAINBOW, "inv rainbow", True),
]


# =============================================================================
# Utility Functions
# =============================================================================

def is_raspberry_pi() -> bool:
    """Check if running on a Raspberry Pi."""
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except Exception:
        return False


def raw_to_celsius(raw_value: float) -> float:
    """Convert raw thermal value to Celsius."""
    return (round(((raw_value / 64.0) - 273.15) * 2)) / 2.0


def celsius_to_raw(celsius: float) -> float:
    """Convert Celsius to raw thermal value."""
    return (celsius + 273.15) * 64.0


def format_temperature(temp: float) -> str:
    """Format temperature for display."""
    return f"{temp} C"


def detect_camera_type(device: int) -> Optional[str]:
    """
    Auto-detect camera type by probing device characteristics.

    Returns:
        'infiray', 'topdon', or None if detection fails.
    """
    # Try Infiray P2 first (256x384 raw format)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L)
    if not cap.isOpened():
        # Try path format for Topdon
        cap = cv2.VideoCapture(f'/dev/video{device}', cv2.CAP_V4L)
        if not cap.isOpened():
            return None

    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    # Read a test frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    height, width = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) == 3 else 1
    dtype = frame.dtype

    print(f"  Frame: {width}x{height}, {channels} channel(s), dtype={dtype}")

    # Infiray P2: 256x384 (image + thermal stacked)
    if width == 256 and height == 384:
        # Check if it looks like YUYV data (Topdon)
        if channels == 2:
            print("  Format: YUYV (2 channels)")
        return 'infiray'

    # Topdon TC001: 256x384 in YUYV format (splits differently)
    # TC001 typically returns 256x384 but with different data layout
    if width == 256 and height == 384:
        if channels == 2:
            print("  Format: YUYV (2 channels)")
            return 'topdon'

    # Topdon might also appear as 512x384 or similar
    if width == 512 and height == 384:
        print("  Format: 512x384 (Topdon TC001)")
        return 'topdon'

    # Default to infiray for 256-width thermal cameras
    if width == 256:
        return 'infiray'

    print(f"  Unknown format: {width}x{height}")
    return None


class CameraInitError(Exception):
    """Raised when camera initialization fails."""
    pass


# =============================================================================
# Thermal Data Processing
# =============================================================================

@dataclass
class ThermalData:
    """Processed thermal data from a frame."""
    center_temp: float
    max_temp: float
    min_temp: float
    avg_temp: float
    max_pos: Tuple[int, int]  # (row, col)
    min_pos: Tuple[int, int]  # (row, col)


def extract_thermal_data(frame: np.ndarray, height: int, camera_type: str = 'infiray') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract image and thermal data from raw frame.
    
    Args:
        frame: Raw frame from camera
        height: Sensor height (192)
        camera_type: 'infiray' or 'topdon'
    
    Returns:
        Tuple of (image_data, thermal_data)
    """
    if camera_type == 'topdon':
        # TC001: use array_split to divide frame in half
        imdata, thdata = np.array_split(frame, 2)
    else:
        # Infiray: manual split, handle single-plane fallback
        if frame.shape[0] >= height * 2:
            imdata = frame[:height, :]
            thdata = frame[height:height * 2, :]
        else:
            imdata = frame[:height, :]
            thdata = imdata
    return imdata, thdata


def process_thermal_data(thdata: np.ndarray, camera_type: str = 'infiray', width: int = 256, use_median_filter: bool = False) -> ThermalData:
    """Process raw thermal data and extract temperature information.
    
    Args:
        thdata: Raw thermal data array
        camera_type: 'infiray' for Infiray P2 or 'topdon' for TC001
        width: Sensor width for TC001 position calculation
        use_median_filter: Apply 3x3 spatial median filter to reduce noise
    """
    if camera_type == 'topdon':
        # TC001: Uses high-byte channel for finding extremes, then combines hi+lo*256
        # Center temperature
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        rawtemp = hi + lo * 256
        center_temp = round((rawtemp / 64) - 273.15, 1)
        
        # Max temperature - find using high byte channel
        lomax = thdata[..., 1].max()
        posmax = thdata[..., 1].argmax()
        mcol, mrow = divmod(posmax, width)
        himax = thdata[mcol][mrow][0]
        maxraw = himax + lomax * 256
        max_temp = round((maxraw / 64) - 273.15, 1)
        max_pos = (mcol, mrow)
        
        # Min temperature - find using high byte channel
        lomin = thdata[..., 1].min()
        posmin = thdata[..., 1].argmin()
        lcol, lrow = divmod(posmin, width)
        himin = thdata[lcol][lrow][0]
        minraw = himin + lomin * 256
        min_temp = round((minraw / 64) - 273.15, 1)
        min_pos = (lcol, lrow)
        
        # Average temperature
        loavg = thdata[..., 1].mean()
        hiavg = thdata[..., 0].mean()
        avgraw = hiavg + loavg * 256
        avg_temp = round((avgraw / 64) - 273.15, 1)
    else:
        # Infiray P2: little-endian 16-bit values
        if len(thdata.shape) == 3 and thdata.shape[2] >= 2:
            raw16 = thdata[..., 0].astype(np.uint16) + (thdata[..., 1].astype(np.uint16) << 8)
        else:
            raw16 = thdata.astype(np.uint16)
        
        # Optionally apply spatial median filter to reduce noise
        if use_median_filter:
            raw16_work = cv2.medianBlur(raw16, 5)
        else:
            raw16_work = raw16
        
        # Center temperature
        center_y, center_x = raw16_work.shape[0] // 2, raw16_work.shape[1] // 2
        center_temp = raw_to_celsius(raw16_work[center_y, center_x])
        
        # Max temperature
        max_pos = np.unravel_index(np.argmax(raw16_work), raw16_work.shape)
        max_temp = raw_to_celsius(raw16_work[max_pos])
        
        # Min temperature
        min_pos = np.unravel_index(np.argmin(raw16_work), raw16_work.shape)
        min_temp = raw_to_celsius(raw16_work[min_pos])
        
        # Average temperature (use unfiltered for accuracy)
        avg_temp = raw_to_celsius(raw16.mean())
    
    return ThermalData(
        center_temp=center_temp,
        max_temp=max_temp,
        min_temp=min_temp,
        avg_temp=avg_temp,
        max_pos=max_pos,
        min_pos=min_pos,
    )


def convert_to_bgr(imdata: np.ndarray, camera_type: str = 'infiray') -> np.ndarray:
    """Convert thermal image data to BGR format for display."""
    if camera_type == 'topdon':
        # TC001: always YUV to BGR
        return cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
    else:
        # Infiray: handle various formats
        if len(imdata.shape) == 3:
            if imdata.shape[2] == 2:
                return cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
            elif imdata.shape[2] == 3:
                return imdata
            elif imdata.shape[2] == 4:
                return cv2.cvtColor(imdata, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(imdata, cv2.COLOR_GRAY2BGR)


# =============================================================================
# Rendering Functions
# =============================================================================

def apply_colormap(image: np.ndarray, colormap_index: int) -> Tuple[np.ndarray, str]:
    """Apply colormap to image and return the result with colormap name."""
    if colormap_index >= len(COLORMAPS):
        return image, "None"
    
    cmap, name, needs_rgb = COLORMAPS[colormap_index]
    result = cv2.applyColorMap(image, cmap)
    
    if needs_rgb:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result, name


def draw_crosshairs(image: np.ndarray, center_x: int, center_y: int, temp: float) -> None:
    """Draw crosshairs at center with temperature label."""
    # White outline
    cv2.line(image, (center_x, center_y + 20), (center_x, center_y - 20), (255, 255, 255), 2)
    cv2.line(image, (center_x + 20, center_y), (center_x - 20, center_y), (255, 255, 255), 2)
    # Black inner line
    cv2.line(image, (center_x, center_y + 20), (center_x, center_y - 20), (0, 0, 0), 1)
    cv2.line(image, (center_x + 20, center_y), (center_x - 20, center_y), (0, 0, 0), 1)
    
    # Temperature label
    label = format_temperature(temp)
    pos = (center_x + 10, center_y - 10)
    cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def draw_temperature_marker(image: np.ndarray, x: int, y: int, temp: float, 
                           color: Tuple[int, int, int]):
    """Draw a temperature marker at the specified position."""
    h, w = image.shape[:2]
    cv2.circle(image, (x, y), 5, (0, 0, 0), 2)
    cv2.circle(image, (x, y), 5, color, -1)
    
    label = format_temperature(temp)
    
    # Estimate label size (approx 8 pixels per char at 0.45 scale, 15 pixels height)
    label_width = len(label) * 8
    label_height = 15
    
    # Default: label to the right and below
    label_x = x + 10
    label_y = y + 5
    
    # Adjust if too close to right edge
    if label_x + label_width > w:
        label_x = x - label_width - 5
    
    # Adjust if too close to bottom edge
    if label_y + label_height > h:
        label_y = y - 10
    
    # Adjust if too close to top edge
    if label_y - label_height < 0:
        label_y = y + label_height + 5
    
    # Adjust if too close to left edge
    if label_x < 0:
        label_x = 5
    
    pos = (label_x, label_y)
    cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


# Shared HUD / help style constants
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX   # mono-spaced look
_HUD_FONT_SCALE = 0.5
_HUD_LINE_H = 16
_HUD_PAD = 5
_HUD_BG_ALPHA = 0.55  # background opacity (0 = invisible, 1 = solid)
_HUD_COLOR = (200, 200, 200)
_HUD_HIGHLIGHT = (0, 255, 255)
_HUD_REC_COLOR = (40, 40, 255)


def _draw_overlay_box(image: np.ndarray, lines: List[str],
                      colors: List[Tuple[int, int, int]],
                      columns: Optional[List[Tuple[str, str]]] = None) -> None:
    """Draw a semi-transparent box with text lines on *image* (in-place).

    The box is placed in the lower-left corner with a small margin.

    If *columns* is provided it must be a list of (left_text, right_text)
    pairs rendered as two aligned columns.  *lines* and *colors* are then
    ignored for content but *columns* length determines row count.
    """
    img_h, img_w = image.shape[:2]
    margin = 6  # gap between box edge and image edge
    two_col = columns is not None

    # --- measure box width ------------------------------------------------
    col_gap = 8
    max_left = 0
    max_right = 0
    if two_col:
        for left, right in columns:
            (lw, _), _ = cv2.getTextSize(left, _HUD_FONT, _HUD_FONT_SCALE, 1)
            (rw, _), _ = cv2.getTextSize(right, _HUD_FONT, _HUD_FONT_SCALE, 1)
            max_left = max(max_left, lw)
            max_right = max(max_right, rw)
        box_w = _HUD_PAD + max_left + col_gap + max_right + _HUD_PAD
        n_rows = len(columns)
    else:
        max_w = 0
        for text in lines:
            (tw, _), _ = cv2.getTextSize(text, _HUD_FONT, _HUD_FONT_SCALE, 1)
            max_w = max(max_w, tw)
        box_w = max_w + _HUD_PAD * 2
        n_rows = len(lines)

    box_h = _HUD_LINE_H * n_rows + _HUD_PAD * 2

    # Position: lower-left corner
    x0 = margin
    y0 = img_h - box_h - margin

    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, _HUD_BG_ALPHA, image, 1 - _HUD_BG_ALPHA, 0, image)

    # --- draw text --------------------------------------------------------
    y = y0 + _HUD_PAD + _HUD_LINE_H
    if two_col:
        right_x = x0 + _HUD_PAD + max_left + col_gap
        for (left, right), color in zip(columns, colors):
            cv2.putText(image, left, (x0 + _HUD_PAD, y), _HUD_FONT,
                        _HUD_FONT_SCALE, color, 1, cv2.LINE_AA)
            cv2.putText(image, right, (right_x, y), _HUD_FONT,
                        _HUD_FONT_SCALE, color, 1, cv2.LINE_AA)
            y += _HUD_LINE_H
    else:
        for text, color in zip(lines, colors):
            cv2.putText(image, text, (x0 + _HUD_PAD, y), _HUD_FONT,
                        _HUD_FONT_SCALE, color, 1, cv2.LINE_AA)
            y += _HUD_LINE_H


def draw_hud(image: np.ndarray, config: CameraConfig, thermal: ThermalData,
             colormap_name: str, snaptime: str, elapsed: str, extra_info: bool,
             fps: float = 0.0) -> None:
    """Draw the heads-up display with camera information."""
    lines: List[str] = [
        f"Temp {round(thermal.min_temp)}-{round(thermal.max_temp)}C, avg {thermal.avg_temp}C"
    ]
    if extra_info:
        lines.append(f"Colormap: {colormap_name} [m]")
        lines.append(f"Zoom: {config.scale}x [+/-]")
        lines.append(f"Contrast: {config.alpha} [c/d]")
        lines.append(f"Median filter: {'On' if config.use_median_filter else 'Off'} [n]")
        if config.range_locked:
            lines.append(f"Range: {config.locked_min_temp}-{config.locked_max_temp}C [l]")
        else:
            lines.append("Range: auto [l]")
        lines.append(f"FPS: {fps:.1f}")

    colors: List[Tuple[int, int, int]] = [_HUD_COLOR] * len(lines)
    if config.recording:
        lines.append(f"Recording: {elapsed}")
        colors.append(_HUD_REC_COLOR)

    _draw_overlay_box(image, lines, colors)


def draw_help_page(image: np.ndarray) -> None:
    """Draw a help overlay with key bindings."""
    columns: List[Tuple[str, str]] = [
        ("Key",   "Action"),
        ("+/-",   "Zoom In/Out"),
        ("c/d",   "Contrast Up/Down"),
        ("n",     "Toggle Median Filter"),
        ("l",     "Lock/Unlock Range"),
        ("[/]",   "Adjust Range Min"),
        (";/'",   "Adjust Range Max"),
        ("r/t",   "Start/Stop Recording"),
        ("f",     "Toggle Fullscreen"),
        ("p",     "Snapshot"),
        ("m/,",   "Cycle ColorMaps"),
        ("h",     "Cycle HUD"),
        ("Space", "Pause"),
        ("q/ESC", "Quit"),
    ]
    colors: List[Tuple[int, int, int]] = [
        _HUD_HIGHLIGHT if i == 0 else _HUD_COLOR
        for i in range(len(columns))
    ]
    _draw_overlay_box(image, [], colors, columns=columns)


# =============================================================================
# Main Application Class
# =============================================================================

class ThermalCameraViewer:
    """Main thermal camera viewer application."""

    def __init__(self, device: int, camera_type: Optional[str] = None,
                 use_median_filter: bool = False, output_dir: str = 'recordings'):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_pi = is_raspberry_pi()

        # Auto-detect camera type if not specified
        if camera_type is None:
            print(f"Auto-detecting camera on /dev/video{device}...")
            camera_type = detect_camera_type(device)
            if camera_type is None:
                raise CameraInitError(
                    f"Could not auto-detect camera type on /dev/video{device}. "
                    "Please specify --camera infiray or --camera topdon."
                )
            print(f"Detected camera type: {camera_type}")

        self.config = CameraConfig(camera_type=camera_type, use_median_filter=use_median_filter)

        # Recording state
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_filename: str = ""
        self.record_start_time: float = 0.0
        self.elapsed_time: str = "00:00:00"
        self.snaptime: str = "None"

        # FPS tracking
        self.frame_times: deque[float] = deque(maxlen=30)
        self.current_fps: float = 0.0

        # Position hysteresis - keep previous positions if temp unchanged
        self.prev_max_temp: Optional[float] = None
        self.prev_max_pos: Optional[Tuple[int, int]] = None
        self.prev_min_temp: Optional[float] = None
        self.prev_min_pos: Optional[Tuple[int, int]] = None

        # Initialize camera
        self.cap = self._init_camera()
        self._init_window()
    
    def _init_camera(self) -> cv2.VideoCapture:
        """Initialize the video capture device.

        Raises:
            CameraInitError: If camera cannot be opened or configured.
        """
        device_path = f'/dev/video{self.device}'

        if self.config.camera_type == 'topdon':
            # TC001: use path string format
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L)
            if not cap.isOpened():
                raise CameraInitError(
                    f"Failed to open Topdon TC001 on {device_path}. "
                    "Check device number with: v4l2-ctl --list-devices"
                )
            # Pi-specific RGB conversion handling
            if self.is_pi:
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
            else:
                cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
        else:
            # Infiray P2
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L)
            if not cap.isOpened():
                raise CameraInitError(
                    f"Failed to open Infiray P2 on {device_path}. "
                    "Check device number with: v4l2-ctl --list-devices"
                )
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
            cap.set(cv2.CAP_PROP_FORMAT, -1)  # Raw video stream

        # Verify we can read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise CameraInitError(
                f"Camera opened but failed to read frames from {device_path}. "
                "The device may be in use or not a thermal camera."
            )

        return cap
    
    def _init_window(self) -> None:
        """Initialize the display window."""
        cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Thermal', self.config.display_width, self.config.display_height)

    def start_recording(self) -> None:
        """Start video recording."""
        timestamp = time.strftime("%Y-%m-%d-%H_%M_%S")
        self.record_filename = os.path.join(self.output_dir, f"thermal-camera-{timestamp}.avi")
        self.video_writer = cv2.VideoWriter(
            self.record_filename,
            cv2.VideoWriter.fourcc(*'XVID'),
            25,
            (self.config.display_width, self.config.display_height)
        )
        self.config.recording = True
        self.record_start_time = time.time()
        print(f"Recording started: {self.record_filename}")

    def stop_recording(self) -> None:
        """Stop video recording."""
        if not self.config.recording:
            return
        self.config.recording = False
        elapsed = self.elapsed_time
        self.elapsed_time = "00:00:00"
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print(f"Recording stopped ({elapsed}): {self.record_filename}")

    def take_snapshot(self, image: np.ndarray) -> None:
        """Save a snapshot of the current frame."""
        timestamp = time.strftime("%Y-%m-%d-%H_%M_%S")
        filename = os.path.join(self.output_dir, f"thermal-camera-{timestamp}.png")
        cv2.imwrite(filename, image)
        self.snaptime = time.strftime("%H:%M:%S")
        print(f"Snapshot saved: {filename}")

    def set_fullscreen(self, enabled: bool) -> None:
        """Toggle fullscreen mode."""
        self.config.fullscreen = enabled
        if enabled:
            cv2.namedWindow('Thermal', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Thermal', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow('Thermal', cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty('Thermal', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('Thermal', self.config.display_width, self.config.display_height)

    def update_scale(self, delta: int) -> None:
        """Update the display scale."""
        self.config.scale = max(1, min(5, self.config.scale + delta))
        if not self.config.fullscreen and not self.is_pi:
            cv2.resizeWindow('Thermal', self.config.display_width, self.config.display_height)
    
    def toggle_range_lock(self, thermal: ThermalData) -> None:
        """Toggle colormap range locking."""
        if self.config.range_locked:
            self.config.range_locked = False
            self.config.locked_min_temp = None
            self.config.locked_max_temp = None
            print("Colormap range: auto")
        else:
            self.config.range_locked = True
            self.config.locked_min_temp = thermal.min_temp
            self.config.locked_max_temp = thermal.max_temp
            print(f"Colormap range locked: {thermal.min_temp}C - {thermal.max_temp}C")

    def adjust_locked_range(self, which: str, delta: float) -> None:
        """Adjust locked range min or max."""
        if not self.config.range_locked:
            return
        if which == 'min' and self.config.locked_min_temp is not None:
            new_val = round(self.config.locked_min_temp + delta, 1)
            if self.config.locked_max_temp is not None and new_val < self.config.locked_max_temp:
                self.config.locked_min_temp = new_val
        elif which == 'max' and self.config.locked_max_temp is not None:
            new_val = round(self.config.locked_max_temp + delta, 1)
            if self.config.locked_min_temp is not None and new_val > self.config.locked_min_temp:
                self.config.locked_max_temp = new_val

    def handle_keypress(self, key: int, heatmap: np.ndarray,
                        thermal: ThermalData) -> bool:
        """
        Handle keyboard input.

        Returns:
            False if should quit, True otherwise.
        """
        if key == -1:
            return True

        key_handlers: Dict[int, Callable[[], Optional[bool]]] = {
            ord('+'): lambda: self.update_scale(1),
            ord('='): lambda: self.update_scale(1),  # + without shift
            ord('-'): lambda: self.update_scale(-1),
            ord('c'): lambda: setattr(self.config, 'alpha', min(3.0, round(self.config.alpha + 0.1, 1))),
            ord('d'): lambda: setattr(self.config, 'alpha', max(0.0, round(self.config.alpha - 0.1, 1))),
            ord('h'): lambda: setattr(self.config, 'hud_mode', (self.config.hud_mode + 1) % 4),
            ord('m'): lambda: setattr(self.config, 'colormap_index', (self.config.colormap_index + 1) % len(COLORMAPS)),
            ord(','): lambda: setattr(self.config, 'colormap_index', (self.config.colormap_index - 1) % len(COLORMAPS)),
            ord('n'): lambda: setattr(self.config, 'use_median_filter', not self.config.use_median_filter),
            ord('l'): lambda: self.toggle_range_lock(thermal),
            ord('['): lambda: self.adjust_locked_range('min', -1.0),
            ord(']'): lambda: self.adjust_locked_range('min', 1.0),
            ord(';'): lambda: self.adjust_locked_range('max', -1.0),
            ord("'"): lambda: self.adjust_locked_range('max', 1.0),
            ord('f'): lambda: self.set_fullscreen(not self.config.fullscreen),
            ord('r'): lambda: self.start_recording() if not self.config.recording else None,
            ord('t'): lambda: self.stop_recording(),
            ord('p'): lambda: self.take_snapshot(heatmap),
            ord(' '): lambda: setattr(self.config, 'paused', not self.config.paused),
            ord('q'): lambda: False,
            27: lambda: False,  # ESC key
        }

        if key in key_handlers:
            result = key_handlers[key]()
            if result is False:
                return False

        return True
    
    def _normalize_for_colormap(self, bgr: np.ndarray, thermal: ThermalData) -> np.ndarray:
        """
        Normalize image for colormap application.

        If range is locked, normalizes to the locked temperature range.
        Otherwise, auto-scales to current frame's min/max.
        """
        if not self.config.range_locked:
            return bgr

        # Convert to grayscale for normalization
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Map current frame's temperature range to locked range
        # Assuming linear relationship between pixel intensity and temperature
        current_min = thermal.min_temp
        current_max = thermal.max_temp
        locked_min = self.config.locked_min_temp or current_min
        locked_max = self.config.locked_max_temp or current_max

        if current_max == current_min:
            return bgr

        # Normalize: map current temps to 0-255 based on locked range
        gray_float = gray.astype(np.float32)

        # Scale factor: what fraction of locked range does current pixel represent
        temp_range = locked_max - locked_min
        if temp_range <= 0:
            return bgr

        # Map pixel values: assume they linearly correspond to temperature
        # pixel 0 = current_min, pixel 255 = current_max
        # We want to remap so locked_min->0, locked_max->255
        scale = 255.0 / temp_range
        offset = -locked_min * scale

        # Convert pixel intensities to "temperature", then to new pixel values
        pixel_to_temp_scale = (current_max - current_min) / 255.0
        pixel_to_temp_offset = current_min

        # temp = pixel * pixel_to_temp_scale + pixel_to_temp_offset
        # new_pixel = temp * scale + offset
        # new_pixel = pixel * pixel_to_temp_scale * scale + pixel_to_temp_offset * scale + offset

        combined_scale = pixel_to_temp_scale * scale
        combined_offset = pixel_to_temp_offset * scale + offset

        normalized = gray_float * combined_scale + combined_offset
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[ThermalData]]:
        """Process a single frame and return the rendered heatmap and thermal data."""
        # Track frame time for FPS
        frame_start = time.perf_counter()

        # Extract thermal data
        imdata, thdata = extract_thermal_data(frame, self.config.sensor_height, self.config.camera_type)
        thermal = process_thermal_data(thdata, self.config.camera_type, self.config.sensor_width, self.config.use_median_filter)

        # Convert to BGR and apply processing
        bgr = convert_to_bgr(imdata, self.config.camera_type)
        bgr = cv2.convertScaleAbs(bgr, alpha=self.config.alpha)
        bgr = cv2.resize(bgr, (self.config.display_width, self.config.display_height),
                         interpolation=cv2.INTER_CUBIC)

        # Normalize for locked colormap range
        bgr = self._normalize_for_colormap(bgr, thermal)

        # Apply colormap
        heatmap, colormap_name = apply_colormap(bgr, self.config.colormap_index)

        # Draw crosshairs at center
        center_x = self.config.display_width // 2
        center_y = self.config.display_height // 2
        draw_crosshairs(heatmap, center_x, center_y, thermal.center_temp)

        # Draw HUD
        if self.config.hud_mode == 0:
            draw_hud(heatmap, self.config, thermal, colormap_name,
                    self.snaptime, self.elapsed_time, False, self.current_fps)
        if self.config.hud_mode == 1:
            draw_hud(heatmap, self.config, thermal, colormap_name,
                    self.snaptime, self.elapsed_time, True, self.current_fps)
        elif self.config.hud_mode == 2:
            draw_help_page(heatmap)

        # Draw floating temperature markers with position hysteresis
        scale = self.config.scale

        # Pause indicator
        if self.config.paused:
            label = "PAUSED"
            font = _HUD_FONT
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            px = (self.config.display_width - tw) // 2
            py = th + _HUD_PAD
            # background pill
            overlay = heatmap.copy()
            cv2.rectangle(overlay, (px - _HUD_PAD, py - th - 4),
                          (px + tw + _HUD_PAD, py + 4), (0, 0, 0), -1)
            cv2.addWeighted(overlay, _HUD_BG_ALPHA, heatmap,
                            1 - _HUD_BG_ALPHA, 0, heatmap)
            cv2.putText(heatmap, label, (px, py), font,
                        font_scale, _HUD_COLOR, 2, cv2.LINE_AA)

        # Max marker: only update position if temperature changed
        if self.prev_max_temp is None or thermal.max_temp != self.prev_max_temp:
            self.prev_max_pos = thermal.max_pos
            self.prev_max_temp = thermal.max_temp
        max_x = int(self.prev_max_pos[1] * scale)
        max_y = int(self.prev_max_pos[0] * scale)
        draw_temperature_marker(heatmap, max_x, max_y, thermal.max_temp, (0, 0, 255))

        # Min marker: only update position if temperature changed
        if self.prev_min_temp is None or thermal.min_temp != self.prev_min_temp:
            self.prev_min_pos = thermal.min_pos
            self.prev_min_temp = thermal.min_temp
        min_x = int(self.prev_min_pos[1] * scale)
        min_y = int(self.prev_min_pos[0] * scale)
        draw_temperature_marker(heatmap, min_x, min_y, thermal.min_temp, (255, 0, 0))

        # Update FPS
        self.frame_times.append(time.perf_counter() - frame_start)
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

        return heatmap, thermal
    
    def run(self) -> None:
        """Main application loop."""
        print(__doc__)
        last_frame: Optional[np.ndarray] = None
        last_thermal: Optional[ThermalData] = None
        consecutive_failures = 0
        max_failures = 30  # ~1 second at 30fps

        while self.cap.isOpened():
            if not self.config.paused:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        print("Error: Too many consecutive frame read failures. Exiting.")
                        break
                    continue
                consecutive_failures = 0
                last_frame = frame
            else:
                frame = last_frame
                if frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    last_frame = frame

            heatmap, thermal = self.process_frame(frame)
            if heatmap is None or thermal is None:
                continue
            last_thermal = thermal

            # Display
            cv2.imshow('Thermal', heatmap)

            # Handle recording
            if self.config.recording:
                elapsed = time.time() - self.record_start_time
                self.elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                if self.video_writer:
                    self.video_writer.write(heatmap)

            # Handle input
            key = cv2.waitKey(1)
            if not self.handle_keypress(key, heatmap, thermal):
                break

        self.cleanup()
    
    def cleanup(self) -> None:
        """Release resources."""
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()


# =============================================================================
# Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Thermal Camera Viewer for Infiray P2 / Topdon TC001",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=2,
        help="Video device number (use v4l2-ctl --list-devices to find)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        choices=['infiray', 'topdon'],
        default=None,
        help="Camera type: 'infiray' for Infiray P2, 'topdon' for Topdon TC001 (default: auto-detect)"
    )
    parser.add_argument(
        "--median-filter", "-m",
        action='store_true',
        help="Apply 5x5 spatial median filter to reduce temperature noise"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default='recordings',
        help="Directory for recordings and snapshots (default: recordings)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        viewer = ThermalCameraViewer(
            device=args.device,
            camera_type=args.camera,
            use_median_filter=args.median_filter,
            output_dir=args.output_dir
        )
        viewer.run()
    except CameraInitError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
