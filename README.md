# Football Analyzer

A comprehensive computer vision system for analyzing football matches using YOLO object detection and tracking algorithms. This project can track players, ball, and referees in football videos, estimate player speeds and distances, assign teams, and determine ball possession.

## Features

- **Object Detection & Tracking**: Uses YOLOv8 to detect and track players, ball, and referees
- **Team Assignment**: Automatically assigns players to teams based on jersey colors
- **Ball Possession**: Determines which player has control of the ball
- **Speed & Distance Calculation**: Estimates player speeds (km/h) and total distances covered
- **Camera Movement Compensation**: Adjusts for camera movement to improve tracking accuracy
- **View Transformation**: Transforms player positions to a top-down view of the field

## Project Structure

```
football-analyser/
├── main.py                          # Main execution script
├── requirements.txt                 # Python dependencies
├── yolo_inference.py               # YOLO model inference utilities
├── camera_movement_estimator/      # Camera movement detection and compensation
├── player_ball_assigner/           # Ball possession assignment logic
├── speed_and_distance_estimator/   # Speed and distance calculation
├── team_assigner/                  # Team color detection and assignment
├── trackers/                       # Object tracking implementation
├── utils/                          # Utility functions for video and bbox operations
├── view_transformer/               # Field view transformation
├── train/                          # Training notebooks and datasets
├── models/                         # Trained model files
├── data/                           # Video data and processed results
└── runs/                           # Training and inference outputs
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/noirrs/football-analyser.git
cd football-analyser
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the required model files:
   - Place your trained YOLO model as `models/best.pt`
   - Download YOLOv8x weights as `yolov8x.pt`

## Usage

### Basic Analysis

Run the main analysis on a football video:

```bash
python main.py
```

The script will:

1. Load a video from `data/raw/`
2. Detect and track all objects (players, ball, referees)
3. Assign teams based on jersey colors
4. Calculate player speeds and distances
5. Determine ball possession
6. Output annotated video with all metrics

### Training Custom Models

Use the Jupyter notebook in the `train/` directory to train your own YOLO model on football player detection:

```bash
jupyter notebook train/football_training_yolo.ipynb
```

### Color Assignment Analysis

Explore team color assignment algorithms:

```bash
jupyter notebook dev/color_assignment.ipynb
```

## Key Components

### Tracker

- Implements YOLO-based object detection
- Tracks multiple objects across video frames
- Handles object interpolation for missing detections

### Team Assigner

- Analyzes jersey colors to determine team membership
- Uses K-means clustering for color classification
- Assigns consistent team colors throughout the match

### Speed & Distance Estimator

- Calculates real-world speeds and distances
- Compensates for camera movement
- Provides frame-by-frame speed analysis

### View Transformer

- Transforms player positions to field coordinates
- Enables accurate distance measurements
- Supports perspective correction

## Configuration

Key parameters can be adjusted in the respective modules:

- **Frame window**: Processing batch size (default: 5 frames)
- **Frame rate**: Video FPS (default: 24 fps)
- **Model paths**: Update paths to your trained models
- **Stub files**: Enable/disable caching for faster processing

## Data Flow

1. **Video Input** → Object Detection (YOLO)
2. **Raw Detections** → Object Tracking
3. **Tracked Objects** → Camera Movement Compensation
4. **Compensated Tracks** → View Transformation
5. **Transformed Positions** → Speed/Distance Calculation
6. **Enhanced Tracks** → Team Assignment
7. **Team Data** → Ball Possession Analysis
8. **Final Analysis** → Annotated Video Output

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy
- Matplotlib
- scikit-learn
- Jupyter (for training notebooks)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was built following the comprehensive tutorial: [AI Football Analysis System Tutorial](https://www.youtube.com/watch?v=neBZ6huolkg&t=11732s)
- YOLOv8 by Ultralytics for object detection
- OpenCV for computer vision utilities
- Football detection dataset from Roboflow

## Future Enhancements

- [ ] Real-time processing capabilities
- [ ] Advanced tactical analysis
- [ ] Player identification and statistics
- [ ] Heat map generation
- [ ] Multi-camera support
- [ ] Web interface for analysis
