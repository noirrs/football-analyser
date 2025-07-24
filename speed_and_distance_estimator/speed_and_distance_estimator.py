import cv2
import sys
sys.path.append('../')
from utils import measure_distance, get_foot_position
class SpeedAndDistanceEstimator:
    def __init__(self):
        self.frame_window = 5 # batch size for processing frames (every 5 frames)
        self.frame_rate = 24 # frames per second

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == 'ball' or object == 'referees':
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames-1) # if frame_num + self.frame_window exceeds number of frames, use the last frame

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]: 
                        continue

                    # Check if transformed_position exists in both frames
                    if 'transformed_position' not in object_tracks[frame_num][track_id]:
                        continue
                    if 'transformed_position' not in object_tracks[last_frame][track_id]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['transformed_position']
                    end_position = object_tracks[last_frame][track_id]['transformed_position']

                    if start_position is None or end_position is None: 
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate # time in seconds
                    speed = distance_covered / time_elapsed if time_elapsed > 0 else 0
                    speed_in_kmh = speed * 3.6 # convert m/s to km/h

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_in_kmh
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, video_frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referees':
                    continue
                for _, track_data in object_tracks[frame_num].items():
                   if 'speed' in track_data and 'distance' in track_data:
                        speed = track_data.get('speed', None)
                        distance = track_data.get('distance', None)
                        if speed is None or distance is None:
                            continue
                        bbox = track_data['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1]+= 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f'Speed: {speed:.2f} km/h', position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(frame, f'Distance: {distance:.2f} m', (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            output_frames.append(frame)
        return output_frames

                   