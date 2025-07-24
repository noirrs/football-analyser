from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../') 
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):#precision and reccal

        # f curves f1
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_data in track.items():
                    bbox = track_data['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position
                

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing ball positions
        df_ball_positions = df_ball_positions.interpolate() # method='linear', inplace=True
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames): # get detections for each frame in the video and return them as a list
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch     
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None): # stub: a temporary placeholder or substitute

        if read_from_stub and stub_path is not None and os.path.exists(stub_path): # if stub exists, read from it
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames) # get detections for each frame in the video

        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections): # each detection corresponds to a frame in the video with a list of detected objects
            cls_names = detection.names # referee, player, ball, etc. format: {0: 'player', 1: 'referee', 2: 'ball', 3: 'goalkeeper'}
            cls_names_inv = {v:k for k, v in cls_names.items()} # inverse mapping of class names to class ids format: {'player': 0, 'referee': 1, 'ball': 2}

            detection_supervision = sv.Detections.from_ultralytics(detection)  # convert the detection to supervision format to be used by the tracker

            # converting goalkeeper to player object (goalkeeper can't be well-detected by the model due to the lack of training data on roboflow)
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks: # each frame_detection corresponds to a detected object in the frame, with the format [bbox, confidence, class_id, track_id] (WE ARE STILL IN THE UPPER FRAME LOOP)
                bbox = frame_detection[0].tolist() #bounding box in xyxy format
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {'bbox': bbox} # in each frame, each player has a unique track id

                elif cls_id == cls_names_inv['referee']: # same for referee, each referee has a unique track id
                    tracks["referees"][frame_num][track_id] = {'bbox': bbox} 

            for frame_detection in detection_supervision: # for ball detection, we don't need track id, just the bounding box, for reasons such as the ball being a small object and not being tracked well
                bbox = frame_detection[0].tolist()  # [x1, y1, x2, y2] x1, y1 are the coordinates of the top-left corner of the bounding box, and x2, y2 are the coordinates of the bottom-right corner

                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {'bbox': bbox}

            print(detections_with_tracks)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)


        return tracks     

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2= int(bbox[3]) # bottom right y coordinate x1y1x2y2

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, center=(x_center, y2), axes=(int(width), int(width * 0.35)), angle=0.0, startAngle=-45, endAngle=235, color=color,thickness=2,lineType=cv2.LINE_4)
         
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2 # "//"" => floor division 5/2=2, "/" # => normal division 5/2=2.5
        x2_rect = x_center + rectangle_width // 2
        y1_rect =  (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
           
            x1_text = x1_rect + 12
            if track_id > 99: # if track id is greater than 99, we need to shift the text to the left to make it centered
                x1_text -= 10  
            
            cv2.putText(frame, str(track_id), (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y= int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y], [x-10, y-20], [x+10, y-20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame
 
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Semi-transparent overlay for team ball control
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), cv2.FILLED)
        alpha = 0.4 # transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1] # get the team ball control till the current frame; plus one because frame_num is zero-indexed
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f'Team 1 Ball Control: {team_1*100:.2f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2*100:.2f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num] # dictionary
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
              color = player.get('team_color', (0,0,255))
              frame = self.draw_ellipse(frame, player['bbox'],color, track_id)

              if player.get('has_ball', False):
                  frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))

            for _, referee in referee_dict.items():
              frame = self.draw_ellipse(frame, referee['bbox'],(0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            # Draw Team Ball Control Status
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames