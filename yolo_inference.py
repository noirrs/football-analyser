from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('data/raw/test (3).mp4', save=True)
print(results[0])
print('==========================================')
for box in results[0].boxes:
    print(f"Box: {box.xyxy}, Confidence: {box.confidence}, Class: {box.cls}")

