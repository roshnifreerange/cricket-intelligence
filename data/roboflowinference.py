import json
from roboflow import Roboflow
import supervision as sv
import cv2

rf = Roboflow(api_key="gSRCcCrETNkUwsYFkQlt")
project = rf.workspace().project("cricket-oftm6")
model = project.version(3).model

result = model.predict("test_img/img1.jpg", confidence=5, overlap=60).json()

print("=== Raw predictions ===")
print(json.dumps(result, indent=2))
print(f"\nTotal predictions: {len(result.get('predictions', []))}")

labels = [item["class"] for item in result["predictions"]]
print(f"Classes found: {labels}")

detections = sv.Detections.from_inference(result)

label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

image = cv2.imread("test_img/img1.jpg")

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(image=annotated_image, size=(16, 16))
