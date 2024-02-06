from roboflow import Roboflow

rf = Roboflow(api_key="xRQkaPHacbILowtZDeWp")
project = rf.workspace().project("tumor-cancer-aneurysm-detection")
model = project.version(2).model

image_path = ("/Users/charlestimmes/PycharmProjects/pythonProject/DiscoverAI/Images/archive-5/files/aneurysm/1.jpg")
prediction_result = model.predict(image_path)
prediction_json = prediction_result.json()

confidence_number = 0.60


max_confidence_prediction = max(prediction_json['predictions'][0]['predictions'], key=lambda x: x['confidence'])
max_class_label = max_confidence_prediction['class']
max_confidence = max_confidence_prediction['confidence']


print(f"Based off the image there is an {max_class_label} we report this with a confidence of {max_confidence*100}%.")

