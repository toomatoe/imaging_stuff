#imports here

from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from inference import get_model


model = get_model(model_id="yolov8n")
pipeline = InferencePipeline(
    model_id = "vehicle-lqrn8/2",
    max_fps = 0.5,
    confidence = 0.3,
    results = model.infer("https://www.youtube.com/live/9SLt3AT0rXk?si=KaF2ZTo6pF_CBh7I"),
    on_prediction = render_boxes,
    api_key = "my_api_key"
    )
pipeline.start()
pipeline.join()