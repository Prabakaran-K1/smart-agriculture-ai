# PlantVillage leaf classifier

This folder contains the ONNX export and 15 output labels from [imaflower/plantvillage-mobilenetv3](https://huggingface.co/imaflower/plantvillage-mobilenetv3), distributed under MIT. The model was trained on curated PlantVillage images and is intended as assistive screening, not a field diagnosis. Field photos, unsupported crops, mixed symptoms, and local cultivars can produce incorrect results.

The app runs inference in the browser through ONNX Runtime Web. Images remain in the browser and are not sent to Flask for classification. Only tomato early/late blight labels currently map to source-backed treatment entries; other labels never auto-select a pesticide.
