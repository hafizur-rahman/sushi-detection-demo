# Training Tensorflow Object Detection Model

## Prepare Env

* Create a dir with foloowing structure:
    ```
    detection_demo
        ├─annotations
        ├─config
        ├─data
        │  └─train
        ├─inference
        ├─models
        ├─test_images
        └─training
    ```
* Install Anaconda
* Install tensorflow
* Install tensorflow models
    - Clone `tensorflow/models` git repo

## Steps
* What kind of objects to be detected
* Collect respresentative images for objects to be detected
* Annotate Images
* Create TFRecord
* Train the model
* Validate

### What kind of objects to be detected

### Collect images
* [Using Google Image Downloader](https://google-images-download.readthedocs.io/en/latest/index.html)
* Other public datasets (https://skymind.ai/wiki/open-datasets)

**Note**: put images to `data/train` dir

### Annotate images
* Install LabelImg
    - For windows, download binary executable from https://github.com/tzutalin/labelImg/releases
    - Otherwise, build from source (https://github.com/tzutalin/labelImg)
* Annotate images and save to the same dir

### Create TFRecord
```
python preprocess.py --inputPath=data --outputPath=data

python generate_tfrecord.py --csv_input=data\train_labels.csv --output_path=annotations\train.record --img_pathh=data\train
```

### Train the model
* Download pretrained models from [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Tried with [faster_rcnn_resnet50](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) 
* Unzip the model dist
* Prepare training config
* Train the model
    ```
    python c:\dev\tensorflow\models\research\object_detection\legacy\train.py --train_dir=training --pipeline_config_path=config\faster_rcnn_resnet50.config
    ```

### Export Inference Graph
```
python c:\dev\tensorflow\models\research\object_detection\export_inference_graph.py --input_type image_tensor --pipeline_config_path config\faster_rcnn_resnet50.config --trained_checkpoint_prefix training\model.ckpt-100 --output_directory inference
```

### Validate


# Links
* https://github.com/tensorflow/models/tree/master/research/object_detection
* https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
* https://heartbeat.fritz.ai/building-a-real-time-object-recognition-ios-app-that-detects-sushi-c4a3a2c32298?gi=69a50bc084ca
