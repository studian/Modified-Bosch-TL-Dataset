## 0. pip install packages 
* bleach (1.5.0)
* cffi (1.9.1)
* conda (4.3.11)
* cryptography (1.7.1)
* cycler (0.10.0)
* decorator (4.2.1)
* entrypoints (0.2.3)
* enum34 (1.1.6)
* html5lib (0.9999999)
* idna (2.2)
* ipykernel (4.8.0)
* ipython (6.2.1)
* ipython-genutils (0.2.0)
* ipywidgets (7.1.1)
* jedi (0.11.1)
* Jinja2 (2.10)
* jsonschema (2.6.0)
* jupyter (1.0.0)
* jupyter-client (5.2.2)
* jupyter-console (5.2.0)
* jupyter-core (4.4.0)
* Markdown (2.6.11)
* MarkupSafe (1.0)
* matplotlib (2.1.2)
* mistune (0.8.3)
* nbconvert (5.3.1)
* nbformat (4.4.0)
* notebook (5.3.1)
* numpy (1.13.3)
* opencv-python (3.4.0.12)
* pandocfilters (1.4.2)
* parso (0.1.1)
* pexpect (4.3.1)
* pickleshare (0.7.4)
* Pillow (5.0.0)
* pip (9.0.1)
* prompt-toolkit (1.0.15)
* protobuf (3.5.1)
* ptyprocess (0.5.2)
* pyasn1 (0.1.9)
* pycosat (0.6.1)
* pycparser (2.17)
* Pygments (2.2.0)
* pyOpenSSL (16.2.0)
* pyparsing (2.2.0)
* python-dateutil (2.6.1)
* pytz (2017.3)
* PyYAML (3.12)
* pyzmq (16.0.4)
* qtconsole (4.3.1)
* requests (2.12.4)
* Send2Trash (1.4.2)
* setuptools (27.2.0)
* simplegeneric (0.8.1)
* six (1.10.0)
* tensorflow-gpu (1.4.1)
* tensorflow-tensorboard (0.4.0)
* terminado (0.8.1)
* testpath (0.3.1)
* tornado (4.5.3)
* tqdm (4.19.5)
* traitlets (4.3.2)
* wcwidth (0.1.7)
* Werkzeug (0.14.1)
* wheel (0.29.0)
* widgetsnbextension (3.1.3)

## 1. Setting Dev-Environment(tensorflow, model, etc.) and Dataset

```bash
#!/bin/bash
# connect to the instance with security key in ~/.ssh/
#
# clone TensorFlow repo to the instance 
echo "cloning tensorflow lib"
git clone https://github.com/tensorflow/models.git
sudo mv models/research/object_detection ./
sudo mv models/research/slim ./
mkdir model && cd model
mkdir train
mkdir eval
cd ..
echo -e "\n downloading bosch dataset"
cd data
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.001
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.002
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.003
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.004
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.005
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.006
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_test_rgb.zip.007
cat dataset_test_rgb.zip* > dataset_test_rgb.zip
find . -type f -name 'dataset_test_rgb.zip.*' -delete
unzip dataset_test_rgb.zip
rm dataset_test_rgb.zip
rm non-commercial_license.docx
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.001
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.002
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.003
wget https://s3-us-west-1.amazonaws.com/bosch-tl/dataset_train_rgb.zip.004
cat dataset_train_rgb.zip* > dataset_train_rgb.zip
find . -type f -name 'dataset_train_rgb.zip.*' -delete
unzip dataset_train_rgb.zip
rm dataset_train_rgb.zip
rm non-commercial_license.docx
# download the pretrained model
#cd ~/models/model
echo -e "\n downloading pretrained models"
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
tar -xf ssd_mobilenet_v1_coco_11_06_2017.tar.gz 
rm ssd_mobilenet_v1_coco_11_06_2017.tar.gz 
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -xf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
rm faster_rcnn_resnet101_coco_11_06_2017.tar.gz
cd ..
echo -e "\n installing dependencies"
pip install tensorflow-gpu
pip install tqdm
echo -e "\n generating train tfrecords"
chmod +x "bosch-to-tfrecords.py"
python bosch-to-tfrecords.py --output_path=data/train.record
echo -e "\n generating test tfrecords"
chmod +x "bosch-to-tfrecords-test.py" 
python bosch-to-tfrecords-test.py --output_path=data/test.record

# set up protobuf
echo -e "\n set up protobuf"
# https://gist.github.com/sofyanhadia/37787e5ed098c97919b8c593f0ec44d8
curl -OL https://github.com/google/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip
unzip protoc-3.4.0-linux-x86_64.zip
sudo mv bin/* /usr/local/bin/
rm -r bin
sudo mv include/* /usr/local/include/
rm -r include

# Protobuf compilation (object detection installation dependencies)
protoc object_detection/protos/*.proto --python_out=.
# add libraries to python path (object detection installation dependencies)
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
#Testing the Installation
python object_detection/builders/model_builder_test.py
```

## 2. Training
```bash
python object_detection/train.py --pipeline_config_path=config/faster_rcnn_traffic_bosch.config --train_dir=training_data/frcnn
```

## 3. Saving for Inference
### 1) modified line 71 of "object_detection/exporter.py"
```
# rewrite_options = rewriter_config_pb2.RewriterConfig(layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
rewrite_options = rewriter_config_pb2.RewriterConfig()
```
### 2) run script
```bash
python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn_traffic_bosch.config --trained_checkpoint_prefix=training_data/frcnn/model.ckpt-12979 --output_directory=frozen_frcnn
```

## 4. Prediction of Test Dataset
### 1) jupyter notebook
run `jupyter notebook`
run `object_detection_bosch.ipynb`
### 2) Run python code
```bash
python Predict_usingTestDB_FasterRCNN_BoschTL.py
```



