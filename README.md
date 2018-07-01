# nazar-server
Tensorflow image classifier with a rest api server to fetch the response from the picture taken using Nazar Application

## Description

You can use the repository to train yaour dataset using the script and then use the classify.py file to classify the images.

## Installation 

- You need to clone this repo `git clone https://github.com/aryaminus/nazar-server/`
- Then `cd nazar-server`
- Next you need to install all dependencies of this project hopefully just run this command `pip install -r requirements.txt`

You are okay to execute python script now !

## Training (optional)

By default, where you have cloned the repository you have a tf_files/data. This is where the magic begins, you need to gather enough image to create a dataset of several collections. 
Yes you can not do just one dataset.

You should have something like that :

```
├── classify.py
├── requierments.txt
├── retrain.py
├── tf_files
│   ├── data
│   │   ├── register
│   │   │   ├── 00.jpeg
│   │   │   ├── 01.jpeg
│   │   │   ├── 02.jpeg
│   │   │   ├── 03.jpeg
│   │   │   ...
│   │   └── capacitor
│   │       ├── 00.jpeg
│   │       ├── 01.jpeg
│   │       ├── 02.jpeg
│   │       ...
└── train_data.sh
```

If you have already some dataset to another folder you can edit variable in the train_data.sh :

```sh
WORKING_DIR="tf_files"

BOTTLENECK_DIR="$WORKING_DIR/bottlenecks"
STEPS=5000
MODEL_DIR="$WORKING_DIR/inception"
OUTPUT_GRAPH="$WORKING_DIR/retrained_graph.pb"
OUTPUT_LABELS="$WORKING_DIR/retrained_labels.txt"
DATA_FOLDER="$WORKING_DIR/data"
SUM_FOLDER="$WORKING_DIR/retrain_logs"
...
```

And in the classify.py

```python
...
WORKING_DIRECTORY="tf_files"
TMP_DIRECTORY = "tmp"
TRAINED_LABELS="%s/retrained_labels.txt" % (WORKING_DIRECTORY)
RETRAINED_GRAPH="%s/retrained_graph.pb" % (WORKING_DIRECTORY)
...
```

Then all is done just run `sh train_data.sh`

You will see something like that : 
![https://i.imgur.com/HxzXqz7.png](https://i.imgur.com/HxzXqz7.png)
And the end output is something like this :
```
INFO:tensorflow:Final test accuracy = 92.31% (N=1000)
INFO:tensorflow:Froze 2 variables.
Converted 2 variables to const ops.
```

## Classify an image

To classify an image you need to run `python classify.py` in background (with systemctl for instance or screen).
Then to check if you have access to the api just do `curl http://0.0.0.:8080/status/`.

To check from local server type in terminal :

```curl
curl -POST -H "Content-type: application/json" -d 
'{
  "data": [{
    "image64" : "base_64_encoded_textbase_64_encoded_text"
  }]
}'
'0.0.0.0:8080/classify_image/'
```

To check from the heroku server type in terminal :

```curl
curl -POST -H "Content-type: application/json" -d 
'{
  "data": [{
    "image64" : "data:image/png;base64,base_64_encoded_text"
  }]
}'
'nazar-server.herokuapp.com/classify_image/'
```

**where the given image64 has extra layer of  `base_64_encoded_text`**
