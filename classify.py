import base64
import errno
#from json import dumps
import json
import math
import os
import sys
import urllib
import urllib.parse
import urllib.request
import urllib.response
import uuid

import numpy as np
import tensorflow as tf
from bottle import request, response, route, run

#from flask import Flask

WORKING_DIRECTORY = "tf_files"
TMP_DIRECTORY = "tmp"
TRAINED_LABELS = "%s/retrained_labels.txt" % (WORKING_DIRECTORY)
RETRAINED_GRAPH = "%s/retrained_graph.pb" % (WORKING_DIRECTORY)

#data = []
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
filer = uuid.uuid4().hex
#app = Flask(__name__)

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

@route('/classify_image/', method='POST')
def index():
    response.content_type='application/json'
    #json = []
    print(request.json['data'])
    for info in request.json['data']:
        '''
        if (info['type'] == 'local'):
            path = save_image(info['path'], info['ext'])
            #json[info['path']] = score(info['path'])
        else:
            path = download_image(info['path'], info['ext'])
        '''
        path = save_image(info['image64'])
        jsonF = score(path)

        # Octopart API Searches the Octopart database 
        url = "http://octopart.com/api/v3/parts/search"
        url += "?apikey="
        url += "6aa164d3"
        args = [
            ('q', jsonF[0]),
            ('include', 'short_description'),
            ('start', 0),
            ('limit', 1)
        ]
        url += '&' + urllib.parse.urlencode(args)

        data = urllib.request.urlopen(url).read().decode()
        searchResponse = json.loads(data)
        for item in searchResponse['results']:
            jsonStream = item['item']
        
        uid = jsonStream['uid']
        brandName = jsonStream['brand']['name']
        manufacturer = jsonStream['manufacturer']['name']
        octopartUrl = jsonStream['octopart_url']
        shortDescription = jsonStream['short_description']
        
        # Performs a part-match with Octopart and includes the tech specs
        url2 = 'http://octopart.com/api/v3/parts/'
        url2 += uid
        url2 += '?apikey='
        url2 += "6aa164d3"
        url2 += '&include[]=specs'
        url2 += '&include[]=imagesets'
        url2 += '&include[]=descriptions'
        data2 = json.loads(urllib.request.urlopen(url2).read().decode())

        specJson = data2['specs']
        specArray = []
        for specName in specJson:
            tmpSpec = specJson[specName]
            name = tmpSpec['metadata']['name']
            value = tmpSpec['display_value']
            specArray.append([name, value])

        descArray = []
        if data2['descriptions']:
            for i in range(0, len(data2['descriptions'])):
                descArray.append(data2['descriptions'][i]['value'])

        specs = specArray
        descriptions = descArray

        #print(data2['imagesets'])
        ident = { "Component": jsonF[0], "Predictions": jsonF[1], "ShortInfo": shortDescription, "octopartUrl": octopartUrl, "brandName": brandName, "manufacturer": manufacturer, "specs": specs, "descriptions": descriptions }
        os.remove(path)
    return dict(ident)

@route('/status/', method='GET')
def status():
    return 'online'

@route('/', method='GET')
def getHome():
    return status()

def save_image(based):
    '''
    image = open('index.jpg', 'rb')
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
    imgdata = base64.b64decode(image_64_encode)
    '''
    based += "=" * ((4 - len(based) % 4) % 4) # add extra padding
    extension = based.split(";")[0].split("/")[1] 
    imgdata = base64.b64decode(based.split(",")[1])
    filename = "{}/{}.{}".format(TMP_DIRECTORY,uuid.uuid4(),extension)
    with open(filename, 'wb') as f:
        f.write(imgdata)
    return filename

'''
def download_image(url, extension):
    filename = TMP_DIRECTORY + '/' + uuid.uuid4().hex + extension
    urllib.request.urlretrieve(url, filename)
    return filename
'''

def create_tmp(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    '''
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    '''
    image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    with tf.Session() as sess:
        result = sess.run(normalized)
    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def score(image_path):
    t = read_tensor_from_image_file(
        image_path,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)
    top_k = results.argsort()[-1:][::-1]
    labels = load_labels(TRAINED_LABELS)
    data =[]
    for i in top_k:
        #print(labels[i], results[i])
        human_string = labels[i]
        score = results[i]
        #data.append('%s:%.5f' % (human_string, score*100))
        data.append('%s' % (human_string))
        data.append('%.5f' % (score * 100))
    return data

graph = load_graph(RETRAINED_GRAPH)
input_name = "import/" + "Placeholder"
output_name = "import/" + "final_result"
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

create_tmp('tmp')
#run(host='127.0.0.1', port=8989, debug=True)
run(host='0.0.0.0', port=os.environ.get('PORT', 8080))
