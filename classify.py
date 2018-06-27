from bottle import request, response, route, run
import tensorflow as tf
import sys, os, errno, urllib, uuid
import numpy as np
import urllib.request
#from flask import Flask

WORKING_DIRECTORY = "tf_files"
TMP_DIRECTORY = "tmp"
TRAINED_LABELS = "%s/retrained_labels.txt" % (WORKING_DIRECTORY)
RETRAINED_GRAPH = "%s/retrained_graph.pb" % (WORKING_DIRECTORY)

data = []
input_height = 299
input_width = 299
input_mean = 0
input_std = 255

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
    json = {}
    print(request.json['data'])
    for info in request.json['data']:
        if (info['type'] == 'local'):
            json[info['path']] = score(info['path'])
        else:
            path = download_image(info['path'], info['ext'])
            json[info['path']] = score(path)
            os.remove(path)
        print(json)
    return str(json)

@route('/status/', method='GET')
def status():
    return 'online'

@route('/', method='GET')
def getHome():
    return status()

def download_image(url, extension):
    filename = TMP_DIRECTORY + '/' + uuid.uuid4().hex + extension
    urllib.request.urlretrieve(url, filename)
    return filename

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
    sess = tf.Session()
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
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(TRAINED_LABELS)

    for i in top_k:
        #print(labels[i], results[i])
        human_string = labels[i]
        score = results[i]
        data.append('%s:%.5f' % (human_string, score))
    return data

graph = load_graph(RETRAINED_GRAPH)
input_name = "import/" + "Placeholder"
output_name = "import/" + "final_result"
input_operation = graph.get_operation_by_name(input_name)
output_operation = graph.get_operation_by_name(output_name)

create_tmp('tmp')
#run(host='127.0.0.1', port=8989, debug=True)
run(host='0.0.0.0', port=os.environ.get('PORT', 8080))
