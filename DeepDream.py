import numpy as np
from functools import  partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile
import png

IMAGE_NAME = 'coverphoto.jpg'
GREY_IMAGE_SIZE = (224,224,3)
IMAGENET_MEAN = 117.0

#Pick a Layer to enhance our image and some feature channel to visualize
ENHANCE_LAYER = 'mixed4d_3x3_bottleneck_pre_relu' 
CHANNEL = 139
#Catch the output from some layer after our input
OUTPUT_LAYER  = 'mixed4c'

class DeepDream(object):
    def __init__(self):
        #Specify Model location. In this case we use googles inception architecture
        self.url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
        self.data_dir = '../data/'
        self.model_name = os.path.split(self.url)[-1]
        self.local_zip_file = os.path.join(self.data_dir, self.model_name)
        self.model_fn = 'tensorflow_inception_graph.pb'

        #Specify Image location and save out image as np array
        self.image_dir = './images'
        self.image_location = os.path.join(self.image_dir, IMAGE_NAME)
        self.img_noise = np.random.uniform(size=GREY_IMAGE_SIZE)

        self.graph = tf.Graph()

    def get_image_as_npArray(self):
        img = PIL.Image.open(self.image_location)
        img = np.float32(img)
        return img

    def maybe_download_inception_model(self):
        if not os.path.exists(self.local_zip_file):
            # Download
            model_url = urllib.request.urlopen(self.url)
            with open(self.local_zip_file, 'wb') as output:
                output.write(model_url.read())
            # Extract
            with zipfile.ZipFile(self.local_zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

    def create_tensorflow_graph(self):
        session = tf.InteractiveSession(graph=self.graph)

        with tf.gfile.FastGFile(os.path.join(self.data_dir, self.model_fn), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
        #define the input tensor
        self.t_input = tf.placeholder(np.float32, name='input')
        #Transform mean of input to 0 and add a dimension  
        imagenet_mean = IMAGENET_MEAN
        t_preprocessed = self.t_input - imagenet_mean
        t_preprocessed = tf.expand_dims(t_preprocessed, 0)

        tf.import_graph_def(graph_def, {'input':t_preprocessed})
        
        #Display the number of layers and feature channels
        layers = [op.name for op in self.graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
        feature_nums = [int(self.graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]     
        print('Number of layers', len(layers))
        print('Total number of feature channels:', sum(feature_nums))


    def get_layer_output_as_tensor(self):
        #Catch the output at the specified layer of the inception graph
        outputTensor = self.graph.get_tensor_by_name("import/%s:0"%OUTPUT_LAYER)

        #WHYY? Siraj squares the tensor. Not sure why..
        outputTensor = tf.square(outputTensor)
        return outputTensor

    def generate_octaves_from_image(self, octave_n=1, octave_scale=1.4):
        # split the image into a number of octaves
        img0 = self.get_image_as_npArray()
        octaves = []
        for _ in range(octave_n-1):
            hw = img0.shape[:2]
            lo = resize(img0, np.int32(np.float32(hw)/octave_scale))
            hi = img0-resize(lo, hw)
            img0 = lo
            octaves.append(hi)
 
    def render_deepdream(self, octaves, iter_n=1, step=1.5):
        outputTensor = self.get_layer_output_as_tensor()
        # defining the optimization objective
        print(outputTensor.graph)
        t_score = tf.reduce_mean(outputTensor)
        # behold the power of automatic differentiation!
        t_grad = tf.gradients(t_score, self.t_input)[0]

        img0 = self.get_image_as_npArray()
        octave_n = len(octaves)
        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img0 = resize(img0, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img0, t_grad)
                img0 += g*(step / (np.abs(g).mean()+1e-7))
        return img0

if __name__ == "__main__":
    dream = DeepDream()
    dream.create_tensorflow_graph()
    octaves = dream.generate_octaves_from_image()
    result = dream.render_deepdream(octaves)

  
    

