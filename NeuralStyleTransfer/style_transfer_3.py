from __future__ import print_function, division
from builtins import range, input


#in this script we are focussing on generating an image
#that attepts to match the content of one input image
#and the style of another

#we will accomplish this by managing the content loss
#and the style loss simultaneously.

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from skimage.transform import resize

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from style_transfer_1 import VGG16_AvgPool, VGG16_AvgPool_Cutoff, unpreprocess, scale_img
from style_transfer_2 import gram_matrix, style_loss, minimise
from scipy.optimize import fmin_l_bfgs_b

#load the content image
def load_img_and_preprocess(path, shape= None):
    img = image.load_img(path, target_size=shape)
    #convert img to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis= 0)
    x = preprocess_input(x)
    
    return x
#we load the content image
content_img = load_img_and_preprocess('./-----.jpg')

#now we load style image and resize it to the content image's size
h, w = content_img.shape[1:3]
style_img = load_img_and_preprocess('./-----.jpg', (h, w))

#we store the batch shape and shape to use throughout the rest of the script
batch_shape = content_img.shape
shape = content_img.shape[1:]
#we create out vgg model here
vgg = VGG16_AvgPool(shape)

#create the content model for only 1 output
content_model = Model(vgg.input, vgg.layers[13].get_output_at(1))
content_target = K.variable(content_model.predict(content_img))

#now we create the style model
#we need multiple outputs
#same approach as in style_transfer_2

symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
#style model which gives multiple outputs like done previously
style_model = Model(vgg.inputs, symbolic_conv_outputs)
#making tagets for the style model
style_model_outputs = [K.variable(y) for y in style_model.predict(style_img)]

#we can weight the losses as well
#here we take the weight of the content loss to be 1
#and only weight the style losses
style_weights = [1, 2, 3, 4, 5]

#create the total loss which is the sum of content + style loss
loss = K.mean(K.square(content_model.output - content_target))

for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_model_outputs):
    loss += w * (style_loss(symbolic[0], actual[0]))

#once again, we will create the gradients and the loss + grads function
#we can use any model's input as they point to the same place in memory
grads = K.gradients(loss, vgg.input)

#like theano
get_loss_and_grads = K.function(inputs=[vgg.input], outputs=[loss] + grads)

def get_loss_and_grads_wrapper(x_vec):
    l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.astype(np.float64)
final_img = minimise(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_img(final_img))
plt.show()