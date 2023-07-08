from __future__ import print_function, division
from builtins import range, input


#in this script we are focusing on generating an image with the same style
#as the input image, but NOT the same content. It should capture just the essence of the style.

#imports    
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from style_transfer_1 import VGG16_AvgPool, unpreprocess, scale_img

from scipy.optimize import fmin_l_bfgs_b
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K


def gram_matrix(img):
    #input shape of image is (H, W, C) where C = #feature maps
    #we have to reshape it to (C, H*W) to calculate the gram matrix
    
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    #now we calculate the gram matrix
    # gram = XX^T / N
    
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
    return G

def style_loss(y, t):
    loss = K.mean(K.square(gram_matrix(y) - gram_matrix(t)))
    return loss

def minimise(fn, epochs, batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    
    for i in range(epochs):
        x, l, _ = fmin_l_bfgs_b(func= fn, x0= x, maxfun= 20)
        x = np.clip(x, -127, 127)
        print('iter=%s, loss=%s' % (i, l))
        losses.append(l)
    
    print('durantion: ', datetime.now() - t0)
    plt.plot(losses)
    plt.show()
    
    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]


if __name__ == '__main__':
    path = './starrynight.jpg'
    img = image.load_img(path)
    
    #convert into array make dimensions from (H, W, C) to (1, H, W, C) and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    
    batch_shape = x.shape
    shape = x.shape[1:]
    
    #we are taking the first convolution at each block to be our target output
    vgg = VGG16_AvgPool(shape)
    
    #since we replaced the maxpool layers of vgg with avg pool,
    #there exist two models in memory. thats just how keras works
    #in order to get our output, we consider the output at index 1
    #index 0 corresponds to the original vgg with maxpool
    
    symbolic_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers if layer.name.endswith('conv1')]
    
    #symbolic_conv_outputs = symbolic_conv_outputs[:2]
    
    #we make a model that outputs multiple layers' outputs
    multi_output_model = Model(vgg.input, symbolic_conv_outputs)
    
    #calculate the targets that are the outputs at each layer
    style_layer_outputs = [K.variable(y) for y in multi_output_model.predict(x)]
    
    #calc the total style loss
    loss = 0
    for symbolic, actual in zip(symbolic_conv_outputs, style_layer_outputs):
        loss += style_loss(symbolic[0], actual[0])
        
    grads = K.gradients(loss, multi_output_model.input)
    
    
    get_loss_and_grads = K.function(inputs=[multi_output_model.input], outputs= [loss] + grads)
    
    def get_loss_and_grads_wrapper(x_vec):
        #we cannot use get_loss_and_grads directly
        #input to the minimiser function must be 1-d array
        #input to get_loss_and_grads must be [batch of images]
        
        #gradient must also be a 1-D array
        #and both loss and grads must be float64 to avoid error
        
        l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
    
    final_img = minimise(get_loss_and_grads_wrapper, 10, batch_shape)
    plt.imshow(scale_img(final_img))
    plt.show()
    
    
    