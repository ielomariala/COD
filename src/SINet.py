from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv2D,
                                        BatchNormalization,
                                        Activation,
                                        Concatenate,
                                        UpSampling2D,
                                        ZeroPadding2D,
                                        MaxPooling2D)

from tensorflow.keras.activations import sigmoid

from ResNet import layer_1, layer_2, layer_3
from ReceptiveField import ReceptiveField as RF
from PartialDecoderComponent import PartialDecoderComponent as PDC
from SearchAttention import SearchAttention_v1 as SA1
from SearchAttention import SearchAttention_v2 as SA2


def SINet_ResNet50(image_size=352, channel=32, search_attention=None):
    x = Input(shape=(image_size, image_size, 3))

    UpSampling = lambda x, size=2: UpSampling2D(size=size, interpolation='bilinear')(x)

    # Head
    x0 = ZeroPadding2D(3)(x)
    x0 = Conv2D(64, 7, strides=2, use_bias=False)(x0)
    x0 = BatchNormalization()(x0)
    
    # Low level features
    x0 = Activation('relu')(x0) 
    x0 = ZeroPadding2D((1, 1))(x0)
    x0 = MaxPooling2D((3, 3), strides=(2, 2))(x0)
    
    x1 = layer_1(x0)
    x2 = layer_2(x1)
    
    # Search Module (SM):
    x01 = Concatenate()([x0, x1])
    x01_down = MaxPooling2D((2, 2), strides=2)(x01)

    rf_low_sm = RF(x01_down.shape[1:], channel)
    x01_sm_rf = rf_low_sm(x01_down)
    
    x2_sm = x2
    x3_sm = layer_3(x2_sm)
    x4_sm = layer_1(x3_sm, filters=[512, 512, 2048], strides=(2, 2))
    
    x2_sm_cat = Concatenate()([x2_sm, UpSampling(x3_sm), UpSampling(UpSampling(x4_sm))])    
    x3_sm_cat = Concatenate()([x3_sm, UpSampling(x4_sm)])
    
    rf2_sm = RF(x2_sm_cat.shape[1:], channel)
    x2_sm_rf = rf2_sm(x2_sm_cat)
    
    rf3_sm = RF(x3_sm_cat.shape[1:], channel)
    x3_sm_rf = rf3_sm(x3_sm_cat)
    
    rf4_sm = RF(x4_sm.shape[1:], channel)
    x4_sm_rf = rf4_sm(x4_sm)
    
    pdc_sm = PDC(x4_sm_rf.shape[1:], x3_sm_rf.shape[1:], x2_sm_rf.shape[1:], x01_sm_rf.shape[1:], channel, module='SearchModule')
    camouflage_map_sm = pdc_sm([x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf])

    # Search Attention (SA)
    if search_attention == "v1":
        tmp = sigmoid(camouflage_map_sm)
        x2_sa = SA1(tmp, x2)
    elif search_attention == "v2":
        tmp = sigmoid(camouflage_map_sm)
        x2_sa = SA2(tmp, x2)
    else:
        x2_sa = x2 # No attention module
    

    # Identification Module (IM)
    x3_im = layer_3(x2_sa)
    # print("Passed layer 3")
    x4_im = layer_1(x3_im, filters=[512, 512, 2048], strides=(2, 2))
    # print("Passed layer 1")

    rf2_im = RF(x2_sa.shape[1:], channel)
    x2_im_rf = rf2_im(x2_sa)
    # print("Passed RF2")

    rf3_im = RF(x3_im.shape[1:], channel)
    x3_im_rf = rf3_im(x3_im)
    # print("Passed RF3")

    rf4_im = RF(x4_im.shape[1:], channel)
    x4_im_rf = rf4_im(x4_im)
    # print("Passed RF4")

    pdc_im = PDC(x4_im_rf.shape[1:], x3_im_rf.shape[1:], x2_im_rf.shape[1:], channel=channel, module='IdentificationModule')
  
    camouflage_map_im = pdc_im([x4_im_rf, x3_im_rf, x2_im_rf])
    # print("Passed PDC")

    camouflage_map_sm = UpSampling2D(size=8, interpolation='bilinear', name='SM')(camouflage_map_sm) 
    camouflage_map_im = UpSampling2D(size=8, interpolation='bilinear', name='IM')(camouflage_map_im) 

    output_model = Model(x, outputs=[camouflage_map_sm, camouflage_map_im])
    return output_model