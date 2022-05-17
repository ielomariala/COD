from ResNet import Conv2D_BN
from tensorflow.keras.layers import Input, UpSampling2D, Concatenate, Conv2D
from tensorflow.keras import Model


# Search Module & Identification module
def PartialDecoderComponent(shape1, shape2, shape3, shape4=None, channel=32, module='SearchModule'):
    # print(shape1, shape2, shape3, shape4)
    x1 = Input(shape=shape1)
    x2 = Input(shape=shape2)
    x3 = Input(shape=shape3)
    if shape4 is not None:
        x4 = Input(shape=shape4)

    # Layers
    UpSampling = UpSampling2D(size=2, interpolation='bilinear')

    conv2D_BN = lambda x, multiplier=1: Conv2D_BN(x, nb_filters=multiplier*channel, kernel_size=3, padding=1)

    conv1x1 = Conv2D(filters=1, kernel_size=1)
    
    # Model 
    if module == 'SearchModule':
        x1_1 = x1
        
        x2_1 = conv2D_BN(UpSampling(x1)) * x2
        
        x3_1 = conv2D_BN(UpSampling(UpSampling(x1))) * conv2D_BN(UpSampling(x2)) * x3
        
        x2_2 = Concatenate(axis=-1)([x2_1, conv2D_BN(UpSampling(x1_1))])
        x2_2 = conv2D_BN(x2_2, 2)

        x3_2 = Concatenate(axis=-1)([x3_1,  conv2D_BN(UpSampling(x2_2), 2), x4])
        x3_2 = conv2D_BN(x3_2, 4)

        x = conv2D_BN(x3_2, 4)

        x = conv1x1(x)
        
        model = Model(inputs=[x1, x2, x3, x4], outputs=x)
    
    elif module == 'IdentificationModule':
        x1_1 = x1

        x2_1 = conv2D_BN(UpSampling(x1)) * x2

        x3_1 = conv2D_BN(UpSampling(UpSampling(x1))) * conv2D_BN(UpSampling(x2)) * x3

        x2_2 = Concatenate(axis=-1)([x2_1, conv2D_BN(UpSampling(x1_1))])
        x2_2 = conv2D_BN(x2_2, 2)

        x3_2 = Concatenate(axis=-1)([x3_1,  conv2D_BN(UpSampling(x2_2), 2)])
        x3_2 = conv2D_BN(x3_2, 3)

        x = conv2D_BN(x3_2, 3)
        x = conv1x1(x)
        
        model = Model(inputs=[x1, x2, x3], outputs=x)
    
    else:
        raise ValueError('Module not found')
    
    return model

