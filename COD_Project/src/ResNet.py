from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add, ZeroPadding2D

def Conv2D_BN(x, nb_filters, kernel_size=3, strides=(1, 1), conv_padding='valid', activation='relu', dilation_rate=1, padding=0):
    if padding != 0:
        x = ZeroPadding2D(padding=padding)(x)
    x = Conv2D(nb_filters, kernel_size, strides=strides, padding=conv_padding, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def IdentityBlock(input_tensor, filters, kernel_size=3, strides=(1, 1)):
    filter_idx = 0

    x = Conv2D_BN(input_tensor, filters[filter_idx], kernel_size=1, strides=strides)
    filter_idx += 1

    x = Conv2D_BN(x, filters[filter_idx], kernel_size=kernel_size, conv_padding='same')
    filter_idx += 1
    
    x = Conv2D(filters[filter_idx], strides=strides, kernel_size=1)(x)
    x = BatchNormalization()(x)  

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def ConvBlock(input_tensor, filters, kernel_size=3, strides=(2, 2)):
    filter_idx = 0

    x = Conv2D_BN(input_tensor, filters[filter_idx], kernel_size=1, strides=strides)
    filter_idx += 1

    x = Conv2D_BN(x, filters[filter_idx], kernel_size=kernel_size, conv_padding='same')
    filter_idx += 1

    x = Conv2D(filters[filter_idx], kernel_size=1)(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters[filter_idx], kernel_size=1, strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def layer_1(input_tensor, filters=[64, 64, 256], strides=(1, 1)):
    x = input_tensor
    x = ConvBlock(x, filters, strides=strides)
    x = IdentityBlock(x, filters)
    x = IdentityBlock(x, filters)
    return x

def layer_2(input_tensor, filters=[128, 128, 512]):
    x = input_tensor
    x = ConvBlock(x, filters)
    x = IdentityBlock(x, filters)
    x = IdentityBlock(x, filters)
    x = IdentityBlock(x, filters)
    return x


def layer_3(input_tensor, filters=[256, 256, 1024]):
    x = input_tensor
    x = ConvBlock(x, filters)
    x = IdentityBlock(x, filters)
    x = IdentityBlock(x, filters)
    x = IdentityBlock(x, filters)
    x = IdentityBlock(x, filters)
    x = IdentityBlock(x, filters)
    return x

