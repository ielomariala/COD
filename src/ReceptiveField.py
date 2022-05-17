from ResNet import Conv2D_BN
from tensorflow.keras.layers import Input, Concatenate, Activation
from tensorflow.keras import Model

# Receptive Field 
def ReceptiveField(input_shape, nb_filters):
    
    input_layer = Input(shape=input_shape)
    # Block 0
    branch0 = Conv2D_BN(input_layer, nb_filters, kernel_size=1)
    
    # Block 1
    branch1 = Conv2D_BN(input_layer, nb_filters, kernel_size=1)
    branch1 = Conv2D_BN(branch1, nb_filters, kernel_size=(1, 3), padding=(0, 1))
    branch1 = Conv2D_BN(branch1, nb_filters, kernel_size=(3, 1), padding=(1, 0))
    branch1 = Conv2D_BN(branch1, nb_filters, kernel_size=3, padding=3, dilation_rate=3)

    # Block 2
    branch2 = Conv2D_BN(input_layer, nb_filters, kernel_size=1)
    branch2 = Conv2D_BN(branch2, nb_filters, kernel_size=(1, 5), padding=(0, 2))
    branch2 = Conv2D_BN(branch2, nb_filters, kernel_size=(5, 1), padding=(2, 0))
    branch2 = Conv2D_BN(branch2, nb_filters, kernel_size=3, padding=5, dilation_rate=5)
    
    # Block 3
    branch3 = Conv2D_BN(input_layer, nb_filters, kernel_size=1)
    branch3 = Conv2D_BN(branch3, nb_filters, kernel_size=(1, 7), padding=(0, 3))
    branch3 = Conv2D_BN(branch3, nb_filters, kernel_size=(7, 1), padding=(3, 0))
    branch3 = Conv2D_BN(branch3, nb_filters, kernel_size=3, padding=7, dilation_rate=7)
    
    # Concatenate models
    merge = Concatenate()([branch0, branch1, branch2, branch3])
    
    # Block 1 of Receptive Field
    conv_cat = Conv2D_BN(merge, nb_filters, kernel_size=3, padding=1)
    
    # Block 2 of Receptive Field
    conv_res = Conv2D_BN(input_layer, 32, kernel_size=1)

    # Final addition
    rf = Activation('relu')(conv_cat+conv_res)

    # Build final Receptive Field model:
    RF = Model(inputs=input_layer, outputs=rf)
    
    return RF
