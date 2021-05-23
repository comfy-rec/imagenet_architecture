from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, LeakyReLU, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_model_16(img_height, img_width, img_channel, class_count, weight_decay):
    input_layer = Input(shape=(img_height, img_width, img_channel))
    # block1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block1_conv1')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ,name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    output_layer = Dense(class_count, activation='softmax', name='predictions')(x)

    model = Model(input_layer, output_layer, name='vgg16')

    return model

def build_model_19(img_height, img_width, img_channel, class_count, weight_decay):
    input_layer = Input(shape=(img_height, img_width, img_channel))
    # block1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block1_conv1')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    output_layer = Dense(class_count, activation='softmax', name='predictions')(x)

    model = Model(input_layer, output_layer, name='vgg19')

    return model

#
# model_build_vgg_16 = build_model_16(224, 224, 3, 1000, 1e-4)
# model_build_vgg_16.summary()
# model_build_vgg_19 = build_model_19(224, 224, 3, 1000, 1e-4)
# model_build_vgg_19.summary()