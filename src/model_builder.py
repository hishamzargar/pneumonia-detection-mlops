import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_cnn_model(input_shape, num_classes):
    """
    Creates a Convolutional Neural Network model using the Keras Functional API.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.models.Model: A Keras Functional API model.
    """
    # Define the input layer explicitly
    # Naming the input can also help tf2onnx
    inputs = Input(shape=input_shape, name="input_image_layer")

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1_1")(inputs)
    x = BatchNormalization(name="bn1_1")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2_1")(x)
    x = BatchNormalization(name="bn2_1")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3_1")(x)
    x = BatchNormalization(name="bn3_1")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Flattening and Dense Layers
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation='relu', name="dense1")(x)
    x = BatchNormalization(name="bn_dense1")(x)
    x = Dropout(0.5, name="dropout_dense1")(x)

    # Output Layer
    output_layer_name = "output_layer" # Define a consistent name
    if num_classes == 1: # Binary classification
        activation = 'sigmoid'
        outputs = Dense(num_classes, activation=activation, name=output_layer_name)(x)
    else: # Multi-class classification
        activation = 'softmax'
        outputs = Dense(num_classes, activation=activation, name=output_layer_name)(x)

    # Create the model by specifying inputs and outputs
    model = Model(inputs=inputs, outputs=outputs, name="PneumoniaCNN_Functional")

    print(f"Functional model created. Input name: {inputs.name}, Output name: {outputs.name}")
    return model

if __name__ == '__main__':
    img_height, img_width, channels = 224, 224, 3
    num_classes_example = 1
    example_model = create_cnn_model((img_height, img_width, channels), num_classes_example)
    example_model.summary()