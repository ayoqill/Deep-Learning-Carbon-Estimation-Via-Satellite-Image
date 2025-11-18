from tensorflow.keras import layers, models

class Estimator:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)
        # Example of a U-Net architecture
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
        merge6 = layers.concatenate([up6, conv4])
        conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
        conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
        merge7 = layers.concatenate([up7, conv3])
        conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
        conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
        merge8 = layers.concatenate([up8, conv2])
        conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
        conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
        merge9 = layers.concatenate([up9, conv1])
        conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
        conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data, train_labels, validation_data, validation_labels, epochs=50, batch_size=16):
        self.model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), 
                       epochs=epochs, batch_size=batch_size)

    def predict(self, input_data):
        return self.model.predict(input_data)