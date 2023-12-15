import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_creation import create_models

def train_and_evaluate(image_paths, labels):
    img_size = (150, 150)
    batch_size = 16

    models = create_models()
    model_names = ['Xception', 'DenseNet169']

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=50,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.25,
        zoom_range=0.1,
        channel_shift_range=20,
        validation_split=0.3
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, test_index in skf.split(image_paths, labels):
        train_paths = np.array(image_paths)[train_index]
        train_labels = np.array(labels)[train_index]
        test_paths = np.array(image_paths)[test_index]
        test_labels = np.array(labels)[test_index]

        train_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        for model, model_name in zip(models, model_names):
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Train the model
            model.fit(train_generator, epochs=10, validation_data=validation_generator)
            
            # Evaluate the model
            evaluation = model.evaluate(validation_generator)
            print(f"{model_name} - Validation Loss: {evaluation[0]}, Validation Accuracy: {evaluation[1]}")

    return models
