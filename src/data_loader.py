# src/data_loader.py
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Assuming load_and_preprocess_image is in src.data_preprocessing
from data_preprocessing import load_and_preprocess_image

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, project_root, batch_size, target_dims, num_classes, shuffle=True, augment=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with 'relative_filepath' and 'label' columns.
            project_root (Path): Absolute path to the project root.
            batch_size (int): Size of each batch.
            target_dims (tuple): Tuple (height, width) for image resizing.
            num_classes (int): Number of classes (1 for binary).
            shuffle (bool): Whether to shuffle data at the end of each epoch.
            augment (bool): Whether to apply data augmentation (not implemented yet).
        """
        self.df = dataframe.copy()
        self.project_root = Path(project_root)
        self.batch_size = batch_size
        self.target_dims = target_dims
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment # Placeholder for now

        # Encode labels: PNEUMONIA=1, NORMAL=0
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['label'])
        print("Label Encoding used by DataGenerator:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"- {class_name}: {self.label_encoder.transform([class_name])[0]}")
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_df = self.df.iloc[batch_indexes]

        # Generate data
        X, y = self.__data_generation(batch_df)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_df):
        """Generates data containing batch_size samples."""
        X = np.empty((self.batch_size, *self.target_dims, 3)) # Assuming 3 channels (RGB)
        y = np.empty((self.batch_size), dtype=int)

        for i, (idx, row) in enumerate(batch_df.iterrows()):
            image_path = self.project_root / row['relative_filepath']
            # Preprocess image using the function from data_preprocessing.py
            img_array = load_and_preprocess_image(str(image_path), target_size=self.target_dims)

            if img_array is not None:
                X[i,] = img_array
                y[i] = row['label_encoded']
            else:
                X[i,] = np.zeros((*self.target_dims, 3))
                y[i] = 0
                print(f"Warning: Failed to load image {image_path}, using zeros.")
                
        return X, y
