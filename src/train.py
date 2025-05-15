import pandas as pd
import yaml
from pathlib import Path
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import logging
import os
from sklearn.model_selection import train_test_split
import platform

from model_builder import create_cnn_model
from data_loader import CustomDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_params():
    """Loads parameters from config/params.yaml"""
    params_path = PROJECT_ROOT / "config" / "params.yaml"
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    logging.info(f"Parameters loaded from {params_path}")
    return params

def train_model():
    params = load_params()

    # --- MPS/GPU Check (Revised based on your test_mps.py output) ---
    logging.info(f"TensorFlow Version: {tf.__version__}")
    logging.info(f"Platform: {platform.system()} {platform.processor()}")

    gpu_devices = tf.config.list_physical_devices('GPU')

    if gpu_devices:
        logging.info(f"Found TensorFlow physical GPU(s): {gpu_devices}")
        if platform.system() == "Darwin" and platform.processor() == "arm":
            logging.info("This is an ARM Mac; the GPU found is the MPS device. TensorFlow will attempt to use it.")
        else:
            logging.info("GPU found. TensorFlow will attempt to use it.")
    else:
        logging.info("No GPU found by tf.config.list_physical_devices('GPU'). TensorFlow will use CPU.")

    # Load data metadata
    metadata_csv_path = PROJECT_ROOT / params['PROCESSED_METADATA_CSV']
    full_df = pd.read_csv(metadata_csv_path)
    logging.info(f"Metadata loaded from {metadata_csv_path}. Shape: {full_df.shape}")

    # --- Modified Data Splitting ---
    # Combine original 'train' and 'val' from the CSV to form a larger pool for training and validation
    # Keep the original 'test' set separate for final evaluation later
    original_train_val_pool_df = full_df[full_df['split'].isin(['train', 'val'])].copy()
    test_df = full_df[full_df['split'] == 'test'].copy().reset_index(drop=True) # Keep test_df for later

    if original_train_val_pool_df.empty:
        logging.error("No data available in 'train' or 'val' splits from metadata.csv. Exiting.")
        return

    logging.info(f"Total samples in original train+val pool: {len(original_train_val_pool_df)}")

    # Split this pool into new training and validation sets
    # Stratify by 'label' to maintain class proportions in both new sets
    try:
        train_df, val_df = train_test_split(
            original_train_val_pool_df,
            test_size=0.20,  # 20% for the new validation set
            random_state=42, # For reproducibility
            stratify=original_train_val_pool_df['label'] if 'label' in original_train_val_pool_df.columns and original_train_val_pool_df['label'].nunique() > 1 else None
        )
    except ValueError as e: # Handles cases where stratification might not be possible (e.g., too few samples of a class)
        logging.warning(f"Stratification failed during train/val split: {e}. Splitting without stratification.")
        train_df, val_df = train_test_split(
            original_train_val_pool_df,
            test_size=0.20,
            random_state=42
        )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logging.info(f"New training samples: {len(train_df)}")
    logging.info(f"New validation samples: {len(val_df)}")
    logging.info(f"Test samples (set aside): {len(test_df)}")
    # --- End Modified Data Splitting ---

    # Create data generators
    target_dims = (params['IMAGE_HEIGHT'], params['IMAGE_WIDTH'])
    train_generator = CustomDataGenerator(
        dataframe=train_df,
        project_root=PROJECT_ROOT,
        batch_size=params['BATCH_SIZE'],
        target_dims=target_dims,
        num_classes=params['NUM_CLASSES'],
        shuffle=True,
        augment=False # Consider enabling simple augmentation later
    )
    val_generator = CustomDataGenerator(
        dataframe=val_df,
        project_root=PROJECT_ROOT,
        batch_size=params['BATCH_SIZE'],
        target_dims=target_dims,
        num_classes=params['NUM_CLASSES'],
        shuffle=False
    )

    # MLflow setup
    mlflow.set_experiment(params['MLFLOW_EXPERIMENT_NAME'])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"Starting MLflow Run ID: {run_id}")
        mlflow.log_params(params)

        input_shape = (params['IMAGE_HEIGHT'], params['IMAGE_WIDTH'], params['NUM_CHANNELS'])
        model = create_cnn_model(input_shape, params['NUM_CLASSES'])
        model.summary(print_fn=logging.info)

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
        loss_function = 'binary_crossentropy' if params['NUM_CLASSES'] == 1 else 'categorical_crossentropy'
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
        logging.info(f"Model compiled with optimizer: Adam, loss: {loss_function}")

        model_save_dir = PROJECT_ROOT / params['SAVED_MODEL_DIR']
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model_path_with_run_id = model_save_dir / f"{Path(params['SAVED_MODEL_NAME']).stem}_{run_id}{Path(params['SAVED_MODEL_NAME']).suffix}"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_path_with_run_id), # Ensure it's a string
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
        ]

        logging.info("Starting model training...")
        history = model.fit(
            train_generator,
            epochs=params['EPOCHS'],
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )

        best_val_accuracy = max(history.history['val_accuracy'])
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        logging.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")

        if model_path_with_run_id.exists():
            logging.info(f"Loading best model from: {model_path_with_run_id}")
            model = tf.keras.models.load_model(str(model_path_with_run_id)) # Ensure path is string
        else:
            logging.warning(f"Best model path not found: {model_path_with_run_id}. Logging the last state model.")

        mlflow.tensorflow.log_model(model=model, artifact_path="pneumonia-cnn-model", registered_model_name="PneumoniaCNN_TF")
        logging.info(f"Model saved to MLflow Run ID: {run_id} under artifact path 'pneumonia-cnn-model'")
        logging.info(f"Model also saved locally at: {model_path_with_run_id}")

        mlflow.set_tag("training_status", "completed")
        logging.info("MLflow run completed.")


if __name__ == '__main__':
    train_model()