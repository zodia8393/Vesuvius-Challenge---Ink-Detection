import os
import numpy as np
import h5py
from skimage.transform import resize

def load_data(data_path):
    with h5py.File(data_path, "r") as file:
        x_ray_scans = np.array(file['xray_scans'])
        infrared_photos = np.array(file['infrared_photos'])
    return x_ray_scans, infrared_photos

def preprocess_xray_scans(x_ray_scans, target_shape):
    preprocessed_xray_scans = np.zeros((x_ray_scans.shape[0], *target_shape))
    
    for i, scan in enumerate(x_ray_scans):
        preprocessed_xray_scans[i] = resize(scan, target_shape, preserve_range=True, anti_aliasing=True)

    # Normalize data to the range [0, 1]
    preprocessed_xray_scans = preprocessed_xray_scans / np.max(preprocessed_xray_scans)
    
    return preprocessed_xray_scans

def preprocess_infrared_photos(infrared_photos, target_shape):
    preprocessed_infrared_photos = np.zeros((infrared_photos.shape[0], *target_shape))
    
    for i, photo in enumerate(infrared_photos):
        preprocessed_infrared_photos[i] = resize(photo, target_shape, preserve_range=True, anti_aliasing=True)

    # Normalize data to the range [0, 1]
    preprocessed_infrared_photos = preprocessed_infrared_photos / np.max(preprocessed_infrared_photos)
    
    return preprocessed_infrared_photos

def split_data(x_ray_scans, infrared_photos, train_ratio=0.8, val_ratio=0.1):
    dataset_size = x_ray_scans.shape[0]
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    x_train = x_ray_scans[train_indices]
    y_train = infrared_photos[train_indices]

    x_val = x_ray_scans[val_indices]
    y_val = infrared_photos[val_indices]

    x_test = x_ray_scans[test_indices]
    y_test = infrared_photos[test_indices]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def preprocess_data(data_path, target_shape=(128, 128, 128)):
    x_ray_scans, infrared_photos = load_data(data_path)

    preprocessed_xray_scans = preprocess_xray_scans(x_ray_scans, target_shape)
    preprocessed_infrared_photos = preprocess_infrared_photos(infrared_photos, target_shape)

    train_data, val_data, test_data = split_data(preprocessed_xray_scans, preprocessed_infrared_photos)

    return train_data, val_data, test_data

if __name__ == "__main__":
    data_path = "path/to/your/dataset.h5"
    train_data, val_data, test_data = preprocess_data(data_path)
    print("Data preprocessed and split into train, validation, and test sets.")
