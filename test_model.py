import h5py

try:
    with h5py.File('analyzer/models/gender_classification_vgg16.h5', 'r') as f:
        print("File is valid.")
except Exception as e:
    print(f"File is invalid: {e}")