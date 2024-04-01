import pandas as pd
import os
import matplotlib.pyplot as plt

train_directory = "00_data/data/train"
test_directory = "00_data/data/test"
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create dataframe from directories
def create_dataframe_from_directory(directory):
    file_paths = []
    labels = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_directory = os.path.join(directory, class_name)
        for filename in os.listdir(class_directory):
            file_paths.append(os.path.join(class_directory, filename))
            labels.append(class_name)
    data = {'file_paths': file_paths, 'labels': labels}
    return pd.DataFrame(data)

# Create dataframes
train_df = create_dataframe_from_directory(train_directory)
test_df = create_dataframe_from_directory(test_directory)
print(train_df.shape)
print(test_df.shape)

save_directory = "00_data"

# Save train dataframe to CSV
train_df.to_csv(os.path.join(save_directory, 'train_data.csv'), index=False)
# Save test dataframe to CSV 
test_df.to_csv(os.path.join(save_directory, 'test_data.csv'), index=False)
