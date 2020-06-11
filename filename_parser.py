import pickle
import os


def get_file_list(dataset_name):
    names = []
    path = f'D:\\PROJEKTY\\Python\\ML risk analysis\\TRAIN_DATA\\{dataset_name}\\'
    for r, d, f in os.walk(path):
        for file in f:
            file = file.replace('-v.pickle', '')
            file = file.replace('-t.pickle', '')
            if file not in names:
                print(file)
                names.append(file)
    return names
