class Progressbar:
    def __init__(self, goal, steps=20, length=60, name=''):
        self.goal = goal
        self.steps = steps
        self.done = 0
        self.name = name
        self.length = length

        self.step_size = goal / steps

    def update(self, i):
        if i % 10 == 0:
            while i >= self.done * self.step_size:
                self.message()
                self.done += 1

    def message(self):
        bar = '[' + '=' * round(self.done * self.length / self.steps) + '>' + '-' * (
                self.length - round(self.done * self.length / self.steps)) + ']'

        print(bar, self.name, round(100 * self.done / self.steps), "%")

    def __del__(self):
        self.message()
        # print(self.name, "finished!")


class Logger:
    def __init__(self, params):
        from datetime import datetime

        self.data = []
        timestampStr = datetime.now().strftime("%d/%m/%Y- %H:%M:%S")
        self.data.append(timestampStr)
        for p in params:
            self.log(p)

    def log(self, new_data):
        self.data.append(new_data)

    def save(self, folder_path, name):
        import csv
        
        try:
            with open(folder_path+'/'+name, 'a+', newline='') as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(self.data)
        except:
            from os import makedirs
            makedirs(folder_path)



class FilenameParser:
    def get_file_list(dataset_name):
        import os
        names = []
        path = f'TRAIN_DATA/{dataset_name}/'
        for r, d, f in os.walk(path):
            for file in f:
                file = file.replace('-v.pickle', '')
                file = file.replace('-t.pickle', '')
                if file not in names:
                    print(file)
                    names.append(file)
        return names
