from datetime import datetime
import pytz
import dateparser
from os import makedirs

class Progressbar:
    def __init__(self, goal, steps=20, length=40, name=''):
        self.goal = goal
        self.steps = steps
        self.done = 0
        self.name = name
        self.length = length

        self.step_size = goal / steps

    def update(self, i):
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

def create_dir(path):
    try:
        makedirs(path)
    except  FileExistsError:
        pass

def date_from_ms(ms):
    return datetime.utcfromtimestamp(ms//1000)

def datetime_to_ms(datetime):
    pass

def now():
    return datetime.now()

def date_to_ms(date_str):
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)