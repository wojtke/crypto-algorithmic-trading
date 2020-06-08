from datetime import datetime
import csv

class Logger():
    def __init__(self, params):
        self.data = []
        timestampStr = datetime.now().strftime("%d/%m/%Y- %H:%M:%S")
        self.data.append(timestampStr)
        for p in params:
            self.log(p)

    def log(self, new_data):
        self.data.append(new_data)

    def save(self, path):
        with open(path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.data)