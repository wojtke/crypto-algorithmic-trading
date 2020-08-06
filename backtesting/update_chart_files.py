import shutil
import os


folder_docelowy = "STATISTICS"

for d in os.listdir(folder_docelowy):
	print(folder_docelowy+"\\"+d)
	shutil.copy2('chart.py', folder_docelowy+"\\"+d)

