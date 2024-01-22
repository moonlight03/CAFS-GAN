import os
dir = '../data/afhq/train'
classes = [d.name for d in os.scandir(dir) if d.is_dir()]
print(classes)