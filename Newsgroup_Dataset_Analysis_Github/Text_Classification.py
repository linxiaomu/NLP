# Text Classification Methods in NLP using Deep Learning
# Using pre-trained word embeddings
# implementations are all based in Keras
# 地址：https://github.com/Sameeksharajsb/20-Newsgroup-Dataset-Analysis/blob/main/Text%20Classification%20methods%20in%20NLP%20using%20Deep%20Learning.ipynb
# 不知道怎么获取数据



import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import logging, sys
logging.disable(sys.maxsize)

data_path = keras.utils.get_file(
    "news20.tar.gz",
    "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
    untar=True
)

# import os
# import pathlib
#
# data_dir = pathlib.Path(data_path).parent / "20_newsgroup"
# dirnames = os.listdir(data_dir)
# print("Number of directories:", len(dirnames))
# print("Directory names:", dirnames)

# samples = []
# labels = []
# class_names = []
# class_index = 0
# for dirname in sorted(os.listdir(data_dir)):
#     class_names.append(dirname)
#     dirpath = data_dir / dirname
#     fnames = os.listdir(dirpath)
#     print("Processing %s, %d files found" % (dirname, len(fnames)))
#     for fname in fnames:
#         fpath = dirpath / fname
#         f = open(fpath, encoding="latin-1")
#         content = f.read()
#         lines = content.split("\n")
#         lines = lines[10:]
#         content = "\n".join(lines)
#         samples.append(content)
#         labels.append(class_index)
#     class_index += 1
#
# print("Classes:", class_names)
# print("Number of samples:", len(samples))


