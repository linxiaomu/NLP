# For this example, we will first have a helper function that
# generates some CSV files with random label and data.

import csv
import random


def generate_csv(file_label, num_rows: int = 5000, num_features: int = 20) -> None:
    fieldnames = ['label'] + [f'c{i}' for i in range(num_features)]

    writer = csv.DictWriter(open(f"sample_data{file_label}.csv", "w",newline=''), fieldnames=fieldnames)
    writer.writerow({col: col for col in fieldnames})  # writing the header row
    for i in range(num_rows):
        row_data = {col: random.random() for col in fieldnames}
        row_data['label'] = random.randint(0, 9)
        writer.writerow(row_data)


# Next, we will build our DataPipes to read and parse through the generated CSV files:
import numpy as np
import torchdata.datapipes as dp


def build_datapipes(root_dir="."):
    # 给定根目录的路径，生成根目录文件的路径名（路径+文件名），可以提供多个根目录
    datapipe = dp.iter.FileLister(root_dir)
    print(list(datapipe))
    # 根据输入filter_fn从源数据管道中过滤出元素
    datapipe = datapipe.filter(filter_fn=lambda filename: "sample_data" in filename and filename.endswith(".csv"))
    # 给定路径名，打开文件并在元组中生成路径名和文件流。
    datapipe = dp.iter.FileOpener(datapipe, mode='rt')
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
    print('------')
    print(list(datapipe))
    print(datapipe)
    datapipe = datapipe.map(lambda row: {"label": np.array(row[0], np.int32),
                                         "data": np.array(row[1:], dtype=np.float64)})
    return datapipe


# Lastly, we will put everything together in '__main__' and pass the DataPipe into the DataLoader.
from torch.utils.data import DataLoader

if __name__ == '__main__':
    num_files_to_generate = 3
    for i in range(num_files_to_generate):
        generate_csv(file_label=i)
    datapipe = build_datapipes()
    dl = DataLoader(dataset=datapipe, batch_size=50, shuffle=True)
    first = next(iter(dl))
    labels, features = first['label'], first['data']
    print(f"Labels batch shape: {labels.size()}")
    print(f"Feature batch shape: {features.size()}")

# from torchdata.datapipes.iter import FileLister
# dp = FileLister(root=".", recursive=True)
# print(list(dp))