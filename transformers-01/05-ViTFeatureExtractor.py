from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


url4 = 'test3.jpg'
image = Image.open(url4)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

plt.figure("dog")
plt.imshow(outputs)
plt.show()
