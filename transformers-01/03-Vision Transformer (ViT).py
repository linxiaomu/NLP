from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

url = '000000039769.jpg'
url2 = 'test.jpg'
url3 = 'test2.jpg'
url4 = 'test3.jpg'
url5 = '6_56884826,白夜ReKi-ゴブレット.jpg'
url6 = '118_57526556,Linfi-MUU-◇レム.jpg'
url7 = 'u=3848402655,92542552&fm=26&gp=0.jpg'
url8 = '下载.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(url7)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

