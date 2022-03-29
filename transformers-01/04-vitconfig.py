from transformers import ViTModel, ViTConfig
# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig()
# Initializing a model from the vit-base-patch16-224 style configuration
model = ViTModel(configuration)
# Accessing the model configuration
configuration = model.config
print(configuration)