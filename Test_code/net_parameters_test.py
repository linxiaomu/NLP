import torchvision
from torch.utils.tensorboard import SummaryWriter

model = torchvision.models.resnet50(False)
# print(list(model.parameters()))

for name, p in model.named_parameters():
    if 'bias' in name:
        print(name)
    # print(p)