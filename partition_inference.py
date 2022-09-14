import torch
import json
import collections
import yaml
from torchvision import transforms
from dnn_architectures.vgg import VGGNet
from PIL import Image
from partial_inference_server import partial_inference_server_side


def load_configuration():
    """
    Reads the configuration.yaml file for model configuration
    """
    configuration = {}
    with open("configuration.yaml", 'r') as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return configuration['model']


print(f"Reading configuration to load model")
model_configuration = load_configuration()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)

model_path = model_configuration['path']
print(f"{model_configuration['architecture']} pre-trained model loaded")

test_model = VGGNet(model_configuration['classes'])
test_model.to(device)

loaded_model_weights = torch.load(model_path)
modified_weights = collections.OrderedDict()

# 应该只是为了后续名称使用方便
for layer_name, weights in loaded_model_weights.items():
    # print('layer_name: ', layer_name)
    new_layer_name = layer_name.replace(".", "_", 1)
    if "classifier_6" not in new_layer_name:
        new_layer_name = new_layer_name.replace(".", ".0.", 1)
    modified_weights[new_layer_name] = weights

# print("----------------------------------------------------")
# for layer_name, weights in modified_weights.items():
#     print('layer_name: ', layer_name)

test_model.load_state_dict(modified_weights)
test_model.eval()


def partial_inference(img, layer=22):
    partition_layer = layer
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)
    with torch.no_grad():
        print(f"Starting VGG-16 inference")
        middle_res = test_model(batch_t, start_layer=0, stop_layer=partition_layer-1)
        final_res = partial_inference_server_side(middle_res, partition_layer)
    return final_res


out_1 = partial_inference(Image.open("images/blueangels.jpg"), 8)
out_2 = partial_inference(Image.open("images/blueangels.jpg"))
print(out_2 == out_1)

