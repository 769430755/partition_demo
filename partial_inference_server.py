import torch
import json
import collections
import yaml
from torchvision import transforms
from dnn_architectures.vgg import VGGNet
from PIL import Image


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


def partial_inference_server_side(out, start_layer):
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

    test_model.load_state_dict(modified_weights)
    test_model.eval()

    with torch.no_grad():
        print(f"internal VGG-16 inference")
        out = test_model(out, start_layer=start_layer, stop_layer=22)
    return out


