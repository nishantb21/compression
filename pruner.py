import torch
from models import get_convolution

def l2_norm(matrix):
    with torch.no_grad():
        x = torch.mul(matrix, matrix)
        x = x.sum(dim=(1, 2, 3))
        x = torch.sqrt(x)
        return x

def get_least_index(weights_matrix):
    with torch.no_grad():
        x = l2_norm(weights_matrix)
        return torch.argmin(x)

def layer_1_pruner(model):
    with torch.no_grad():
        parameters = list(model.parameters())
        init_weights = []
        init_weights.append(parameters[0])
        init_weights.append(parameters[1])
        index_to_remove = get_least_index(init_weights[0])
        
        new_weights = []
        new_weights.append(torch.cat((init_weights[0][0:index_to_remove], init_weights[0][index_to_remove + 1:]), 0))
        new_weights.append(torch.cat((init_weights[1][:, 0:index_to_remove], init_weights[1][:, index_to_remove + 1:]), 1))


        channels_1 = model._modules["conv_1"]._modules["0"].out_channels - 1
        channels_2 = model._modules["conv_2"]._modules["0"].out_channels

        replacement_conv = get_convolution(1, channels_1, 3, 1, 1)
        model._modules["conv_1"] = replacement_conv
        replacement_conv = get_convolution(channels_1, channels_2, 3, 1, 1)
        model._modules["conv_2"] = replacement_conv

        counter = 0
        for p in model.parameters():
            p.data = new_weights[counter].data
            counter += 1
            if counter == 2:
                break

        return model

def layer_2_pruner(model):
    with torch.no_grad():
        channels_1 = model._modules["conv_1"]._modules["0"].out_channels
        channels_2 = model._modules["conv_2"]._modules["0"].out_channels

        parameters = list(model.parameters())
        init_weights = []
        init_weights.append(parameters[1])
        init_weights.append(parameters[2])
        index_to_remove = get_least_index(init_weights[0])
    
        new_weights = []
        new_weights.append(torch.cat((init_weights[0][0:index_to_remove], init_weights[0][index_to_remove + 1:]), 0))
        x = init_weights[1]
        x = x.reshape((x.shape[0], channels_2, -1))
        x = torch.cat((x[:, 0:index_to_remove], x[:, index_to_remove + 1:]), 1)
        x = x.reshape((x.shape[0], -1))
        new_weights.append(x)

        replacement_conv = get_convolution(channels_1, channels_2 - 1, 3, 1, 1)
        model._modules["conv_2"] = replacement_conv
        replacement_dense = torch.nn.Linear(x.shape[1], x.shape[0])
        model._modules["dense_1"] = replacement_dense

        counter = 0
        inner_counter = 0
        for p in model.parameters():
            if counter == 1 and counter ==2:
                p.data = new_weights[inner_counter]
                inner_counter += 1

            counter += 1

        return model