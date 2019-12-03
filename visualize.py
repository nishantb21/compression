import torch
import torchvision
import numpy as np 
from matplotlib import pyplot as plt 
from models import Net

def main():
    global output

    # Load the dataset loader 
    test_loader = torchvision.datasets.MNIST('./files/', train=False, download=True)
    first_image = test_loader[0][0]
    first_image = np.asarray(first_image)

    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(first_image)
    axarr[0].title.set_text("Original Image")

    # Normalize the image for loading into the networks
    normalized_image = torchvision.transforms.Normalize((0.1307,), (0.3081,))(torchvision.transforms.ToTensor()(first_image))
    squeezed = torch.squeeze(normalized_image)

    axarr[1].imshow(squeezed, cmap="hot", interpolation="nearest")
    axarr[1].title.set_text("Normalized Image")

    # Expand to get the batch size dimension 
    x = torch.unsqueeze(normalized_image, dim=0)

    output = None
    def hook(self, input, out):
        global output
        output = out

    mdl = Net(28, 28, 1, 10)
    mdl.load_state_dict(torch.load("save/train_0.992_val_0.976_20.pt"))
    mdl.eval()
    mdl.conv_2.register_forward_hook(hook)
    mdl(x)

    with torch.no_grad():
        data = torch.squeeze(output).sum(dim=0).numpy()
    
    axarr[2].imshow(data, cmap="hot", interpolation="nearest")
    axarr[2].title.set_text("Activation Map(Non Pruned)")

    mdl = torch.load("final_full.pt")
    mdl = mdl.to("cpu")
    mdl.eval()
    mdl.conv_2.register_forward_hook(hook)
    mdl(x)

    with torch.no_grad():
        data = torch.squeeze(output).sum(dim=0).numpy()
    
    axarr[3].imshow(data, cmap="hot", interpolation="nearest")
    axarr[3].title.set_text("Activation Map(Pruned)")

    plt.show()

if __name__ == "__main__":
    main()