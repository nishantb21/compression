from models import Net
import time
import torch
import numpy as np
from tqdm import tqdm

def main(path):
    mdl = Net(28, 28, 1, 10)
    mdl.load_state_dict(torch.load(path))
    mdl.eval()

    mdl_parameters = filter(lambda p: p.requires_grad, mdl.parameters())
    params = sum([np.prod(p.size()) for p in mdl_parameters])

    inference_time = []
    a = torch.ones((1, 1, 28, 28))
    for i in tqdm(range(100)):
        t1 = time.time()
        b = mdl(a)
        t2 = time.time()
        inference_time.append(t2 - t1)
    avg = sum(inference_time) / len(inference_time)
    fps = 1 / avg
    print("FPS:", fps)
    print("PARAMETERS:", params )


if __name__ == "__main__":
    main("D:\\compression\\save\\train_0.991_val_0.979_17.pt")