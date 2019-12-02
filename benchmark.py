from models import Net
import time
import torch
import numpy as np
from tqdm import tqdm
from pthflops import count_ops

def evaluate(model, input_tensor):
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    inference_time = []
    flops = count_ops(model, input_tensor)

    for i in tqdm(range(100)):
        t1 = time.time()
        b = model(input_tensor)
        t2 = time.time()
        inference_time.append(t2 - t1)

    avg = sum(inference_time) / len(inference_time)
    fps = 1 / avg
    print("FPS:", fps)
    print("FLOPS:", flops)

def main(path):
    mdl = Net(28, 28, 1, 10)
    mdl.load_state_dict(torch.load(path))
    a = torch.ones((1, 1, 28, 28))
    evaluate(mdl, a)


if __name__ == "__main__":
    main("./save/train_0.992_val_0.976_20.pt")