import torch
import torchvision
from tqdm import tqdm
from models import Net

def main(batch_size, classes, epochs):
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size, shuffle=True)
    mdl = Net(28, 28, 1, classes)
    mdl.to("cuda:0")
    optimizer = torch.optim.Adam(mdl.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for i in range(epochs):
        print(i + 1, "/", epochs)
        mdl.train()
        obj = tqdm(train_loader)
        total = 0
        correct  = 0
        for x, y in obj:
            x = x.to("cuda:0")
            y = y.to("cuda:0")
            total += batch_size
            optimizer.zero_grad()
            output = mdl(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            correct += int(torch.sum(torch.argmax(output, 1) == y))
            accuracy = correct / total
            obj.set_description(desc="TRAINING Loss: {0:.3f} Accuracy: {1:.3f}".format(loss, accuracy))

        training_accuracy = accuracy
        mdl.eval()
        obj = tqdm(test_loader)
        total = 0
        correct  = 0
        for x, y in obj:
            x = x.to("cuda:0")
            y = y.to("cuda:0")
            total += batch_size
            output = mdl(x)
            correct += int(torch.sum(torch.argmax(output, 1) == y))
            loss = criterion(output, y)
            accuracy = correct / total
            obj.set_description(desc="TESTING Loss: {0:.3f} Accuracy: {1:.3f}".format(loss, accuracy))

        validation_accuracy = accuracy
        torch.save(mdl.state_dict(), "./save/train_{0:.3f}_val_{1:0.3f}_{2:02d}.pt".format(training_accuracy, validation_accuracy, i + 1))

if __name__ == "__main__":
    batch_size = 32
    classes = 10
    epochs = 20
    main(batch_size, classes, epochs)