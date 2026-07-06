import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader #This is for loading data and creating mini batches

train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transforms.ToTensor())
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transforms.ToTensor())

class Network(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.layer1 = nn.Linear(784, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x is the input.
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

#instantiate the model
model = Network()
#make a loss function and tell it what to change and by how much. v' = v - lr(gradient) //so somewhere down the line, we'll need to find the average gradient and apply this.
optimizer = optim.SGD(model.parameters(), lr = 0.1)
loss_fn = nn.CrossEntropyLoss() #Now keep in mind that this is an instanciation for another class as well. So, loss_fn in itself keeps a lot of methods.

#this is the function that helps us to provide a collection of 32 images that collectively
# get their gradient and average loss found out.
train_loader = DataLoader(dataset = train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = 256, shuffle = True)

for epoch in range(10):
    count = 0
    for x, y in train_loader: #here, x is a vector containing 32 images, and y is another vector containing 32 results.
        x = x.view(x.size(0), -1)
        #it's important to write x.size(0) because that is the size of the number of images in this batch.
        # We don't hardcode 32 because if the MNIST number isn't directly divisible by 32, then we have maybe 10, or 12 numbers left in the last batch.
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        predicted_vecs = torch.argmax(prediction, -1)
        count += (predicted_vecs == y).sum().item()
    print(f"Epoch {epoch+1} accuracy: {count/60000*100:.1f}%")

#we take batch size of > 1 in testing because we can analyze multiple images in our
#model parallely. Our model is capable of that. 
model.eval()
test_count = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.view(x.size(0), -1)
        prediction = model(x)
        predicted_vecs = torch.argmax(prediction, -1)
        test_count += (predicted_vecs == y).sum().item()

print(f"Test Data Accuracy: {test_count/10000*100:.1f}%")
