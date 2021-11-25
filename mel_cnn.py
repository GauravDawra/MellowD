
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.optim as optim

import numpy as np

batch_size = 4
num_epochs = 100
learning_rate = 0.001
PATH_TO_SAVE = "./mel_cnn.mdl"

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

images = np.load('./data/genres/mel_data/numerical_data/images.npy')
labels = np.load('./data/genres/mel_data/numerical_data/labels.npy')



# 80-20 train test split
# images_train, images_test = images[:800], images[800:]
# labels_train, labels_test = labels[:800], labels[800:]

images_train = np.zeros(shape=(800, 3, 100, 150))
images_test = np.zeros(shape=(200, 3, 100, 150))
labels_train = np.zeros(shape=(800))
labels_test = np.zeros(shape=(200))
citr = cite = cltr = clte = 0

for i, c in enumerate(classes):
    sub_perm = i*100 + np.random.RandomState(seed=20).permutation(100)
    for z in sub_perm[:80]:
        images_train[citr] = images[z]
        labels_train[cltr] = labels[z]
        citr += 1
        cltr += 1
    for z in sub_perm[80:]:
        images_test[cite] = images[z]
        labels_test[clte] = labels[z]
        cite += 1
        clte += 1


# yo = 0
# for i, l in enumerate(labels_train):
#     if l == 9: yo += 1

# print(yo)

images = torch.from_numpy(images).float()
labels = torch.from_numpy(labels).long()

images_train = torch.from_numpy(images_train).float()
images_test = torch.from_numpy(images_test).float()
labels_train = torch.from_numpy(labels_train).long()
labels_test = torch.from_numpy(labels_test).long()

perm = np.random.RandomState(seed=42).permutation(len(images_train))
images_train, labels_train = images_train[perm], labels_train[perm]

print(labels_train[:5])
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(3, 10, 3)
        # # self.pool =
        # self.fc1 = nn.Linear(10 * 5 * 5, 100)
        # self.fc2 = nn.Linear(100, 70)
        # self.fc3 = nn.Linear(70, 10)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 23 * 35, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 80)
        self.fc4 = nn.Linear(80, 10)

        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 4)
        # self.pool2 = nn.MaxPool2d(5, 5)
        # self.fc1 = nn.Linear(16 * 9 * 14, 240)
        # self.fc2 = nn.Linear(240, 120)
        # self.fc3 = nn.Linear(120, 80)
        # self.fc4 = nn.Linear(80, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # x = x.reshape(nxt.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# conv = nn.Conv2d(3, 6, 5)
# pool = nn.MaxPool2d(2, 2)
# nxt = conv(images)
# # print(nxt.shape)
# print("first", nxt.shape)
# nxt = pool(nxt)
# print("first after pool", nxt.shape)
# conv2 = nn.Conv2d(6, 16, 3)
# nxt = conv2(nxt)
# pool5 = nn.MaxPool2d(2, 2)
# print("second", nxt.shape)
# nxt = pool5(nxt)
# print("second after pool", nxt.shape)
# nxt = torch.flatten(nxt, 1)
# print(nxt.shape)

cnn = CNN()
print("load from saved model [y/n]?")
ans = input()
if ans == 'y':
    cnn.load_state_dict(torch.load(PATH_TO_SAVE))
    cnn.eval()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)

# output = cnn(images[:4])
# print(output.shape)

running_loss_plot = []
iteration_plot = []

# def get_test_loss(cnn):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for 

for epoch in range(num_epochs):
    running_loss = 0.0
    i = 0
    for batch_start in range(0, len(images_train), batch_size):
        batch_images, batch_labels = images_train[batch_start : batch_start + batch_size], labels_train[batch_start : batch_start + batch_size]
        # print(batch_images.dtype, batch_images.shape, batch_labels.shape)
        optimizer.zero_grad()
        outputs = cnn(batch_images)
        # print(outputs.shape)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i%50 == 49:
            print(f"EPOCH {epoch + 1}: step: {i + 1}, loss: {running_loss / 50}")
            running_loss_plot.append(running_loss)
            # iteration_plot.append(i + 1)
            running_loss = 0.0
        i += 1
    perm = np.random.permutation(len(labels_train))
    images_train = images_train[perm]
    labels_train = labels_train[perm]

import matplotlib.pyplot as plt

plt.plot(running_loss_plot)
plt.title('train loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('./train_loss_mel_cnn.jpg')
plt.show()

correct = 0
total = 0
with torch.no_grad():
    # for (images, labels) in test_loader:
    outputs = cnn(images_test)
    _, predicted = torch.max(outputs.data, 1)
    total += labels_test.size(0)
    correct += (predicted == labels_test).sum().item()

print('Accuracy of the network on the test images: %f %%' % (
    100 * (correct / total)))

print("Do you want to save this model [y/n]?")
ans = input()
if ans == 'y':
    torch.save(cnn.state_dict(), PATH_TO_SAVE)

correct_pred = {classname: 0 for classname in range(10)}
total_pred = {classname: 0 for classname in range(10)}

with torch.no_grad():
    # for images, labels in test_loader:    
    outputs = cnn(images_test)    
    _, predictions = torch.max(outputs, 1)
    # collect the correct predictions for each class
    for label, prediction in zip(labels_test, predictions):
        if label == prediction:
            correct_pred[label.item()] += 1
        total_pred[label.item()] += 1

    label_pred = {}
    for k in range(10):
        label_pred[classes[k]] = 100*(correct_pred[k] / total_pred[k])
    names = list(label_pred.keys())
    values = list(label_pred.values())

    plt.bar(range(len(label_pred)), values, tick_label=names)
    plt.title('Genre-wise percentage accuracy')
    plt.savefig('./genre_wise_accuracy_prediction')
    plt.show()