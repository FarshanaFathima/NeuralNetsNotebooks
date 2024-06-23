
from create_train_test import Dataset
from model import MyRNN
import torch
from torch import nn
path = r"/Users/farshanafathima/Documents/playground/NeuralNetsNotebooks/data/names"
dt_obj = Dataset(path)
dt_obj.create_dataset()
train , test = dt_obj.split_data(dt_obj.all_names, dt_obj.all_labels)
print(len(dt_obj.all_labels), len(dt_obj.all_names))


criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
vocab = dt_obj.ex_obj.vocabulary
labels = dt_obj.lang2label
n_hidden = 256
model_obj = MyRNN(len(vocab), n_hidden, len(labels))
optimizer = torch.optim.Adam(model_obj.parameters(), lr=learning_rate)
print(len(vocab), n_hidden, len(labels))

epochs = 2
print_interval = 3000
loss_graph = []


hidden = model_obj.initHidden()
for epoch in range(epochs):
    for i, (x, y) in enumerate(train):
        for char in x:
            output, hidden = model_obj(char, hidden)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_obj.parameters(), 1)
        optimizer.step()

        if (i + 1) % print_interval == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], ", f"Step [{i + 1}/{len(train)}], ",f"Loss: {loss.item():.4f},")
                loss_graph.append(loss.item())
