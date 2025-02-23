import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

model = nn.Sequential(
        nn.Linear(784,15),
        nn.ReLU(),
        nn.Linear(15,10)
)

transform = transforms.ToTensor()

train_dataset = MNIST(root="/data",train=True,transform=transform,download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MNIST(root="/data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

epochs = 100

otimizador = optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
rewards = []
epocas = range(epochs)

##------------------TREINAMENTO-------------------------##
for epoch in range(epochs):
    corretos = 0
    total = 0
    for image, label in train_loader:
        imagem_flattened = image.view(image.size(0), -1)
        otimizador.zero_grad()
        output = model(imagem_flattened)
        loss = criterion(output,label)
        loss.backward()
        otimizador.step()
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        corretos += (predicted == label).sum().item()
    rewards.append(corretos)   
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print(f"Total : {total} Corretos : {corretos}")
acuracia_treinamento = (corretos/total)*100
print(f"Acuracia do treinamento final: {acuracia_treinamento:.2f} ")
##-----------------------------------------------------##

##--------------------------EVAL-----------------------##
model.eval()  
corretos_test = 0
total_test = 0
with torch.no_grad():  
 for image, label in test_loader:
    
    imagem_flattened = image.view(image.size(0), -1)
    output = model(imagem_flattened)
    _, predicted = torch.max(output, 1)
    total_test += label.size(0)
    corretos_test += (predicted == label).sum().item()

##--------------------------EVAL-----------------------##    
acuracia_validacao = (corretos_test / total_test) * 100
print(f"Acuracia no conjunto de teste após a época {epoch+1}: {acuracia_validacao:.2f}%")

sns.lineplot(x=epocas,y=rewards)
plt.title("Recompensa por Epoca Treinamento")
plt.xlabel("Epoca")
plt.ylabel("Recompensa")
plt.show()