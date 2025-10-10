import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from itertools import product


parameters = dict(
    batch_sizes = [1000]
    ,learning_rates = [0.001]
    ,num_workers = [0] 
)
param_values = [p for p in parameters.values()]


for batch_size, lr, num_workers in product(*param_values):
    tb = SummaryWriter() 
    network = Network()
    train_loader = torch.utils.data.DataLoader(
        train_set
        ,batch_size=batch_size
        ,num_workers=num_workers
        ,shuffle=True
    )
    optimizer = optim.Adam(network.parameters(), lr=lr) 

    for epoch in range(30):
        total_loss = 0
        total_correct = 0
        for batch in train_loader:
            images = batch[0]
            labels = batch[1]
                
            predictions = network(images)
            loss = F.cross_entropy(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        
            total_loss += loss.item() * batch_size
            total_correct += predictions.argmax(dim=1).eq(labels).sum().item()
    
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Correct Amount', total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

    tb.close()
