import torch
from dataloader import MRDataset
from model import Net
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), self.margin - euclidean_distance])), 2) * 0.5)
        return loss_contrastive
    
def train(model, train_dataset, epochs, criterion, model_name, indexes):
    device='cuda'
    losses = []
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    for epoch in range(epochs):
        model.train()
        print("Starting epoch " + str(epoch+1))
        np.random.seed(epoch)
        np.random.shuffle(indexes)
        for index in indexes:
            img1, img2, labels = train_dataset.__getitem__(index)
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            output1 = model.forward(img1)
            output2 = model.forward(img2)
            loss = criterion(output1,output2,labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())

    torch.save(model, './outputs/' + model_name)
    pd.DataFrame(losses).to_csv('./outputs/losses_model_name_{}'.format(model_name))
        
    print("Finished Training")  


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_arguments()
    model_name = args.model_name
    epochs = args.epochs
    data_path = args.data_path
    if args.index != []:
        indexes = [int(item) for item in args.index.split(', ')]
    else:
        meta = pd.read_csv('metadata.csv')
        indexes = list(meta.loc[meta['ref_set']==1, 'id'])
          
    train_dataset = MRDataset(data_path, 'sagittal', indexes, transform=None, train=True)
    model = Net()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    criterion = ContrastiveLoss()
    train(model, train_dataset, epochs, criterion, model_name, indexes)





