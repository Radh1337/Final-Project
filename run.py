import numpy as np
import torch
import torch.nn as nn
import argparse
import time
from util import load_data_n_model
from collections import defaultdict

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    inference_start = time.time()
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)

        for label, prediction in zip(labels, predict_y):
            class_total[label.item()] += 1
            if label == prediction:
                class_correct[label.item()] += 1

        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    total_inference_time = time.time() - inference_start
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    print("\nPer-class accuracy:")
    for class_id in sorted(class_total.keys()):
        acc = class_correct[class_id] / class_total[class_id] if class_total[class_id] > 0 else 0.0
        print(f"  Class {class_id}: {acc:.4f} ({class_correct[class_id]}/{class_total[class_id]})")
    return

    
def main():
    root = '/content/gdrive/MyDrive/Data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar', 'MyDataset'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT', 'SNN'])
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model=model,
        tensor_loader= train_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
         )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )
    return


if __name__ == "__main__":
    main()
