import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os


def train(model, num_epochs,train_loader, test_loader):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    logs = []

    for epoch in range(1, num_epochs+1):
        if epoch == 1:
            if os.path.exists('/kw_resources/Mirrored-image-detection/weights/model.pth'):
                checkpoint = torch.load('/kw_resources/Mirrored-image-detection/weights/model.pth')
                checkpoint = checkpoint.to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                t_loss = checkpoint['loss']
                logs = checkpoint['logs']
            
            model.to(device)
            model = nn.DataParallel(model)
        running_loss = 0

        for data in train_loader:
            img = data
            model.train()
            img = img.to(device)
            mini_batch_size = img.size()[0]
            label = torch.full((mini_batch_size,),1.0).to(device)
            outputs = model(img)
            outputs = outputs.view(-1)
            print(outputs)
            t_loss = criterion(outputs, label)
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            running_loss += t_loss.item()
        
        train_loss = running_loss/len(train_loader)
        '''
        #validation
        if epoch%10 == 0:
            if epoch == 0:
                continue
            print('Epoch {}/{}'.format(epoch,num_epochs))
            running_loss = 0.0
            model.eval()
            for data in test_loader:
                img = data
                img = img.to(device)
                mini_batch_size = img.size()[0]
                label = torch.full((mini_batch_size,),1.0).to(device)
                v_outputs = model(img)
                v_loss=criterion(v_outputs,label)
                running_loss += v_loss.item()
                
            
            val_loss = running_loss/len(test_loader)
            train_loss = t_loss.to('cpu')
            print('train_loss : {},  val_loss : {}'.format(train_loss, val_loss))
            
            #ログを保存
            log_epoch = {'epoch' : epoch, 'train_loss' : train_loss, 'val_loss' : val_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('/kw_resources/Mirrored-image-detection/log_out.csv')
        '''
        if epoch % 10 == 0 and epoch != 0:
            print('---------------------------------------------------------------')
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.module.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':train_loss,
                'logs':logs
            },'/kw_resources/Mirrored-image-detection/weights/model.pth')

            #ログを保存
            print('epoch : {}, train_loss : {}'.format(epoch, train_loss))
            log_epoch = {'epoch' : epoch, 'train_loss' : train_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('/kw_resources/Mirrored-image-detection/log_out.csv')
            
        
    
    return model





