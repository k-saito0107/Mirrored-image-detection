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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    logs = []

    for epoch in range(1, num_epochs+1):
        if epoch == 1:
            if os.path.exists('/kw_resources/Mirrored-image-detection/weights/model.pth'):
                checkpoint = torch.load('/kw_resources/Mirrored-image-detection/weights/model.pth')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                logs = checkpoint['logs']
            
            model.to(device)
            model = nn.DataParallel(model)
            cudnn.benchmark = True
            

        for i, data in enumerate(train_loader, 0):
            img, label = data
            model.train()
            img , label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            t_loss = criterion(outputs, label)
            t_loss.backward()
            optimizer.step()
        
        if epoch%10 == 0:
            if epoch == 0:
                continue
            print('Epoch {}/{}'.format(epoch,num_epochs))
            correct = 0 #正解したデータの総数
            total = 0 #予測したデータの総数
            running_loss = 0.0
            model.eval()
            for i, v_data in enumerate(test_loader, 0):
                v_img, v_label = v_data
                v_img , v_label = v_img.to(device), v_label.to(device)
                v_outputs = model(v_img)
                v_loss=criterion(v_outputs,v_label)
                running_loss += v_loss.item()
                _, predicted = torch.max(v_outputs.data, 1)
                total += v_label.size(0)
                # 予測したデータ数を加算
                correct += (predicted == v_label).sum().item()
                #correct += torch.sum(predicted==v_label.data)
            val_acc=correct/total
            val_loss = running_loss/len(test_loader)
            train_loss = t_loss.to('cpu')
            print('train_loss : {},  val_loss : {},  val_acc : {}'.format(train_loss, val_loss, val_acc))

            #ログを保存
            log_epoch = {'epoch' : epoch, 'train_loss' : train_loss, 'val_loss' : val_loss,'val_acc' : val_acc}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('/kw_resources/Mirrored-image-detection/log_out.csv')
        
        if epoch % 10 == 0 and epoch != 0:
            print('---------------------------------------------------------------')
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.module.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':t_loss,
                'logs':logs
            },'/kw_resources/Mirrored-image-detection/weights/model.pth')
            
        
    
    return model





