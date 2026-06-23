import torch
import time
import datetime
import math
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import configs.configs as allcfg

current_model = allcfg.current_model
num_classes = allcfg.classes

def adjust_lr_sub(lr_init, lr_gamma, epoch, step_index):
    # Adjust the learning rate in stages
    if epoch < 1:
        lr = 0.0001 * lr_init
    elif epoch <= step_index[0]:
        lr = lr_init
    elif epoch <= step_index[1]:
        lr = lr_init * lr_gamma
    elif epoch > step_index[1]:
        lr = lr_init * lr_gamma ** 2
    else:
        raise ValueError('step_index do not match adjust_lr_sub function!')
    return lr

def myvalidate(val_batch_data, model, loss_fun, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    correct = 0
    total = 0
    class_loss = torch.zeros(num_classes).to(device)  # Store total loss for each class
    class_count = torch.zeros(num_classes).to(device)  # Store count of samples for each class

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (img1, img2, abu, target, indices) in enumerate(val_batch_data):
            img1 = img1.to(device)
            img2 = img2.to(device)
            abu = abu.to(device)
            target = target.to(device)

            # 设置掩膜，忽略背景值 0
            if allcfg.current_dataset in allcfg.mask0data:
                mask = (target != 0)
                target = target[mask]
                prediction = model(img1, img2, abu)[mask]  # 应用掩膜以忽略背景
            else:
                prediction = model(img1, img2, abu)  # Perform prediction without mask
        
            if allcfg.current_model == 'ML_EDAN':
                prediction, reloss1, reloss2 = model(img1, img2, abu)
                loss = loss_fun(prediction, target.long()) + 0.5* reloss1 + 0.5* reloss2
            else:
                loss = loss_fun(prediction, target.long())
            val_loss += loss.item()  # Sum up batch loss
            predict_label = prediction.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += predict_label.eq(target.view_as(predict_label)).sum().item()
            total += target.size(0)

            for cls in range(num_classes):
                cls_mask = (target == cls)  # Find samples of the current class
                if cls_mask.sum() > 0:  # Only update if there are samples of this class
                    class_loss_val = loss_fun(prediction[cls_mask], target[cls_mask].long())
                    if allcfg.avg_valloss:
                        # For unweighted class average loss
                        class_loss[cls] = class_loss_val.mean().item()  # Average loss for this class
                        class_count[cls] = 1  # Mark that this class has samples 
                    elif allcfg.w_avg_valloss:
                        # For weighted class average loss
                        class_loss[cls] += class_loss_val.sum().item()  # Sum up loss for this class
                        class_count[cls] += cls_mask.sum().item()  # Count samples for this class

                # Calculate the epoch loss based on the selected mode
                if allcfg.avg_valloss:
                    # Unweighted class average loss
                    class_avg_loss = class_loss.sum() / class_count.sum()
                    val_loss = class_avg_loss.item()
                elif allcfg.w_avg_valloss:
                    # Weighted class average loss
                    class_avg_loss = class_loss / class_count  # Avoid dividing by zero
                    class_avg_loss[torch.isnan(class_avg_loss)] = 0  # Handle classes with no samples
                    val_loss = class_avg_loss.mean().item()

    val_accuracy = 100. * correct / total  # Calculate accuracy
    return val_loss, val_accuracy

def train(train_data, val_data, model, loss_fun, optimizer, scheduler, device, cfg):
    torch.autograd.set_detect_anomaly(True)
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']
    save_folder = cfg['save_folder']
    save_name = cfg['save_name']
    current_dataset = cfg['current_dataset']
    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']
    train_ratio = str(cfg['train_ratio'])
    epoch_size = cfg['epoch']
    batch_size = cfg['batch_size']

    # gpu_num
    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    '''# Load the model and start training'''
    model.train()

    if cfg['reuse_model'] == True:
        checkpoint = torch.load(cfg['reuse_file'], map_location=device)
        start_epoch = checkpoint['epoch']
        if checkpoint['type']=='best':
            print('Reloading best model...')
            return
        elif checkpoint['type']=='final':
            print('Reloading final model...')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    else:
        start_epoch = 0
        print('start training...')

    batch_num = math.ceil(len(train_data) / batch_size)
    train_loss_save = []
    train_acc_save = []
    val_loss_save = []
    val_acc_save = []
    tol_time = 0
    bestloss = 99999
    bestacc = 0
    save_epoch_idx = 0

    print("Total parameters in " , current_model, " is ", sum(x.numel() for x in model.parameters()))

    for epoch in range(start_epoch + 1, epoch_size + 1):

        epoch_time0 = time.time()
        epoch_loss = 0
        predict_correct = 0
        label_num = 0

        # Load batch data
        batch_data = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)        # Adjust the learning rate
        val_batch_data = DataLoader(val_data, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        if lr_adjust:
            lr = adjust_lr_sub(lr_init, lr_gamma, epoch, lr_step)
        else:
            lr = lr_init

        class_loss = torch.zeros(num_classes).to(device)  # Store total loss for each class
        class_count = torch.zeros(num_classes).to(device)  # Store count of samples for each class
        for batch_idx, batch_sample in enumerate(batch_data):
            iteration = (epoch - 1) * batch_num + batch_idx + 1
            batch_time0 = time.time()

            img1, img2, abu, target, indices = batch_sample  # batch_sample:img1_pad, img2_pad, label, indices
            img1 = img1.to(device)
            img2 = img2.to(device)
            abu = abu.to(device)
            target = target.to(device)
            # Calculate unweighted class-averaged validation loss
            class_avg_loss = class_loss.sum() / class_count.sum()  # Sum up non-zero losses and divide by the number of classes with samples
            val_loss = class_avg_loss.item()

            prediction = model(img1, img2, abu)  # Perform prediction without mask
            loss = loss_fun(prediction, target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batch_time1 = time.time()
            batch_time = batch_time1 - batch_time0
            batch_eta = batch_time * (batch_num - batch_idx)
            epoch_eta = int(batch_time * (epoch_size - epoch) * batch_num + batch_eta)  # 预估剩余时间

            epoch_loss += loss.item()
            predict_label = prediction.detach().argmax(dim=1, keepdim=True)

            predict_correct += predict_label.eq(target.view_as(predict_label)).sum().item()
            label_num += len(target)

            for cls in range(num_classes):
                cls_mask = (target == cls)  # Find samples of the current class
                if cls_mask.sum() > 0:  # Only update if there are samples of this class
                    class_loss_val = loss_fun(prediction[cls_mask], target[cls_mask].long())
                    if allcfg.avg_valloss:
                        # For unweighted class average loss
                        class_loss[cls] = class_loss_val.mean().item()  # Average loss for this class
                        class_count[cls] = 1  # Mark that this class has samples 
                    elif allcfg.w_avg_valloss:
                        # For weighted class average loss
                        class_loss[cls] += class_loss_val.sum().item()  # Sum up loss for this class
                        class_count[cls] += cls_mask.sum().item()  # Count samples for this class

                # Calculate the epoch loss based on the selected mode
                if allcfg.avg_valloss:
                    # Unweighted class average loss
                    class_avg_loss = class_loss.sum() / class_count.sum()
                    epoch_loss = class_avg_loss.item()
                elif allcfg.w_avg_valloss:
                    # Weighted class average loss
                    class_avg_loss = class_loss / class_count  # Avoid dividing by zero
                    class_avg_loss[torch.isnan(class_avg_loss)] = 0  # Handle classes with no samples
                    epoch_loss = class_avg_loss.mean().item()

        train_acc = 100 * predict_correct/label_num
        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0
        tol_time = tol_time + epoch_time
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        val_loss, val_acc = myvalidate(val_batch_data, model, loss_fun, device)
        model.train()
        print('Epoch: {}/{} || lr: {} || Train loss: {:.4f} || Train acc: {:.2f}% || Val loss: {:.4f} || Val acc: {:.2f}% || '
              'Epoch time: {:.0f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss / batch_num, train_acc, val_loss / batch_num, val_acc,
                      epoch_time, str(datetime.timedelta(seconds=int(epoch_time * (epoch_size - epoch))))
                      )
              )

        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        train_loss_save.append(epoch_loss / batch_num)
        train_acc_save.append(train_acc)
        val_loss_save.append(val_loss / batch_num)  # 保存验证集loss
        val_acc_save.append(val_acc)  # 保存验证集accuracy

        if val_loss <= bestloss :
            bestacc = val_acc
            bestloss = val_loss
            save_epoch_idx = epoch
        # Store the best val_acc model
            save_model = dict(
                model=model.state_dict(),
                epoch=epoch,
                type = 'best',
            )
            torch.save(save_model, os.path.join(save_folder, save_name + '_best.pth'))

        # Store the final model every epoch
        save_model = dict(
            model=model.state_dict(),
            epoch=epoch,
            type = 'final',
        )
        torch.save(save_model, os.path.join(save_folder, save_name + '_Final.pth'))

    if cfg['reuse_model'] == False:
        print('Total training time: {:.4f}min'.format(tol_time / 60), ', save best_model at epoch ',save_epoch_idx)

        # 所有epoch完成后，绘制loss和accuracy曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch_size + 1), train_loss_save, label='Train Loss')
        plt.plot(range(1, epoch_size + 1), val_loss_save, label='Val Loss', linestyle='--')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch_size + 1), train_acc_save, label='Train Accuracy')
        plt.plot(range(1, epoch_size + 1), val_acc_save, label='Val Accuracy', linestyle='--')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./weights/' + current_model + '/'  + current_dataset + '/Train_Val_Loss&Acc_' + train_ratio + '.png')
        plt.close()  # 关闭当前图像