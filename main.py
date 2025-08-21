'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import csv

import os
import argparse

import numpy as np
import random

from models import *
# from utils import progress_bar
from lr_sched import LineSearchScheduler
import time


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--min_lr', default=1e-6, type=float, help='min learning rate')
parser.add_argument('--max_lr', default=1.0, type=float, help='max learning rate')
parser.add_argument('--warmup_epochs', default=0, type=int, help='warmup epochs')
parser.add_argument('--epoch', default=100, type=int, help='epochs')
parser.add_argument('--optimizer', default="AdamW", type=str, help='optimizer')
parser.add_argument('--scheduler', default="LineSearch", type=str, help='scheduler')
parser.add_argument('--condition', default="armijo", type=str, help='condition')
parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
parser.add_argument('--c1', default=1e-4, type=float, help='c1')
parser.add_argument('--c2', default=0.9, type=float, help='c2')
parser.add_argument('--save_dir', default="./", type=str, help='save_dir')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
save_name = f"{args.scheduler}_{args.batch_size}_{args.optimizer}_{args.condition}"





def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


batch_size = args.batch_size
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()

if args.optimizer == "AdamW":
    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),  
        weight_decay=5e-4
    )
elif args.optimizer == "SGD": 
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

elif args.optimizer == "Adam":
    optimizer = optim.Adam(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999)
    )
else:
    print("Optimizer Not Implemented")




if args.scheduler == "LineSearch":
    scheduler = LineSearchScheduler(optimizer, args.condition, args.min_lr, args.max_lr, args.warmup_epochs, net, args.lr)
elif args.scheduler == "Cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
else:
    print("Scheduler Not Implemented")



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(f'./checkpoint/{save_name}_ckpt.pth', weights_only=False)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    missing, unexpected = net.load_state_dict(checkpoint['net'], strict=False)
    print("[Model] missing keys:", missing)
    print("[Model] unexpected keys:", unexpected)

    # optimizer
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("[Optimizer] state loaded successfully")
    except Exception as e:
        print("[Optimizer] load failed:", e)

    # scheduler
    try:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("[Scheduler] state loaded successfully")
    except Exception as e:
        print("[Scheduler] load failed:", e)

    start_epoch = checkpoint['epoch']




def allreduce_loss(loss):
    world_size = dist.get_world_size()
    with torch.no_grad():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= world_size
    return loss


# Training
def train(epoch):
    print("\033[92m" + '\nEpoch: %d' % epoch + "\033[0m")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start = 0
    ref_loss = 10000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print("\033[92m" + f"start batch {batch_idx}"+ "\033[0m")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if start == 0:
            print("\033[92m" + f"gradients at batch {batch_idx} stored in parameters" + "\033[0m")
        
        if isinstance(scheduler, LineSearchScheduler):
            if start == 0 or loss > 1.5 * ref_loss:
                def closure():
                    outputs = net(inputs)
                    loss_val = criterion(outputs, targets)
                    # loss_val = allreduce_loss(loss_val)
                    return loss_val


                gk = torch.cat([p.grad.view(-1) for p in net.parameters() if p.grad is not None]).detach().cpu().numpy()
                ref_loss = scheduler.step(loss=loss, gk=gk, epoch=epoch, loss_fn=closure, c1=args.c1, c2=args.c2)
            optimizer.step()
        else:
            optimizer.step()
            scheduler.step()
        print("\033[92m" + f"loss at batch {batch_idx}: {loss}" + "\033[0m")

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = correct / total
        start = 1
        lr = optimizer.param_groups[0]['lr']
    return lr, train_loss / len(trainloader), acc

    




def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = correct / total
        return test_loss / len(testloader), acc

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc





log_path = os.path.join(args.save_dir, f"{save_name}_log.csv")

if not os.path.exists(log_path):
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "time", "total"])



total_time = 0.0

for epoch in range(start_epoch, start_epoch + args.epoch):
    epoch_start = time.perf_counter()
    lr, train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    epoch_end = time.perf_counter() 
    epoch_time = epoch_end - epoch_start
    total_time += epoch_time

    print(f"Epoch {epoch} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}",
          f"Time: {epoch_time:.2f}s | Total: {total_time/60:.2f}min")

    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, lr, train_loss, train_acc, test_loss, test_acc])

    checkpoint = {
        "epoch": epoch,
        "net": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }
    torch.save(checkpoint, os.path.join(args.save_dir, f"{save_name}_ckpt.pth"))

