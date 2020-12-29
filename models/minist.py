import argparse       # Python 命令行解析工具
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from AAConv2d import AAConv2d
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,)
        self.cl = AAConv2d(1,  32, 3,  
                            12, 12, 4,
                            16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(10816, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.cl(x)
        #x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # torch.Size([64, 64, 13, 13]) -> torch.Size([64, 10816])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    # 如果模型中有Batch Normalization和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
    # Batch Normalization在train时不仅使用了当前batch的均值和方差，也使用了历史batch统计上的均值和方差，
    # 并做一个加权平均 （momentum参数）。在test时，由于此时batchsize不一定一致，因此不再使用当前batch的
    # 均值和方差，仅使用历史训练时的统计值。
    # Dropout在train时随机选择神经元而predict要使用全部神经元并且要乘一个补偿系数
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        """
        pytorch中CrossEntropyLoss是通过两个步骤计算出来的:
               第一步是计算log softmax，第二步是计算cross entropy（或者说是negative log likehood），
               CrossEntropyLoss不需要在网络的最后一层添加softmax和log层，直接输出全连接层即可。
               而NLLLoss则需要在定义网络的时候在最后一层添加log_softmax层(softmax和log层)
        总而言之：CrossEntropyLoss() = log_softmax() + NLLLoss() 
nn.CrossEntropyLoss()
        """
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train_Epoch:{} [{}/{} ({:.2f}%)] \t loss:{:.6f}'.format(epoch, 
                                                                   batch_idx*len(data), len(train_loader),
                                                                   100.0*batch_idx/len(train_loader),
                                                                   loss.item()))
    
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  #
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 默认情况下size_average=False，是mini-batchloss的平均值，然而，如果size_average=False，则是mini-batchloss的总和。
            test_loss += F.nll_loss(output, target,  reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, 
                                                            correct, 
                                                            len(test_loader.dataset),
                                                            100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('-gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('-no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('-seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args(args=[])

    torch.manual_seed(args.seed)  #  #为CPU设置种子用于生成随机数，以使得结果是确定的
    # torch.cuda.manual_seed(args.seed)为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    
    kwargs = {'batch_size': args.batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)
        
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform,)                                         
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    # 训练完成，保存状态字典到linear.pkl
    # torch.save(model.state_dict(), './linear.pkl')
    # model.load_state_dict(torch.load('linear.pth'))
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        
if __name__ == '__main__':
    main()