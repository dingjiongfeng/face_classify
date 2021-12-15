from IPython.terminal.embed import embed
import torch
from util import device
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = torch.tensor(gamma).to(device)
        self.size_average = size_average

    def forward(self, output, target):
        '''
        output: tensor (batch_size, 2) 神经网络的输出
        target: tensor (batch_size, )
        '''
        # 经过sigmoid函数使得变量在(0,1)之间，选取大的值作为预测值
        # embed()
        output = torch.sigmoid(output)
        pred, _ = output.max(dim=1)  # size(batch_size, )
        pred = pred.view(-1, 1)
        pred = torch.cat([1-pred, pred], dim=1)  # size(batch_size, 2)
        class_mask = torch.zeros(pred.shape).to(device)
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)  # 将为真的对应列的数字设为1
        # RuntimeError: Index tensor must have the same number of dimensions as self tensor

        prob = (pred * class_mask).sum(dim=1).view(-1, 1)  # batch_size,1
        prob = prob.clamp_(min=0.0001, max=1.0)
        log_p = torch.log(prob)

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape).to(device)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)  # batch_size, 1
        batch_loss = -alpha * torch.pow((1-prob), self.gamma) * log_p
        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        '''
        pred : batch_size, label size
        target : batch_size, 1
        '''
        out = torch.log_softmax(pred, dim=1)
        # return nn.NLLLoss()(out, target)
        # NLLLoss的计算方式就是将上面输出的值与对应的Label中的类别拿出来去掉负号，求均值

        class_mask = torch.zeros(pred.shape).to(device)
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)
        batch_loss = -(class_mask * out).sum(dim=1).view(-1, 1)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


'''
x_input:
 tensor([[ 2.8883,  0.1760,  1.0774],
        [ 1.1216, -0.0562,  0.0660],
        [-1.3939, -0.0967,  0.5853]])
y_target
 tensor([1, 2, 0])

 crossentropyloss_output:
 tensor(2.3185)
'''

if __name__ == '__main__':
    # class_mask = torch.zeros((4, 2), dtype=torch.float32)
    # target = torch.FloatTensor([[1], [0], [0], [1]])
    # class_mask.scatter_(1, target.long(), 1.)
    # print(class_mask)

    # criterion = CrossEntropyLoss()
    # x_input = torch.tensor([[2.8883,  0.1760,  1.0774],
    #                         [1.1216, -0.0562,  0.0660],
    #                         [-1.3939, -0.0967,  0.5853]]).to(device)
    # y_target = torch.tensor([1, 2, 0]).to(device)
    # criterion1 = nn.CrossEntropyLoss()
    # print(criterion(x_input, y_target))
    # print('crossentropy:', criterion1(x_input, y_target))

    x_input = torch.tensor([[2.8883,  0.1760],
                            [1.1216, -0.0562],
                            [-1.3939, -0.0967]]).to(device)
    y_target = torch.tensor([1, 0, 0]).to(device)
    criterion = FocalLoss()
    print('FocalLoss:', criterion(x_input, y_target))
