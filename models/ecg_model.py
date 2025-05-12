import torch.nn as nn
import torch.nn.functional as F


class SE_Attention(nn.Module):
    def __init__(self):
        super(SE_Attention, self).__init__()
        self.mlp1 = nn.Linear(512, 64)
        self.mlp2 = nn.Linear(64, 512)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        att_weights = F.sigmoid(self.mlp2(x))
        
        return att_weights

class CNN1D_Att(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D_Att, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.maxpool1 = nn.MaxPool1d(2, stride=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool1d(2, stride=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool1d(2, stride=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=0.2)

        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.adaptpool = nn.AdaptiveAvgPool1d(1)

        self.att_block = SE_Attention()

        self.final_mlp = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(p=0.5)
        self.output = nn.Linear(256, num_classes)

          
    def forward(self, x):
        x = self.dropout1(self.maxpool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.maxpool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.maxpool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.adaptpool(F.relu(self.bn4(self.conv4(x))))

        reshaped_x = x.squeeze(-1)
        # print('x:', x.shape)

        att_weights = self.att_block(reshaped_x)
        x_att = reshaped_x + att_weights
        classes = self.output(self.dropout5(F.relu(self.final_mlp(x_att))))
        
        return classes
