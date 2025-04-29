import torch.nn as nn
import torch.nn.functional as F

# https://pdf.sciencedirectassets.com/271505/1-s2.0-S0950705125X00058/1-s2.0-S095070512500348X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDMaCXVzLWVhc3QtMSJHMEUCIQCgKqhV2Glu9zSo5DndVv%2FaJnVtwi3nobC5fyLD6%2FxwwQIgV7rEiHr9OapPKkhGgER7rk06Rln7a1fJeuvLjS%2FzOyAqvAUIu%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDK3XwQl1Figi7u21ZCqQBQuFABv%2FjwGHRq%2BALghVL7cYNPf3qrZrfbBYZzbSlEbk1G3Qh%2Bm6psyCMRw1z4WYpjL9CvVN6PXKoQfp3oaWq3QyVgoJA5WjCuyFmfeceuo3TDSzT7PNMDMFKwYLyz6QBemQ9vbhf065WVfeT%2BfHPAHYC7pU%2FKWEXxI4OgQoLuntpyXgGgVPIytK2Cs3eZfcn2QweOxxkjm4eG%2BCS3J0Rfextc%2Bcrw4rwRx%2Fxi%2Fm2prxM3rfSmEq469vc7RX8ECTfPalpXRl5QF2CSInbjgMeGNcF0WfRU8TOAjRdLs4ugXTYPXCBHDnWMG%2FTC1xwbLxLNykERUNhAnvE25w4DL07QrjCAA6Z6KlEwDeXdJ9bIqvz7zIKoDU6m20%2BCd1gGOCVkxxq7qWVi3Hho8vqI0m1dpszYmvN%2FKSPnFF6DTo7EbxLpC8W7ApLFnumXs8cy0KG7NWinuOQT0%2BjhIJ%2FKCWuScpOb4zBIXICzQnmYhhlhPM2ehh7GWc2VztkqS5%2B%2FphQHUAE4x0i3jtxzUIxAlFsbPCFEjOt3xuFaBFbRHJd9oy9h4SqIFOxgcZspUd%2BrYY9HBi0YorHBGFUAVLclhH3KgeppNtj0WTPgnQL5HD4qxyvMDShKX6oPVgRbyktNo1ZO3pCJ1lFpVqM%2Bvf6Tqp2FTCd8Ve5GhAouwp2HLnUiKJG6u9l78JZGhVm3lJNdObyrOVN19%2FYuiVu7CvhFJ0149AD9Ch75QoAFf8VEeTXRvKBbzZXQFj1WP%2FNmiirlJGUcbxs15xOWt10FlNWZMIpnRmBgiKf3FqiYhsKKBKyQ0L9U2V4IhXn37op76IsCn6quBEHxdk6fSKhTJKXuYyf7s1%2BfV4G0D05PhRO%2FY0me%2BOMMW1mMAGOrEBQfRRGle%2B%2BQjtN1H7WGjcN0KkUY1wsbVsLRsAZaUOWh0pkg76GtG5KuaMhjMfVJlwePWBx5Xj%2FCUKBJhDfCvgr3Y4bvlxV8krqgifMuDHvJXlesh7cm82JrZ56B0hKt105vg%2BXrEd8c3MJnOn3mnBS2jjt%2B%2BK5YKypYd%2BiP5PXKoD%2FT5J%2FeyShem9DFrGjDrCCSbGYvFm9t3qRYrsr%2FYKlKCwYU14%2BsgMBE4jPUrUI%2B3n&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250421T112904Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY23IPBPIP%2F20250421%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cd7ffe1e81628e6ff7b2b788bbeadda1e486539e97c2fa7f0646fea91932c59c&hash=10c1e1a703ee3e80601ce86c9e97b14cabc57af8c2e7bdc7510ab9def6bc851d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S095070512500348X&tid=spdf-9019d78c-533f-463c-9f53-5867c9fb824e&sid=e07704d5457ee4415d5986f319eef390e0cfgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1114595752005f03025556&rr=933c8aa21da4eab1&cc=kr

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