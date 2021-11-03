import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet18']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
}

def decom_resnet18():
    model = resnet18(pretrained=False)

    C1 = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1
    )
    C2 = model.layer2
    C3 = model.layer3
    C4 = model.layer4

    return nn.Sequential(*C1), nn.Sequential(*C2), nn.Sequential(*C3), nn.Sequential(*C4)

class resnet(nn.Module):
    def __init__(self, stages, o_cn=1, final='abs'):
        super(resnet, self).__init__()
        self.C1, self.C2, self.C3, self.C4 = stages
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, o_cn, 1)
        )
        self.final = final
        self._initialize_weights()

    def forward(self, x):
        c1_out = self.C1(x)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        x = self.reg_layer(c4_out)
        if self.final == 'abs':
            x = torch.abs(x)
        elif self.final == 'relu':
            x = torch.relu(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



def resnet18_fpn():
    model = resnet(decom_resnet18(), o_cn=1, final='abs')
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


if __name__ == "__main__":
    print(resnet18_fpn())
