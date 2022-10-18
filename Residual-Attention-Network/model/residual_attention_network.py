import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from .basic_layers import ResidualBlock
from .attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from .attention_module import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar
import random
import PIL


class ResidualAttentionModel_448input(nn.Module):
    # for input size 448
    def __init__(self):
        super(ResidualAttentionModel_448input, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # tbq add
        # 112*112
        self.residual_block0 = ResidualBlock(64, 128)
        self.attention_module0 = AttentionModule_stage0(128, 128)
        # tbq add end
        self.residual_block1 = ResidualBlock(128, 256, 2)
        # 56*56
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        out = self.residual_block0(out)
        out = self.attention_module0(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def is_bn(m):
    return isinstance(m, nn.modules.batchnorm.BatchNorm2d) | isinstance(m, nn.modules.batchnorm.BatchNorm1d)


def take_bn_layers(model):
    for m in model.modules():
        if is_bn(m):
            yield m


class FreezedBnModel(nn.Module):
    def __init__(self, model, is_train=True):
        super(FreezedBnModel, self).__init__()
        self.model = model
        self.bn_layers = list(take_bn_layers(self.model))

    def forward(self, x):
        is_train = self.bn_layers[0].training
        if is_train:
            self.set_bn_train_status(is_train=False)
        predicted = self.model(x)
        if is_train:
            self.set_bn_train_status(is_train=True)

        return predicted

    def set_bn_train_status(self, is_train: bool):
        for layer in self.bn_layers:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train  # TODO: layer.requires_grad = is_train - check is its OK
            layer.bias.requires_grad = is_train


class batch_RAN_Deepfake(nn.Module):
    def __init__(self, model, grad_layer, num_classes,
                 am_pretraining_epochs=1, ex_pretraining_epochs=1,
                 grad_magnitude=1, last_ex_epoch=500):
        super(batch_RAN_Deepfake, self).__init__()

        self.model = model

        # using freezed BN model configuration on the second path in AM training to not influence the statistics
        self.freezed_bn_model = FreezedBnModel(model)

        # print(self.model)
        self.grad_layer = grad_layer

        self.num_classes = num_classes
        # print("before norm")
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        # norm = Normalize(mean=mean, std=std)
        # print("before em fill")
        self.em_fill_color = torch.tensor([228.0 / 255.0, 249.0 / 255.0, 52.0 / 255.0]).view(1, 3, 1, 1).cuda()
        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.5 # 0.5
        self.omega = 30
        self.grad_magnitude = grad_magnitude

        self.am_pretraining_epochs = am_pretraining_epochs
        self.ex_pretraining_epochs = ex_pretraining_epochs
        self.last_ex_epoch = last_ex_epoch
        self.cur_epoch = 0
        self.enable_am = False
        self.enable_ex = False
        if self.am_pretraining_epochs == 0:
            self.enable_am = True
        if self.ex_pretraining_epochs == 0:
            self.enable_ex = True

        self.use_hook = 0

    def _register_hooks(self, grad_layer):
        def forward_hook(module, input, output):
            self.feed_forward_features = output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        # print("print named modules")
        for idx, m in self.model.named_modules():
            if idx in self.grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                # break
        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe_multibatch(self, labels):

        ohe = torch.nn.functional.one_hot(labels, self.num_classes).float()
        ohe = torch.autograd.Variable(ohe)

        return ohe

    def forward(self, images, labels, train_flag=False, image_with_masks=None,
                e_masks=None, has_mask_indexes=None):  # TODO: no need for saving the hook results ; Put Nan

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.model.training

        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            logits_cl = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            if not is_train:
                pred = F.softmax(logits_cl).argmax(dim=1)
                # print("pred")
                # print(pred)
                # print(F.softmax(logits_cl))
                labels_ohe = self._to_ohe_multibatch(pred).cuda()
                # print("labels_ohe")
                # print(labels_ohe)
            else:
                if type(labels) is tuple:
                    labels_ohe = torch.stack(labels)
                else:
                    labels_ohe = labels

            # gradient = logits * labels_ohe
            # old version: grad_logits = (logits_cl * labels_ohe).sum(dim=1)  # BS x num_classes
            # print("old grad_logits")
            # print((logits_cl * labels_ohe).sum(dim=1))
            # new version:
            grad_logits = logits_cl.sum(dim=1)  # BS x num_classes
            # print("new grad_logits")
            # print(grad_logits)
            # exit(1)
            grad_logits.backward(retain_graph=True, gradient=torch.ones_like(grad_logits))
            self.model.zero_grad()

        backward_features = self.backward_features  # BS x C x H x W
        fl = self.feed_forward_features  # BS x C x H x W
        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        # print("Ac.shape")
        # print(Ac.shape)
        Ac = F.relu(Ac)
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear')
        # print("interpolate.shape")
        # print(Ac.shape)
        Ac_min, _ = Ac.view(len(images), -1).min(dim=1)
        Ac_max, _ = Ac.view(len(images), -1).max(dim=1)
        import sys
        eps = torch.tensor(sys.float_info.epsilon).cuda()
        scaled_ac = (Ac - Ac_min.view(-1, 1, 1, 1)) / \
                    (Ac_max.view(-1, 1, 1, 1) - Ac_min.view(-1, 1, 1, 1)
                     + eps.view(1, 1, 1, 1))

        return logits_cl, scaled_ac

    def EX_enabled(self):
        return self.enable_ex


class ResidualAttentionModel_92(nn.Module):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_92, self).__init__()
        # Feed-forward features
        self.output_dir = None
        self.output_intermediate = None
        self.output_num = None
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.attention_module2_2 = AttentionModule_stage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.attention_module3_2 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionModule_stage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.pre_mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048,1)

    def forward(self, x, filename_list=None, output_intermediate=False, output_num=3, output_dir="/home/shuoli/"):
        # print(x.shape)  # 3, 224, 224
        self.output_intermediate = output_intermediate
        self.output_num = output_num
        self.output_dir = output_dir
        out = self.conv1(x)
        # print(out.shape)  # 64, 112, 112
        out = self.mpool1(out)
        # print(out.shape)  # 64, 56, 56
        # print(out.data)
        out = self.residual_block1(out)
        # print(out.shape)  # 256, 56, 56
        out = self.attention_module1(out)
        print(out.shape)
        exit(1)
        if self.output_intermediate:
            candidates = random.sample(range(256), self.output_num)
            for channel_idx in candidates:
                PIL.Image.fromarray((out[channel_idx].squeeze().cpu().detach().numpy()).round().astype(
                    np.uint8), 'L').save("stage1.png")

        # print(out.shape)  # 256, 56, 56
        out = self.residual_block2(out)
        # print(out.shape)  # 512, 28, 28
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        # print(out.shape)  # 512, 28, 28
        out = self.residual_block3(out)
        # print(out.shape)  # 1024, 14, 14
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        # print(out.shape)  # 1024, 14, 14
        out = self.residual_block4(out)
        # print(out.shape)  # 2048, 7, 7
        out = self.residual_block5(out)
        # print(out.shape)  # 2048, 7, 7
        out = self.residual_block6(out)
        # print(out.shape)  # 2048, 7, 7
        out = self.pre_mpool2(out)
        # print(out.shape)  # 2048, 1, 1
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)  # 2048
        out = self.fc(out)
        # print(out.shape)  # 1

        return out


class ResidualAttentionModel_56(nn.Module):
    # for input size 224
    def __init__(self):
        super(ResidualAttentionModel_56, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel_92_32input(nn.Module):
    # for input size 32
    def __init__(self):
        super(ResidualAttentionModel_92_32input, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 16*16
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128)  # 16*16
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 8*8
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256)  # 8*8
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256)  # 8*8 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 4*4
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 4*4 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 4*4
        self.residual_block5 = ResidualBlock(1024, 1024)  # 4*4
        self.residual_block6 = ResidualBlock(1024, 1024)  # 4*4
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4, stride=1)
        )
        self.fc = nn.Linear(1024,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResidualAttentionModel_92_32input_update(nn.Module):
    # for input size 32
    def __init__(self):
        super(ResidualAttentionModel_92_32input_update, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        # self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(32, 32), size2=(16, 16))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(16, 16))  # 16*16 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(1024,10)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out