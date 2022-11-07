import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import os

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

# Data loading for Pascal VOC 2007
class PascalVOC2007DataModule(pl.LightningDataModule):
    def setup(self, stage):
        # download
        os.system('wget -nc https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar')
        os.system('wget -nc https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar')
        # unpack
        os.system('tar --skip-old-files -xf VOCtrainval_06-Nov-2007.tar')
        os.system('tar --skip-old-files -xf VOCtest_06-Nov-2007.tar')

        data_aug_transforms = [transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(180),
                            transforms.RandomGrayscale()]
                            
        img_transform = transforms.Compose(data_aug_transforms+[ 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    transforms.CenterCrop((2*2*2*2 * 23, 2*2*2*2 * 31))
                                    ])

        seg_mask_transform = transforms.Compose([
                                        transforms.Lambda(lambda x : torch.from_numpy(np.array(x)).long()),
                                        transforms.CenterCrop((2*2*2*2 * 23, 2*2*2*2 * 31))
                                        ])

        self.train_dataset = torchvision.datasets.VOCSegmentation(root='./', 
                                               year = '2007', 
                                               image_set = 'train',
                                               transform=img_transform, 
                                               target_transform=seg_mask_transform, 
                                               download = False)

        self.val_dataset = torchvision.datasets.VOCSegmentation(root='./', 
                                               year = '2007', 
                                               image_set = 'val',     
                                               transform=img_transform, 
                                               target_transform=seg_mask_transform, 
                                               download = False)

        self.test_dataset = torchvision.datasets.VOCSegmentation(root='./', 
                                               year = '2007', 
                                               image_set = 'test',     
                                               transform=img_transform, 
                                               target_transform=seg_mask_transform, 
                                               download = False)

        # for testing
        #mask = list(range(2))
        #self.train_dataset = torch.utils.data.Subset(self.train_dataset, mask)
        #self.val_dataset = torch.utils.data.Subset(self.val_dataset, mask)
        #self.test_dataset = torch.utils.data.Subset(self.test_dataset, mask)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=32,
                                           shuffle=False,
                                           num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset,
                                           batch_size=32,
                                           shuffle=False,
                                           num_workers=8)

# Exercise 1 (5 points):
# Read the U-Net paper by Ronneberger et al. 
# - Why do the authors state that the U-Net works good for segmentation.

# The argumentation of the authors is that the U-Net is able to capture context in its contracting path whilst being able 
# to exercise precise localization in its expanding path. The usual contracting network is supplemented by upsampling operators 
# that replace pooling operators. Further, in order to improve localization in the expanding path, high resolution features from
# the contracting path are combined with the upsampled output at each step. Succesive convolution layers, consequently, learn to
# assemble more precise output based on this information. Lastly, they argue that their upsampling part is able to propagate more
# context to higher resolution layers, as they have a large number of feature channels.

# - Do you think their argumentation  makes sense?

# Their argumentation makes sense. Their architecture features a lot of properties that makes it successful. They employ large 
# number of feature channels allowing for better traversing of information among each convolution layer. The contraction path
# may be pretty standard but their expanding path is the heart of this architecture. The process of appending the learned feature
# maps from each step of the contraction path to the corresponding expending step is clever, in that it ensures that the features 
# that are learned during contraction are then reused in reconstruction. This seemingly improves localization by multiple 
# magnitudes. This is conceivable, because upsampling is a sparse operation and a good prior from earlier stages helps to 
# better achieve localization.

# - What are your arguments that the U-Net works well?

# A majority of arguments, why we think that the U-Net works well was already stated in the previous question. Though, we think 
# that is not all there is to it. We imagine that the loss calculation in the U-Net may also play a major role in its success.
# U-Net uses a loss weighting scheme for each pixel such that there is a higher weight at the border of segmented objects. This
# loss helped the U-Net model segment cells in biomedical imgages such that even individual cells were easily identified within 
# the binary segmentation map. 


# Exercise 2 (20 points):
# Implement a U-Net with the help of the following classes:
# 1) Block: conv -> relu -> conv. The first convolution maps input channel to output channel. 
#           Dimensions are to be retained, use filter size of 3.
#           For better convergence use Batch-norm after each ReLU.
# 2) Encoder: Compose multiple blocks and downsample between blocks by factor of 2 with max-pooling
# 3) Decoder: Compose multiple transpose convolutions and convolutions blocks that 
#             (i) upsample the input from the previous layer through transpose convolutions
#             (ii) concatenate the upsampled features with the corresponding layer from the encoder
#             (iii) pass it through a convolution block
# 4) UNet: Arrange encoder and decoder blocks into an U-Net. Add final 1x1 convolution to produce pixel-wise predictions.
# Note:
# - We do not completely follow the original U-Net architecture which slighly reduced dimensions by using convolutions without padding. Input and output dimensions should stay same.
# - We use pytorch lightning for the training. It might be good to acquaint yourself with the framework.
#
# Exercise 3:
# Log train loss, train accuracy, train mean IoU (also called Jaccard index), validation accuracy and and validation mean IoU.
# To this end you can use the tensorboard graphs and write a short report.
# 
# Extra exercise: (you can earn extra points, but it is not required, 10 points)
# Improve the U-Net by using augmentations, better training, more involved U-Net blocks (see the network components from Lecture 5) or better loss.
# The highest obtained mean IoU (with proof) will get a honorable mention.

class UNet(pl.LightningModule):

    def __init__(self, chn=(3,64,128,256,512,1024), num_class=21):
        super().__init__()
        self.num_class = num_class
        self.learning_rate = 0.01#1e-3

        self.train_pixel_acc = torchmetrics.classification.MulticlassAccuracy( num_classes=21, ignore_index=255, average='weighted')
        self.train_J = torchmetrics.classification.MulticlassJaccardIndex(num_classes = 21, ignore_index = 255, average='weighted')

        self.val_pixel_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = 21, ignore_index = 255, average='weighted')
        self.val_J = torchmetrics.classification.MulticlassJaccardIndex(num_classes = 21, ignore_index = 255, average='weighted')

        self.test_pixel_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = 21, ignore_index = 255, average='weighted')
        self.test_J = torchmetrics.classification.MulticlassJaccardIndex(num_classes = 21, ignore_index = 255, average='weighted')

        # TODO: Define U-Net layers #
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )
            
        class Up(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()

                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = double_conv(in_channels, out_channels) 

            def forward(self, x1, x2):
                x1 = self.up(x1)

                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                            diffY // 2, diffY - diffY // 2])

                x = torch.cat([x2, x1], dim=1)
                return self.conv(x)
        
        self.inc = double_conv(chn[0], chn[1])  #    3 ->   64
        self.down1 = down(chn[1], chn[2])       #   64 ->  128      
        self.down2 = down(chn[2], chn[3])       #  128 ->  256
        self.down3 = down(chn[3], chn[4])       #  256 ->  512
        self.down4 = down(chn[4], chn[5])       #  512 -> 1024
        self.up1 = Up(chn[5], chn[4])           # 1024 ->  512
        self.up2 = Up(chn[4], chn[3])           #  512 ->  256
        self.up3 = Up(chn[3], chn[2])           #  256 ->  128
        self.up4 = Up(chn[2], chn[1])           # 128 ->    64
        self.out = nn.Conv2d(chn[1], num_class, kernel_size=1)

        #############################

    def forward(self, x):
        # TODO: Define U-Net layers #
        x1  = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
        #############################

    def decode_segmap(self, prediction):
        label_colors = torch.tensor([(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = torch.zeros_like(prediction, dtype=torch.uint8)
        g = torch.zeros_like(prediction, dtype=torch.uint8)
        b = torch.zeros_like(prediction, dtype=torch.uint8)
        for l in range(0, self.num_class):
            idx = prediction == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = torch.stack([r, g, b], axis=1)
        return rgb

    def training_step(self, train_batch, batch_idx):
        images, seg_mask = train_batch

        # Forward pass
        outputs = model(images)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

        loss = criterion(outputs, seg_mask)
        self.log('train_loss', loss)

        # train acc and mIoU
        pred_seg_mask = torch.argmax(outputs, 1)
        self.train_pixel_acc(pred_seg_mask, seg_mask)
        self.train_J(pred_seg_mask, seg_mask)

        self.log('train_acc', self.train_pixel_acc, on_step=False, on_epoch=True)
        self.log('train_mIoU', self.train_J, on_step=False, on_epoch=True)

        # visualize CNN prediction
        if batch_idx == 2:
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.255]
                )
            pred = torch.argmax(outputs,1)
            pred_seg_map = self.decode_segmap(pred)
            gt_seg_map = self.decode_segmap(seg_mask)

            img_grid = torchvision.utils.make_grid(
                torch.cat((inv_normalize(images.float()), gt_seg_map, pred_seg_map), dim=0), 
                nrow=2)
            self.logger.experiment.add_image('train_pred', img_grid.float(), self.current_epoch)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images, seg_mask = val_batch
        outputs = model(images)
        pred_seg_mask = torch.argmax(outputs, 1)

        # pixel-wise accuracy
        self.val_pixel_acc(pred_seg_mask, seg_mask)

        # the Jaccard index (mean IoU)
        self.val_J(pred_seg_mask, seg_mask)

        self.log('val_acc', self.val_pixel_acc, on_step=False, on_epoch=True)
        self.log('val_mIoU', self.val_J, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, seg_mask = batch
        outputs = model(images)
        pred_seg_mask = torch.argmax(outputs, 1)

        self.test_pixel_acc(pred_seg_mask, seg_mask)
        self.test_J(pred_seg_mask, seg_mask)

        self.log("test_acc", self.test_pixel_acc)
        self.log("test_mIoU", self.test_J)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return opt


data_module = PascalVOC2007DataModule()

# train
model = UNet()
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=35, log_every_n_steps=1, callbacks=[lr_monitor], auto_lr_find=True)
#trainer.tune(model, datamodule=data_module)
trainer.fit(model, data_module)
trainer.test(model, data_module)

