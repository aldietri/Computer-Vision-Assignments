import torch
import pytorch_lightning as pl
import os
import importlib
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchmetrics
import random
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import PIL
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SimCLR losses: Fill out all the loss and their heler functions below
# We will below use the following quantities:

# cosine similarity
#              < x, y >
# sim(x, y) = ----------
#             ||x||||y||

# InfoNCE:
#                          exp(sim(z , z ) / tau)
#                                   i   j
# l(i, j) =  - log------------------------------------------
#                  __ 2N
#                 \         1         exp(sim(z , z ) / tau)
#                 /__ k = 1  {k != i}          i   k

# SimCLR loss:
#      1    __ N
# L = ---  \        [l(k, k + N) + l(k + N, k)]
#     2N   /__ k = 1

# note that in the SimCLR loss L the positive pairs are, in contrast to the lecture, pairs {i,i+N}, where N is the batch size

def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    numerator = torch.dot(z_i, z_j)
    denominator = torch.linalg.norm(z_i) * torch.linalg.norm(z_j)
    norm_dot_product = torch.div(numerator, denominator)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product

# test sim code
# sim_test = torch.rand((10))
# print('the below values should be 1')
# print(sim(sim_test, sim_test).item()) 
# print(sim(0.5*sim_test, sim_test).item()) 
# print(sim(sim_test, 0.5*sim_test).item()) 
# print('the below values should be -1')
# print(sim(sim_test, -sim_test).item()) 
# print(sim(0.5*sim_test, -0.5*sim_test).item()) 
# print(sim(0.5*sim_test, -sim_test).item()) 


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        numerator_l_k = torch.exp(sim(z_k, z_k_N) / tau)
        numerator_l_k_N = torch.exp(sim(z_k_N, z_k) / tau)

        denominator_l_k = 0
        denominator_l_k_N = 0

        for i in range(2*N):
            if i != k:
                z_i = out[i]
                denominator_l_k += torch.exp(sim(z_k, z_i) / tau)
            if i != (k+N):
                z_i = out[i]
                denominator_l_k_N += torch.exp(sim(z_k_N, z_i) / tau)

        l_k = - torch.log(numerator_l_k / denominator_l_k)
        l_k_N  = - torch.log(numerator_l_k_N / denominator_l_k_N)

        total_loss += (l_k + l_k_N)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss



def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    numerator = torch.diag(out_left @ out_right.T)
    denominator = torch.linalg.norm(out_left, dim=1) * torch.linalg.norm(out_right, dim=1)
    pos_pairs = torch.div(numerator, denominator)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


# test sim_positive_pairs vs. looping over sim
#test_sim_pos_pairs_left = torch.rand(10,32)
#test_sim_pos_pairs_right = torch.rand(10,32)
#test_sim_pos_pairs = sim_positive_pairs(test_sim_pos_pairs_left, test_sim_pos_pairs_right)
#print('sim_poitive_pairs vs loog sim on positive pairs')
#for i in range(10):
#    sim_pos = sim(test_sim_pos_pairs_left[i], test_sim_pos_pairs_right[i])
#    print(str(test_sim_pos_pairs[i].item()) + ' vs ' + str(sim_pos.item()))


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    numerator = out @ out.T
    denominator = torch.linalg.norm(out, axis=1).view(-1, 1) @ torch.linalg.norm(out, axis=1).view(1, -1)
    sim_matrix = torch.div(numerator, denominator)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix

# test sim_matrix
#test_sim_matrix_input = torch.rand(10,32)
#test_sim_matrix = compute_sim_matrix(test_sim_matrix_input)
#print('test sim_matrix')
#for i in range(10):
#    for j in range(10):
#        print(str(test_sim_matrix[i,j]) + ' vs ' + str(sim(test_sim_matrix_input[i], test_sim_matrix_input[j])))


def simclr_loss_vectorized(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.

    exp_matrix = torch.exp(sim_matrix / tau)

    # This binary mask zeros out terms where k=i.
    mask = ~torch.eye(2*N, dtype=torch.bool, device=device)
    
    # We apply the binary mask.
    exp_matrix = exp_matrix * mask
    # exp_matrix[mask] = 0

    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denominator = torch.sum(exp_matrix, dim=1)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sim_ij = torch.diag(sim_matrix, N)
    sim_ji = torch.diag(sim_matrix, -N)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    positives = torch.cat([sim_ij, sim_ji], dim=0)
    numerator = torch.exp(positives / tau)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    loss_partial = -torch.log(numerator / denominator)
    loss = torch.sum(loss_partial) / (2*N)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss

# compare simclr_loss_vectorized against simclr_loss_naive
#test_out_left = torch.rand(10,32).to(device)
#test_out_right = torch.rand(10,32).to(device)
#print('compare simclr_loss_vectorized vs simclr_loss_naive')
#print(str(simclr_loss_vectorized(test_out_left, test_out_right, 0.1).item()) + ' vs ' + str(simclr_loss_naive(test_out_left, test_out_right, 0.1).item()))



# next fill out the data augmentation

def compute_simclr_train_transform():
    """
    This function returns a composition of data augmentations to a single training image.
    Complete the following lines. Hint: look at available functions in torchvision.transforms
    """
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = torchvision.transforms.Compose([
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Check out transformation functions defined in torchvision.transforms #
        # The first operation is filled out for you as an example.
        ##############################################################################
        # Step 1: Randomly resize and crop to 32x32.
        torchvision.transforms.RandomCrop(32),
        
        # Step 2: Horizontally flip the image with probability 0.5
        torchvision.transforms.RandomHorizontalFlip(p=0.5),

        # Step 3: With a probability of 0.8, apply color jitter (you can use "color_jitter" defined above.
        torchvision.transforms.RandomApply([color_jitter], p=0.8),

        # Step 4: With a probability of 0.2, convert the image to grayscale (and do not forget to force it to have 3 chanels still!)
        torchvision.transforms.RandomGrayscale(p=0.2),

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform

def compute_train_transform():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform
    
def compute_test_transform():
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform

# CIFAR10
                               
class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = PIL.Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            ##############################################################################
            # TODO: Start of your code.                                                  #
            #                                                                            #
            # Apply self.transform to the image to produce x_i and x_j in the paper #
            ##############################################################################

            x_i = self.transform(img)
            x_j = self.transform(img)

            ##############################################################################
            #                               END OF YOUR CODE                             #
            ##############################################################################

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target

batch_size = 2048
pair_train_set = CIFAR10Pair(root='data', train=True,
                                         transform=compute_simclr_train_transform(), download=True)
pair_train_loader = torch.utils.data.DataLoader(pair_train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=compute_train_transform())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=compute_test_transform())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
 

# SimCLR training class
class SimCLR(pl.LightningModule):
    def __init__(self, encoder, tau, encoder_feature_dim, projection_dim):
        super().__init__()
        self.encoder = encoder
        self.tau = tau
        self.encoder_feature_dim = encoder_feature_dim
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4

        for param in self.encoder.parameters():
            param.requires_grad = True

        self.g = nn.Sequential(nn.Linear(encoder_feature_dim, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, projection_dim, bias=True))

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return torch.nn.functional.normalize(feature, dim=-1), torch.nn.functional.normalize(out, dim=-1)

    # check how many positive pairs were recognized correctly (resp. in top 5 predictions)
    def accuracy(self, proj_left, proj_right):
        N = proj_left.shape[0]

        ########
        # TODO #
        ########

        sim_matrix = compute_sim_matrix(torch.cat([proj_left, proj_right]))
        sim_matrix.fill_diagonal_(float("-inf"))

        target = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(device)

        top1_acc = sum(torch.argmax(sim_matrix, dim=-1) == target).float().item() / (2*N)
        top5_acc = sum([target[i] in torch.topk(sim_matrix, k=5, dim=-1).indices[i] for i in range(2*N)]) / (2*N)
 
        # Compute top1 and top5 accuracy of how often any element was correctly predicted (resp. whether it was in top 5 predictions)
        return top1_acc, top5_acc

    def training_step(self, x):
        left_img, right_img, labels = x

        # pass through the model and compute loss (better use the vectorized version!)
        encoder_left, proj_left  = self(left_img)
        encoder_right, proj_right  = self(right_img)

        loss = simclr_loss_vectorized(proj_left, proj_right, self.tau)

        self.log('simclr_train_loss', loss)
        top1_acc, top5_acc = self.accuracy(proj_left, proj_right)
        self.log('simclr_train_top1_acc', top1_acc)
        self.log('simclr_train_top5_acc', top5_acc)

        return loss

    def validation_step(self, x):
        left_img, right_img, _ = x

    def configure_optimizers(self):
        #opt = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=500,
                                                            eta_min=self.learning_rate/50)
        return [optimizer], [lr_scheduler]

    
# now let us train:
encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

temperature=0.07
encoder_feature_dim = 1000
projection_feature_dim = 128

simclr_model = SimCLR(encoder, temperature, encoder_feature_dim, projection_feature_dim)
simclr_logger = pl.loggers.TensorBoardLogger("simclr_logs", name="simclr_lr_" + str(simclr_model.learning_rate) + "_temp_" + str(simclr_model.tau) + "_batch_size_" + str(batch_size))
simclr_trainer = pl.Trainer(logger=simclr_logger, accelerator="auto", devices=1, max_epochs=200)
simclr_trainer.fit(simclr_model, pair_train_loader)
#simclr_trainer.test(model, test_loader)

# save model for later reuse
model_save_path = 'resnet50_simclr_pretrained_ad_2'
torch.save(simclr_model.encoder.state_dict(), model_save_path)

# We will train a classification head on the pre-trained model
class Classifier(pl.LightningModule):
    def __init__(self, encoder, encoder_feature_dim, num_class):
        super(Classifier, self).__init__()
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6

        self.train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = num_class, average='weighted')
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = num_class, average='weighted')
        self.test_acc = torchmetrics.classification.MulticlassAccuracy(num_classes = num_class, average='weighted')

        # Encoder.
        self.encoder_feature_dim = encoder_feature_dim
        self.encoder = encoder
        
        # Classifier.
        self.fc = nn.Linear(1000, num_class, bias=True)

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

    def training_step(self, x):
        image, labels = x
        outputs = self(image)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.log('classifier_train_loss', loss)

        pred_labels = torch.argmax(outputs, 1)
        self.train_acc(pred_labels, labels)

        self.log('classifier_train_acc', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        outputs = self(images)
        pred_labels = torch.argmax(outputs, 1)

        self.val_acc(pred_labels, labels)

        self.log('classifier_val_acc', self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        pred_labels = torch.argmax(outputs, 1)

        self.test_acc(pred_labels, labels)

        self.log("classifier_test_acc", self.test_acc)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
        return opt

pretrained_encoder = torchvision.models.resnet18(weights=None)
pretrained_encoder.load_state_dict(torch.load(model_save_path))

# fine tune pre-trained model
finetuned_classifier_model = Classifier(pretrained_encoder, encoder_feature_dim, 10)
finetuned_classifier_model.freeze_encoder()
classifier_logger = pl.loggers.TensorBoardLogger("simclr_logs", name="finetune_pretrained_classifier_lr_" + str(finetuned_classifier_model.learning_rate) + "_batch_size_" + str(batch_size))
classifier_trainer = pl.Trainer(logger=classifier_logger, accelerator="auto", devices=1, max_epochs=10)
classifier_trainer.fit(finetuned_classifier_model, train_loader)
classifier_trainer.test(finetuned_classifier_model, test_loader)

# fine tune random model
rand_feature_classifier_model = Classifier(torchvision.models.resnet18(weights=None), encoder_feature_dim, 10)
rand_feature_classifier_model.freeze_encoder()
classifier_logger = pl.loggers.TensorBoardLogger("simclr_logs", name="finetune_random_classifier_lr_" + str(rand_feature_classifier_model.learning_rate) + "_batch_size_" + str(batch_size))
classifier_trainer = pl.Trainer(logger=classifier_logger, accelerator="auto", devices=1, max_epochs=10)
classifier_trainer.fit(rand_feature_classifier_model, train_loader)
classifier_trainer.test(rand_feature_classifier_model, test_loader)

# train model from scratch fully supervised
full_training_classifier_model = Classifier(torchvision.models.resnet18(weights=None), encoder_feature_dim, 10)
full_training_classifier_model.unfreeze_encoder()
classifier_logger = pl.loggers.TensorBoardLogger("simclr_logs", name="train_full_classifier_lr_" + str(full_training_classifier_model.learning_rate) + "_batch_size_" + str(batch_size))
classifier_trainer = pl.Trainer(logger=classifier_logger, accelerator="auto", devices=1, max_epochs=50)
classifier_trainer.fit(full_training_classifier_model, train_loader)
classifier_trainer.test(full_training_classifier_model, test_loader)        