import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import KFold
from torchvision.transforms import v2
from PIL import ImageFile, Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, f1_score
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import random
import timm
from torchvision.models import ( vit_b_16,ViT_B_16_Weights,swin_t,Swin_T_Weights)
from Data_Sort import data_preparation
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#vit_weights  = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
#swin_weights = Swin_T_Weights.IMAGENET1K_V1

class OCTDataset():
    def __init__(self, data, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        label, pil_image = self.data[idx]
        label = int(label)

        pil_image = pil_image.convert("RGB")
         
        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = v2.functional.to_tensor(pil_image)

        return image_tensor, label

    def __len__(self):
        return len(self.data)

def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
        
"""
class SwinT(nn.Module):
    def __init__(self, num_classes: int = 2,
                 weights: Swin_T_Weights = Swin_T_Weights.IMAGENET1K_V1):
        super().__init__()

        # backbone with pretrained weights
        self.backbone = swin_t(weights=weights)

        # swap the 1000-way ImageNet head for your 2-way head
        in_feats = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)
         
class ViTModel(nn.Module):
    def __init__(self, num_classes: int = 2,
        weights: ViT_B_16_Weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1):
    
        #IMAGENET1K_V1  – original ViT-B-16 weights  
        #IMAGENET1K_SWAG_E2E / _LINEAR – stronger SWAG-pretrained variants
        
        super().__init__()

        # backbone with pretrained weights
        self.backbone = vit_b_16(weights=weights)

        # swap the 1000-way ImageNet head for your 2-way head
        in_feats = self.backbone.heads.head.in_features     # <- note the extra 'heads'
        self.backbone.heads.head = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)
"""
class SwinT(nn.Module):
  def __init__(self):
    super(SwinT, self).__init__()
    self.model1 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

    self.fc1 = nn.Linear(1000, 2)

  def forward(self, x):
     x = self.model1(x)
     x = self.fc1(x)
     return x

class ViTModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTModel, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
 
class LargerCNN2D(nn.Module):
    def __init__(self, num_classes=2):
        super(LargerCNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    

    

def dataset(img_directory, augmented_folder, backbone):
    
    
    common_transforms = v2.Compose([
        v2.Resize(size=[224, 224]),
        lambda img: img.convert("RGB") if img.mode != "RGB" else img,
        lambda img: v2.functional.to_image(img),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.ConvertImageDtype(torch.float32),
    ])
    
    """
    if backbone.lower().startswith("vit"):
        common_transforms = vit_weights.transforms()
    else:
        common_transforms = swin_weights.transforms()
    """
    high_res_dp = data_preparation(
       label_directory='pairedOCT(Master List).csv',
       img_directory=img_directory,
       augmented_folder=augmented_folder
    )
    high_res_dp.empty_directory(augmented_folder)
    Glaucoma_Data = []
    Control_Data = []
    AMD_Data = []

    for i in high_res_dp.data_set:
        if i[0] == -1:   # i[0] == -1 means Glaucoma
            Glaucoma_Data.append((1, i[1]))
        if i[0] == 1:
            AMD_Data.append((1, i[1]))
        elif i[0] == 0:  # i[0] ==  0 means Control
            Control_Data.append((0, i[1]))

    binary_data = AMD_Data + Control_Data
    return OCTDataset(binary_data, transform=common_transforms)
    
def test(network, testloader, loss_function):
    network.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = network(inputs)
            loss = loss_function(outputs, targets)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_val_loss = running_val_loss / len(testloader)
    val_accuracy = 100.0 * correct / total

    return network, avg_val_loss, val_accuracy

def run(model, og_folder, augmented_folder, test_og, test_augmented, epochs, seed): 
     # Configuration options
    k_folds = 5
    num_epochs = epochs
    loss_function = nn.CrossEntropyLoss()
    
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    all_test_loss = []
    all_ai_loss = []
    all_ai_acc = []
    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(seed)
    
    #figure out how to make AI be tested with different dataset
    
    if og_folder != '1000':
        #training_dataset = dataset(og_folder, augmented_folder)
        #AI_dataset = dataset(test_og, test_augmented)
        
        training_dataset = dataset(og_folder, augmented_folder, backbone=type(model).__name__)
        AI_dataset = dataset(test_og, test_augmented, backbone=type(model).__name__)
    else:
        #AI_test_and_train_dataset = dataset(og_folder, augmented_folder)
        AI_test_and_train_dataset = dataset(og_folder, augmented_folder, backbone=type(model).__name__)
        dataset_size = len( AI_test_and_train_dataset)
        
        test_size = int(0.2 * dataset_size)
        train_size = dataset_size - test_size

        torch.manual_seed(42)

        training_dataset, AI_dataset = random_split(AI_test_and_train_dataset, [train_size, test_size])
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    avg_AI_acc = 0.0
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(training_dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Subsets for training and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        ai_sampler = torch.utils.data.SubsetRandomSampler([i for i in range(len(AI_dataset))])
        
        trainloader = DataLoader(training_dataset, batch_size=10, sampler=train_subsampler)
        testloader = DataLoader(training_dataset, batch_size=10, sampler=test_subsampler)
        ai_loader = DataLoader(AI_dataset, batch_size=10, sampler=ai_sampler)
        # Init the neural network
        network = model.to(device)
        network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)
         # ↓ add this:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
        scheduler.step()
        """
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',         # we want to lower lr when val loss stops decreasing
            factor=0.5,         # new_lr = old_lr * factor
            patience=3,         # wait 3 epochs without improvement
        )
        """
        loss_function = nn.CrossEntropyLoss()

    
        final_val_loss = 0.0
        final_val_acc = 0.0
        
        # Training loop
        for epoch in range(num_epochs):
            # Switch to train mode
            network.train()
            running_train_loss = 0.0

            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = network(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            # Compute average training loss over all batches
            avg_train_loss = running_train_loss / len(trainloader)

            # ---- Evaluate on the validation set after this epoch ----
            #check how to pass a model around
            network, avg_val_loss, val_accuracy=test(network, testloader, loss_function)
            
            #scheduler.step(avg_val_loss)
            
            final_val_loss = avg_val_loss
            final_val_acc = val_accuracy
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_accuracy:.2f}%")
            
        print('Training process has finished. Saving trained model.')
        # Save the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(network.state_dict(), save_path)

        # Final evaluation for this fold
        network, train_loss, train_accuracy = test(network, trainloader, loss_function)
        network, test_loss, _ = test(network, testloader, loss_function)
        network, ai_test_loss, AI_accuracy = test(network,ai_loader, loss_function)
        
        print(f"Fold {fold} Summary:")
        print(f"  - Train Accuracy: {train_accuracy:.2f}%")
        print(f"  - Validation Loss: {final_val_loss:.4f}, Validation Accuracy: {final_val_acc:.2f}%")
        print(f"  - Test Loss: {test_loss:.4f}")
        print(f"  - AI Test Loss: {ai_test_loss:.4f}, AI Accuracy: {AI_accuracy:.2f}%")
        
        #results[fold] = (accuracy, AI_accuracy)
        all_train_acc.append(train_accuracy)
        all_val_loss.append(final_val_loss)
        # The question specifically asks for "Validation Accuracy":
        all_val_acc.append(final_val_acc)
        all_test_loss.append(test_loss)
        all_ai_loss.append(ai_test_loss)
        all_ai_acc.append(AI_accuracy)
        #AI_Test_loss, AI_Test_Acc = AI_test.test_model(network, loss_function)
        
        #figure out why avg ai accuracy no longer works

    avg_train_acc = sum(all_train_acc) / len(all_train_acc)
    avg_val_loss = sum(all_val_loss) / len(all_val_loss)
    avg_val_acc = sum(all_val_acc) / len(all_val_acc)
    avg_test_loss = sum(all_test_loss) / len(all_test_loss)
    avg_ai_loss = sum(all_ai_loss) / len(all_ai_loss)
    avg_ai_acc = sum(all_ai_acc) / len(all_ai_acc)
    # Print fold results
    
    return [avg_test_loss, avg_train_acc, avg_val_loss, avg_val_acc, avg_ai_loss, avg_ai_acc]

    
if __name__ == '__main__':
    #figure out how to implement pretrained weights
    Glaucoma_VitModel_High_res = []
    
    print("remove changing LR. Check chagbt for og code")
    print("Glaucoma ViTModel high res")
    for i in range(0,5):
        print("Running Round: "+str(i+1))
        Glaucoma_VitModel_High_res.append(run(SwinT(), '1000', 'augmented_AI_gen', '1000', 'augmented_AI_gen', 120, 40+i))
    
    for i in range(0,5):
        avg_test_loss, avg_train_acc, avg_val_loss, avg_val_acc, avg_ai_loss, avg_ai_acc = Glaucoma_VitModel_High_res[i]
        print("\n===== CROSS-VALIDATION AVERAGES =====")
        print(f"Average Train Loss:             {avg_test_loss:.4f}")
        print(f"Average Train Accuracy:        {avg_train_acc:.2f}%")
        print(f"Average Validation Loss:       {avg_val_loss:.4f}")
        print(f"Average Validation Accuracy:   {avg_val_acc:.2f}%")
        print(f"Average AI Test Loss:          {avg_ai_loss:.4f}")
        print(f"Average AI Accuracy:           {avg_ai_acc:.2f}%")
        print(" ")