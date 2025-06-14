
import shutil
from pathlib import Path
import pandas as pd
import os
from PIL import Image
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
from PIL import ImageFile
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, f1_score
ImageFile.LOAD_TRUNCATED_IMAGES = True

class data_preparation():
    def __init__(self, label_directory, img_directory, augmented_folder):
        self.augmented_folder = augmented_folder
        # cropped_high_res images images[fileName]=img
        self.images = self.parse_images(img_directory)
        self.len_augmented_data_set = len(os.listdir(self.augmented_folder))
        # (label, img)
        self.og_data = self.create_labels(label_directory, self.images)
        # creates augmented images: (label, img) and adds to augmented file
        self.augmented_imgs = self.create_labels_Augmented(self.augmented_folder)
        self.data_set = self.og_data + self.augmented_imgs


        # self.Training_dataset, self.testing_dataset = random_split(self.data_set, [.8 * len(self.data_set), .2 * len(self.data_set)])

    def label(self, description, img):
        if "glaucoma" in description.lower():
            return (-1, img)
        if "control" in description.lower():
            return (0, img)
        if "amd" in description.lower():
            return (1, img)
        return (-2, img)

    def create_labels(self, Directory, img_dict):
        data = pd.read_csv(Directory)
        descriptions = {"OD": data["R (OD)"].values, "OS": data["L (OS)"].values}
        all = []
        labeled = []
        #figure out how to configure the title to be compatible for 1000 file
        for key, value in img_dict.items():
            key = key.replace("/content/drive/MyDrive/", "")
            key = key.replace("1000/", "")
            O=key[5:7]
            id = int(key[2:4]) - 1
            
            diagnosis = descriptions[O][id]
            
            # labeled.append((diagnosis, value))
            all.append((diagnosis, value))
            labeled.append(self.label(diagnosis, value))
        for _ in range(5):
            self.create_augmented_images(all, self.augmented_folder)
        return labeled

    def create_labels_Augmented(self, Directory):
        all = []
        for filename in os.listdir(Directory):
            file_path = os.path.join(Directory, filename)
            try:
                all.append(self.label(filename, Image.open(open(file_path, 'rb'))))
            except:
                print("")

        return all

    def parse_AI_gen(self, Directory):
        img_dict = {}
        for filename in os.listdir(Directory):
            file_path = os.path.join(Directory, filename)
            if file_path[-5]=='0':
                try:
                    img_dict[Directory] = Image.open(open(file_path, 'rb'))
                except:
                    print("")
            #else:
                #self.trash(filename)
        
        return img_dict

    def parse_images(self, directory_path):
        # 1096_OD and 1096_OS to be delt w later
        # .DS_Store => .png
        img_dict = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isdir(file_path):
                for key,img in self.parse_AI_gen(file_path).items():
                    img_dict[key] = img
                
            else:
                try:
                    img_dict[filename] = Image.open(open(file_path, 'rb'))
                except:
                    print("")
        return img_dict

    def create_augmented_images(self, images, folder):
        transforms = v2.Compose([
            v2.Resize(size=[224, 224]),
            v2.RandomHorizontalFlip(.5),
            v2.RandomVerticalFlip(.3),
            v2.RandomRotation(degrees=20, fill=0),
            v2.RandomRotation(degrees=35, fill=0),
            v2.ElasticTransform([50.0], 5.0, Image.BILINEAR, 0),
            v2.GaussianBlur(kernel_size=[5, 5], sigma=(.01, 0.1)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        ])
        counter = self.len_augmented_data_set
        for label, img in images:
            augmented_img = transforms(img)
            pil_img_img = v2.functional.to_pil_image(augmented_img)
            lbl = ""
            if "amd" in label.lower():
                lbl = "AMD"
            if "glaucoma" in label.lower():
                lbl = "Glaucoma"
            if "control" in label.lower():
                lbl = "Control"
            filename = f"{folder}/{lbl}_{counter}.png"
            file_path = os.path.join("augmented_AI_gen", filename)
            pil_img_img.save(filename)
            # vector = Image.open(open(file_path, 'rb'))
            # all.append((label, vector))
            counter += 1
        # figure out how to display as "PIL.Image.Image" and figure out properties of "torchvision.tv_tensors._image.Image".
        return

    def split(self, data):
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        test_size = len(data) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            data, [train_size, val_size, test_size]
        )

        batch_size = 5  # Optional: change the batch size and explore it's effect in following training steps
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        dataloaders = {
            'train': train_dataloader,
            'validation': val_dataloader,
            'test': test_dataloader
        }
        return dataloaders
    
    def trash(self,path):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()  # Remove file or symlink
            elif path.is_dir():
                shutil.rmtree(path)  # Remove directory and its contents recursively
        except Exception as e:
            print(f'Failed to delete {path}. Reason: {e}')
    
    def empty_directory(self, directory):
        deleted=0
        total=0
        AMC=0
        Control=0
        Glaucoma=0
        i=200
        dir_path = Path(directory)
        for path in dir_path.iterdir():
            total+=1
            img_file = path.name[0]
            if img_file=='A':
                AMC+=1
                if AMC>i:
                    self.trash(path)
                    deleted+=1
                    
            if img_file=='G':
                Glaucoma+=1
                if Glaucoma>i:
                    self.trash(path)
                    deleted+=1
                    
            if img_file=='_' or img_file=='C':
                Control+=1
                if Control>i:
                    self.trash(path)
                    deleted+=1
        print("total: "+str(total))
        print("deleted: "+str(deleted))
        
        return
    
   