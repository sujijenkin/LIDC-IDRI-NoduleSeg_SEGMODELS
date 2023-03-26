#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install required libs
#!pip install -U segmentation-models-pytorch albumentations --user 


# In[ ]:


#!pip uninstall -y segmentation-models-pytorch


# ## Loading data

# For this example we will use **CamVid** dataset. It is a set of:
#  - **train** images + segmentation masks
#  - **validation** images + segmentation masks
#  - **test** images + segmentation masks
#  
# All images have 320 pixels height and 480 pixels width.
# For more inforamtion about dataset visit http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/.

# In[3]:
import time
from torch import nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:

begin= time.time()
#DATA_DIR = '/home/sujiwosa/Downloads/gpuvscpu/data/CamVid/'
DATA_DIR = '/home/sujiwosa/Downloads/gpuvscpu/data/lidc_idri_nodule_png_fromDCM/'

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')


# In[5]:


# x_train_dir = os.path.join(DATA_DIR, 'train')
# y_train_dir = os.path.join(DATA_DIR, 'trainannot')

# x_valid_dir = os.path.join(DATA_DIR, 'val')
# y_valid_dir = os.path.join(DATA_DIR, 'valannot')

# x_test_dir = os.path.join(DATA_DIR, 'test')
# y_test_dir = os.path.join(DATA_DIR, 'testannot')

# x_train_dir = os.path.join(DATA_DIR, 'train')
# y_train_dir = os.path.join(DATA_DIR, 'train_cars_bw_labels')

# x_valid_dir = os.path.join(DATA_DIR, 'val')
# y_valid_dir = os.path.join(DATA_DIR, 'val_cars_bw_labels')

# x_test_dir = os.path.join(DATA_DIR, 'test')
# y_test_dir = os.path.join(DATA_DIR, 'test_cars_bw_labels')

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'test_labels')

# In[6]:


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# ### Dataloader
# 
# Writing helper class for data extraction, tranformation and preprocessing  
# https://pytorch.org/docs/stable/data

# In[7]:


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


# In[8]:


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        #print()

        #print(self.images_fps)
        #print(self.masks_fps)
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(self.images_fps[i])
        #print(self.masks_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)
        mask=mask/255.0
        mask=np.expand_dims(mask,axis=2)
        # extract certain classes from mask (e.g. cars)
     #   masks = [(mask == v) for v in self.class_values]
#        print(mask.type)
        #mask = np.stack(mask, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


# In[9]:


# Lets look at data we have

dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])

image, mask = dataset[220] # get some sample
print(image)
print(mask)
print(image.shape)
print(mask.shape)
print(mask.max())
visualize(
    image220=image, 
    groundtruth220=mask.squeeze(),
)


# ### Augmentations

# Data augmentation is a powerful technique to increase the amount of your data and prevent model overfitting.  
# If you not familiar with such trick read some of these articles:
#  - [The Effectiveness of Data Augmentation in Image Classification using Deep
# Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
#  - [Data Augmentation | How to use Deep Learning when you have Limited Data](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
#  - [Data Augmentation Experimentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b)
# 
# Since our dataset is very small we will apply a large number of different augmentations:
#  - horizontal flip
#  - affine transforms
#  - perspective transforms
#  - brightness/contrast/colors manipulations
#  - image bluring and sharpening
#  - gaussian noise
#  - random crops
# 
# All this transforms can be easily applied with [**Albumentations**](https://github.com/albu/albumentations/) - fast augmentation library.
# For detailed explanation of image transformations you can look at [kaggle salt segmentation exmaple](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb) provided by [**Albumentations**](https://github.com/albu/albumentations/) authors.

# In[10]:


import albumentations as albu


# In[11]:


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.RandomCrop(height=384, width=480, always_apply=True),
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
#    print("main:to_tensor")
#    print(x.shape)
    xx=x.transpose(2, 0, 1).astype('float32')
#    print(xx.shape)
    # return x.transpose(2, 0, 1).astype('float32')
    return xx


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# In[12]:


#### Visualize resulted augmented images and masks

#print(len(dataset))
#print("augmented")
augmented_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    classes=['car'],
)
print(len(augmented_dataset))

# same image with different random transforms
for i in range(3):
    image, mask = augmented_dataset[225]
#    print(image.shape)
#    print(mask.shape)
    #visualize(image1=image, mask1=mask.squeeze(-1))


# # ## Create model and train

# # In[13]:
#print("after augmented")

import torch
import numpy as np

import segmentation_models_pytorch as smp
import smp_losses
import smp_metrics
import smp_train
#1- trial run
#2- ablation study
SN='3_'
OPTIMIZER='LBFGS'
BACKBONEMODEL='Unet_'
ENCODER = 'efficientnet-b3'  #to update - encoder
ENCODER_WEIGHTS = 'imagenet'#'imagenet'  'instagram'  #to update
CLASSES = ['car']
INPUTDATASET='NODULE_'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'


FILENAME='./output/'+SN+OPTIMIZER+BACKBONEMODEL+ENCODER+'_'+ENCODER_WEIGHTS+'_'+INPUTDATASET

# create segmentation model with pretrained encoder
model = smp.Unet(    #to update- model
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

parmodel = nn.DataParallel(model, device_ids = [0,1])
#model.to(0)
# In[15]:


train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# valid_dataset = Dataset(
#     x_valid_dir, 
#     y_valid_dir, 
#     augmentation=get_validation_augmentation(), 
#     preprocessing=get_preprocessing(preprocessing_fn),
#     classes=CLASSES,
# )

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)


#dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])

#tt=train_loader[0]
#print("XXXXX")
image, mask = valid_dataset[5]
#print(image.shape)
#print(mask.shape)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
#print(len(train_loader))
#print(len(valid_loader))


#vl=valid_loader[0]

# # In[16]:


# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

#loss = smp.utils.losses.DiceLoss()
loss = smp_losses.DiceLoss()
metrics = [
    #smp.utils.metrics.IoU(threshold=0.5),
    smp_metrics.IoU(threshold=0.5),
    smp_metrics.Fscore(),
    smp_metrics.Accuracy(),
    smp_metrics.Precision(),
    smp_metrics.Recall(),
    
]

optimizer = torch.optim.Adam([ 
    dict(params=parmodel.parameters(), lr=0.0001),
])


# In[17]:


# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
# train_epoch = smp.utils.train.TrainEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     optimizer=optimizer,
#     device=DEVICE,
#     verbose=True,
# )

# valid_epoch = smp.utils.train.ValidEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     device=DEVICE,
#     verbose=True,
# )
train_epoch = smp_train.TrainEpoch(
    parmodel, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp_train.ValidEpoch(
    parmodel, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

#In[19]:


# train model for 40 epochs

max_score = 0
train_logs_list, valid_logs_list = [], []
timetakenlist = []
MDFN='./'+FILENAME+'.pth'
for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    train_logs_list.append(train_logs)
    valid_logs_list.append(valid_logs)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, MDFN)
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
    print("train and valid logs")
    print(train_logs)
    print(valid_logs)

end=time.time()
print(f"Total runtime of the \"train and valid\" program is {end - begin}")
timetaken=end-begin
timetakenlist.append(timetaken)    

train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)
train_logs_df.T


# ## Test best saved model

# In[20]:


# load best saved checkpoint
begin=time.time()

best_model = torch.load(MDFN)


# In[21]:

logs_list=[]
# create test dataset
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)


# In[22]:


# evaluate model on test set
test_epoch = smp_train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)

logs_list.append(logs)
logs_df = pd.DataFrame(logs_list)
end=time.time()
print(f"Total runtime of the program is {end - begin}")
timetaken=end-begin
timetakenlist.append(timetaken)
timetaken_df=pd.DataFrame(timetakenlist)
XLFN=FILENAME+'.xlsx'
with pd.ExcelWriter(XLFN) as writer: 
    train_logs_df.to_excel(writer, sheet_name='Train_logs')
    valid_logs_df.to_excel(writer, sheet_name='Validation_logs')
    logs_df.to_excel(writer, sheet_name='Test_logs')
    timetaken_df.to_excel(writer,sheet_name='Time Taken')



# ## Visualize predictions

# In[23]:


# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
)


# In[24]:


for i in range(5):
    n = np.random.choice(len(test_dataset))
    
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )


plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('IoU Score', fontsize=20)
plt.title('IoU Score Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(FILENAME+'iou_score_plot.png')
plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Dice Loss', fontsize=20)
plt.title('Dice Loss Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(FILENAME+'dice_loss_plot.png')
plt.show()
