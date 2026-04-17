import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from PIL import Image
import h5py

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
import scipy.signal as ss
import scipy.ndimage as snd

import sys
sys.path.append('../..')

import clip
from model import CLIP, ClipWrapper, load_concept_model
from simple_tokenizer import SimpleTokenizer
import skimage.transform as skt


class RandomResize(object):
    def __init__(self, size=[256, 256], border=[0, 0]):
        self.size = size
        self.border = border

        assert self.border[0] < 0.2 and self.border[1] < 0.2

    def __call__(self, img):
        #nSize = random.randint(self.size[0], self.size[1])
        nSize = self.size[0]
        #nSize = min(nSize, min(img.shape[0], img.shape[1]))
        #nSize = max(nSize, 224) # temporary hack

        nR = int(self.border[0]*img.shape[0])
        nC = int(self.border[1]*img.shape[1])

        img = img[nR:img.shape[0] - nR, nC:img.shape[1]-nC]
        return skt.resize(img, (nSize, nSize), anti_aliasing=True, mode='constant')
    
class ToTensor(object):
    def __call__(self, img):
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img = np.repeat(img, 3, axis=2)

        img = torch.from_numpy(img).float()
        img = img.transpose(0, 2)
        img = img.transpose(1, 2) #3, w, x

        return img
    
   
class CXRFinetuneDataset(data.Dataset):

    def __init__(self, img_path, txt_path, column='report', size=None, transform=None, concept_mode = True, trainmode=True):
        super().__init__()
        self.do_lr = True
        self.input_file_dir = img_path     
        
        with open(txt_path,'r') as f:
            self.txt_dset = f.readlines()
        self.img_label_list = []   
        for line in self.txt_dset:
            line = line.strip().split(',')
            self.img_label_list.append(int(line[1]))
        self.trainmode = trainmode

        self.transform = transforms.Compose([RandomResize(size=[900, 900], border=[0, 0]),
                                             Normalize(),
                                             ToTensor()])
        self.concept_mode = concept_mode
        self.label_limit = 180
            
    def __len__(self):
        return len(self.txt_dset)
    
    def readImage (self, fname):
            ffname = os.path.join(self.input_file_dir, fname)
            #read image 
            img = np.asarray(dcm.pixel_array)
 
            ##Data preprocessing       
            return img
    
    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            line = self.txt_dset[idx].strip().split(',') 
            #img = self.img_dset[idx] # np array, (320, 320)
            #read image in dicom format 
            imname = line[0]
            lbl = int(line[1])
            assert lbl in [0, 1]

            img = self.readImage(imname)      
            plbl = []   #list of patch numbers that have the finding
 
            if (line[1] == '1'):
                if self.trainmode == False:                
                    if (line[2] == 'R' or (line[2] =='L' and line[3] != 'R')): #patch only one side 
                        for i in range(len(line)-7):
                            plbl.append(int(line[7+i]))
                    elif (line[2] == 'L' and line[3] == 'R'):
                        for i in range(len(line)-11):
                            plbl.append(int(line[11+i]))
                else:                     
                    for i in range(len(line)-3):
                            plbl.append(int(line[3+i]))
            else: 
                plbl.append(0)

            txt = " "
            if lbl == 1:
                txt = "Finding"
            else:
                txt = "No finding"

            #img = torch.from_numpy(img) # torch, (3, 320, 320)
            if self.transform:
                img = self.transform(img)     
                
            
            if self.concept_mode:
                if (len(plbl)<self.label_limit): #this is needed to make all the batch plbls the same length, set to max
                    #print("Landed in length of plbl < 4, append 0")
                    for m in range(self.label_limit-len(plbl)):
                        plbl.append(0)
                elif (len(plbl)>=self.label_limit):
                    print("Length of label list > {}".format(self.label_limit))
                    print(line[0], plbl," print the patch labels for image")
                    print("Length of the label",len(plbl)) 
            
            sample = {'img': img, 'txt': txt, 'lbl': lbl, 'plbl': plbl, 'idx': idx}
            #print(sample, len(plbl))
            #assert img.shape == (900, 900), f"Image {imname} at index {idx} has an unexpected size: {img.shape}"
            #assert len(plbl) == label_limit, f"Length of plbl for image {imname} is not equal to {label_limit}"
            

        except AssertionError:
            print("Label assert threw and error for index ",idx)
            return None
        except RuntimeError:
            print("Experiennced a runtime error, but want to continue")
            idx = 1
            line = self.txt_dset[idx].strip().split(',') 
            imname = line[0]
            lbl = 0
            txt = "No finding"
            img = self.readImage(imname) 
            if self.transform:
                img = self.transform(img) 
            plbl = [] 
            plbl.append(0)
            if (len(plbl)<self.label_limit): #this is needed to make all the batch plbls the same length, set to max
                    #print("Landed in length of plbl < 4, append 0")
                    for m in range(self.label_limit-len(plbl)):
                        plbl.append(0)

            sample = {'img': img, 'txt': txt, 'lbl': lbl, 'plbl': plbl, 'idx': idx}
            return sample 
        except Exception as e:
            print("Exception: MyException error message ", e , idx)
            return None

        
        return sample
    
class CXRDatasetBlore(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='report', size=None, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        with open(txt_path,'r') as f:
            self.txt_dset = f.readlines()

        self.img_label_list = []
        for line in self.txt_dset:
            line = line.strip().split(',')
            self.img_label_list.append(int(line[1]))

        self.transform = transform
        self.label_limit = 500
    def __len__(self):
        #return 300000
        return len(self.txt_dset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)

        img = torch.from_numpy(img) # torch, (3, 320, 320)
        if self.transform:
            img = self.transform(img)

        line = self.txt_dset[idx].strip().split(',')
        lbl = int(line[1])
        plbl = []

        if (line[1] == '1'):
             if ((line[2] == 'L' and line[3] !='R' ) or line[2] == 'R'):
                 for i in range(len(line)-3):
                     plbl.append(int(line[3+i]))
             else:
                 for i in range(len(line)-4):
                     plbl.append(int(line[4+i]))
        else:
                plbl.append(0)
        if lbl == 1:
                txt = "Finding"
        else:
                txt = "No finding"

        if True:
                if (len(plbl)<self.label_limit): #this is needed to make all the batch plbls the same length, set to max
                    #print("Landed in length of plbl < 4, append 0")
                    for m in range(self.label_limit-len(plbl)):
                        plbl.append(0)
                elif (len(plbl)>=self.label_limit):
                    print("Length of label list > {}".format(self.label_limit))
                    print(line[0], plbl," print the patch labels for image")
                    print("Length of the label",len(plbl))

        sample = {'img': img, 'txt': txt, 'lbl': lbl, 'plbl': plbl, 'idx': idx}

        return sample

class CXRDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='report', size=None, transform=None):
        super().__init__() 
        self.img_dset = h5py.File(img_path, 'r')['cxr']        
        with open(txt_path,'r') as f:
            self.txt_dset = f.readlines()

        self.img_label_list = []   
        for line in self.txt_dset:
            line = line.strip().split(',')
            self.img_label_list.append(int(line[1]))

        self.transform = transform
        self.label_limit = 500
            
    def __len__(self):
        #return 300000
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        #txt = self.txt_dset[idx] # python str
        #if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
        #    txt = " "

        img = torch.from_numpy(img) # torch, (3, 320, 320)
        if self.transform:
            img = self.transform(img)

        line = self.txt_dset[idx].strip().split(',') 
        lbl = int(line[1])
        plbl = []   
 
        if (line[1] == '1'):
            nbox = int(line[2])
            for i in range(len(line)-3-3*nbox):
               plbl.append(int(line[3+3*nbox+i]))

            #for i in range(len(line)-7):
            #   plbl.append(int(line[7+i]))
            # if ((line[2] == 'L' and line[3] !='R' ) or line[2] == 'R'):
            #     for i in range(len(line)-3):
            #         plbl.append(int(line[3+i]))
            # else:
            #     for i in range(len(line)-4):
            #         plbl.append(int(line[4+i]))
        else: 
                plbl.append(0)


            
        if lbl == 1:
                txt = "Finding"
        else:
                txt = "No finding"    
                
            
        if True:
                if (len(plbl)<self.label_limit): #this is needed to make all the batch plbls the same length, set to max
                    #print("Landed in length of plbl < 4, append 0")
                    for m in range(self.label_limit-len(plbl)):
                        plbl.append(0)
                elif (len(plbl)>=self.label_limit):
                    print("Length of label list > {}".format(self.label_limit))
                    print(line[0], plbl," print the patch labels for image")
                    print("Length of the label",len(plbl)) 

        sample = {'img': img, 'txt': txt, 'lbl': lbl, 'plbl': plbl, 'idx': idx}

        #sample = {'img': img, 'txt': txt }
        
        return sample

def load_data(cxr_filepath, txt_filepath, batch_size=4, column='report', pretrained=False, verbose=False): 
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    if pretrained: 
        input_resolution = 224
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        print("Finished image transforms for pretrained model.")
    else: 
        input_resolution = 1024
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])
        print("Finished image transforms for CLIP model.")
    
    torch_dset = CXRDataset(img_path=cxr_filepath,txt_path=txt_filepath, column=column, transform=transform)
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample['img'][0])
            plt.show()
            print(i, sample['img'].size(), sample['txt'])
            if i == 3:
                break
    
    loader_params = {'batch_size':batch_size, 'shuffle': True, 'num_workers': 0}
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device

def load_clip(model_path=None, pretrained=False, context_length=77, concept_name="Effusion", input_resolution=1024, clip_input_resolution=320):

    params = {
        'embed_dim':768,
        'image_resolution': 320,
        'vision_layers': 12,
        'vision_width': 768, #check if this should be 512?
        'vision_patch_size': 16,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if pretrained: 
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else: 
        model = CLIP(**params)
        print("Loaded in clip model.")
    
    from model import load_concept_model
    # if a model_path is provided, load in weights to backbone
    concept_model = ClipWrapper(model, concept_name, input_resolution=input_resolution, clip_input_resolution=clip_input_resolution)
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))        
        #Uncomment below if loading a pretrained Clip Wrapper model
        #concept_model = load_concept_model(model_path, concept_name, context_length=context_length, input_resolution=input_resolution, clip_input_resolution=clip_input_resolution)
    return concept_model
    #return model 
    
    
def preprocess_text(texts, model):
    if model.context_length is None: 
        model = model.module    
       
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def make(config, cxr_filepath, txt_filepath, model_path=None): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer. 
    
    args: 
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
    '''
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=config.batch_size, pretrained=config.pretrained, column=config.column)
    model = load_clip(model_path=model_path, pretrained=config.pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    # todo: incorporate - torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return model, data_loader, device, criterion, optimizer


def train_main(cxr_filepath, txt_filepath, hyperparams, output_path, model_path=None, pretrained=False): 
    '''
    args: 
        * cxr_filpath- str filepath to cxr images
        * txt_filepath- str filepath to text reports
        * hyperparams- dictionary with the following hyperparams:
        `batch_size`, `criterion`, `learning_rate`, `momentum`, `epochs`
        * output_path- str filepath to where the trained model will be saved
        * model_path- str filepath to model that will be used as baseline model for training. 
        If not provided, a model will be trained from scratch
        * pretrained- whether or not the clip model was pretrained with generic images 
    This function is the main train function for CXR-CLIP. 
    '''
    
    # unpack `hyperparams`
    batch_size = hyperparams['batch_size']
    criterion = hyperparams['criterion']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    epochs = hyperparams['epochs']
    
    # load input cxr + report data
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=batch_size, pretrained=pretrained)
    model = load_clip(model_path=model_path, pretrained=pretrained)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_clip(model, data_loader, device, criterion, optimizer, epochs, output_path)
    return model
