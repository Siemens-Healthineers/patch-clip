import os
import pprint
import argparse
from tqdm import tqdm

import torch
import numpy as np
from torch.utils import data
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms

import torch.distributed as dist
import torch.backends.cudnn as cudnn

from model import  load_concept_model, load_pretrained_model, LocalConceptLoss
from torch.utils.data import WeightedRandomSampler

from train import preprocess_text, CXRFinetuneDataset 
from zero_shot import CXRTestDataset
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, auc, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from data_process import load_single_image

'''
    This will finetune the clip wrapper model that was pretrained. The wrapper model already aligns the local combination of 
    patch embeddings to the text. 
    We expect further fintuning to give better localization compared to the pre-trained model. 
    elect model_version as 0 if we don't want to use the help of additional LR labels 
    Select the model_version as 1 if we want to use the help of additional LR labels along with patch labels. this intializes 
    the LR feature embeddings which are also used for the loss. 

    set model_version = 1 if we want to use LR labels for finetuning. this will copy all the common layers to finetune model and retrain the Linear reduction (missing layers)
    model version = 0 CLipWrapper, same model as pretrained , no changes to self pretrained
    model version = 1 ClipWrapper_LRHead, additional LR labels, changes made to self pretrained arch head
    model version = 3 Original ChexZero basemodel , ClipWrapper, input image 1024, downscales,  doesnt use LR labels
    model version = 4 Original ChexZero basemodel, ClipWrapper_LRHead, uses LR labels @TODO
    model_version = 5 Original ChexZero basemodel, ClipWrapper_v5, input image 224 resolution, no downscaler ClipWrapper_v5

'''
model_version=0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_datapath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--train_csvpath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--val_datapath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--val_csvpath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--test_datapath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--test_csvpath', type=str, default='../../xxxx.txt', help="Filename of patchwise labels for validation file.") 
    parser.add_argument('--summary_file', type=str, default='report.csv', help="File for output summary")     
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt")
    parser.add_argument('--checkpoint', type=str, default=None, help="the name of the checkpoint file")
    parser.add_argument('--start_model', type=str, default="")
    parser.add_argument('--concept', type=str, default="")
    parser.add_argument('--arch',type=str, default='vitb32')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--world_size", default=-1, type=int, help=""" number of processes: it is set automatically and should not be passed as argument""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process: it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int, help="this argument is not used and should be ignored")

    args = parser.parse_args()
    return args


import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:        
        for i in range(32):
            elem = batch[i] 
            print(f"Error in collate_fn for batch: {elem['idx']}")
            print(elem['img'].size(), elem['lbl'], elem['txt'], elem['plbl'])
        print(f"Error in collate function, there's a problem with this batch: details: {e}")
        return None

def model_pipeline(config, verbose=0): 
    world_size = os.environ["WORLD_SIZE"]
    rank = os.environ["RANK"]  
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    device_id = local_rank % torch.cuda.device_count()
    cudnn.benchmark = True

    # make the model, data, and optimization problem
    model, data_loader, val_data_loader, test_data_loader, criterion, optimizer = make(config, device_id)

    # and use them to train the model
    finetune(model, data_loader, val_data_loader, test_data_loader, device_id, criterion, optimizer, config)
    dist.destroy_process_group()

    return model

def make(config, device_id): 
    pretrained = not config.random_init
    if dist.get_rank()==0:
        print("The checkpoint used", config.checkpoint)
    #define the input resolution
    if model_version==5:
        input_resolution = 224 
    else: 
        input_resolution = 900
    clip_input_resolution = 224

    transform = {
        'train' : transforms.Compose([  
                transforms.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
                transforms.Resize(input_resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                ]),
        'val' : transforms.Compose([  
                transforms.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
                transforms.Resize(input_resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                ])
    }

    from train import CXRDatasetBlore
    train_dset = CXRDatasetBlore(img_path=config.train_datapath, txt_path=config.train_csvpath, column=config.concept, size=None, transform=transform['train'])
    val_dset = CXRTestDataset(img_path=config.val_datapath, label_list=config.val_csvpath, transform=transform['val'])
    test_dset = CXRTestDataset(img_path=config.test_datapath, label_list=config.test_csvpath, transform=transform['val'])

    train_sampler = WeightedRandomSampler(train_dset, num_replicas=4, rank=device_id, replacement=True, shuffle=True)
    data_loader = data.DataLoader(train_dset, shuffle=False, pin_memory=False, num_workers=4, batch_size=config.batch_size, sampler=train_sampler, drop_last=False) 
    val_data_loader = data.DataLoader(val_dset, shuffle=False, pin_memory=False, num_workers=4, batch_size=1, drop_last=True) 
    test_data_loader = data.DataLoader(test_dset, shuffle=False, pin_memory=False, num_workers=4, batch_size=1, drop_last=True) 

    model = load_pretrained_model(model_path=config.checkpoint, context_length=config.context_length, \
                                concept_name=config.concept, input_resolution=input_resolution, clip_input_resolution=clip_input_resolution)      

    model.to(device_id)
    model.float()

    if not config.evaluate:
        model.local_clip.eval() 
        model.local_clip.requires_grad_(False) 
        model.local_clip.visual.out_spatial_features = True
        model.concept_proj.train()
        model.conv_downsample.train()
        model.local_clip.visual.proj.requires_grad = True            

    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    criterion = LocalConceptLoss() 

    # make the optimizer 
    if config.optimizer == "adam": 
        optimizer = optim.Adam(ddp_model.parameters(), lr=config.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=config.wd)
    elif config.optimizer == "sgd": 
        #optimizer = optim.SGD(ddp_model.parameters(), lr=config.lr, momentum=config.momentum)
        optimizer = optim.SGD(
            [{'params': ddp_model.module.local_clip.parameters(), 'lr': config.lr},
            {'params': ddp_model.module.conv_downsample.parameters(), 'lr': 0.01},
            {'params': ddp_model.module.concept_proj.parameters(), 'lr': 0.01}],
            lr=config.lr, momentum=config.momentum)


    return ddp_model, data_loader, val_data_loader, test_data_loader, criterion, optimizer

def calculate_classification_metrics(y_gt, y_pred, global_thresh):
    
    try:
        auc = roc_auc_score(y_gt, y_pred, average='weighted')
        fpr, tpr, thresholds = roc_curve(y_gt, y_pred, drop_intermediate=False)
        index = np.argwhere(tpr>=global_thresh)
        selected_thresh = thresholds[index[0]] 
        y_pred[y_pred>=selected_thresh] = 1
        y_pred[y_pred<selected_thresh] = 0

        accuracy = balanced_accuracy_score(y_gt, y_pred)
        precision = precision_score(y_gt, y_pred, average='weighted')
        recall = recall_score(y_gt, y_pred, average='weighted')    
        f1 = f1_score(y_gt, y_pred, average='weighted')
    except Exception as e:
        print("Caught exception at classification metrics as ", e)
        print("Exception: debug: sum of gt ",np.sum(y_gt), y_gt.shape, y_gt[y_gt==1])
        return 0, 0, 0, 0, 0

    return auc, accuracy, precision, recall, f1


def evaluate(model, loader, device_id, criterion, epoch, config, ax, filename, isval=True):
    model.eval()
    cxr_pair_template: Tuple[str] = ["{}".format(config.concept), "no {}".format(config.concept)]
    tmp_sm = torch.nn.Softmax(dim=1)
    y_pred, y_true, p_pred, y_pred_patch = [], [], [], []
    sample_ct = 0

    for data in tqdm(loader):
        if data is not None:               
            images = data['img']
            sample_ct += len(images)

            texts = preprocess_text(cxr_pair_template, model.module.local_clip) #for concept
            img_labels = data['lbl']
            patch_labels = data['plbl']
            y_true.append(img_labels.cpu().numpy())
            for m in range(len(patch_labels)):
                  patch_labels[m] = patch_labels[m].to(device_id, non_blocking=True)



            images, texts, img_labels = images.to(device_id, non_blocking=True), texts.to(device_id, non_blocking=True), img_labels.to(device_id, non_blocking=True)
            with torch.no_grad(): 
                logits_per_image, logits_per_text, logits_from_patch_all, sparse_logits_from_patch_all, p_logits_per_image  = model(images, texts, patch_labels, False) #for concept

                pos_pred_img = tmp_sm(logits_per_image)[:,0] #img2txt global
                pos_pred_txt = tmp_sm(logits_per_text)[0, 0] #txt2img_global
                pos_pred_img_all = tmp_sm(logits_from_patch_all)[:,0]
                sparse_pred_img_all = tmp_sm(sparse_logits_from_patch_all)[:,0]
                pos_pred = (pos_pred_img.cpu().numpy() + pos_pred_txt.cpu().numpy())/2 #+ pos_pred_img_all.cpu().numpy() + sparse_pred_img_all.cpu().numpy())/4
                #y_pred_patch.append(pos_pred_img_all.cpu().numpy())
                y_pred.append(pos_pred)
         
                #calculate patch predictions  
                num_patches = len(p_logits_per_image)       
                for k in range(len(p_logits_per_image)):
                    pos_p_pred = tmp_sm(p_logits_per_image[k])[:,0]
                    pos_p_pred_flatten = pos_p_pred.cpu().numpy().reshape(-1,1)
                    p_pred.append(pos_p_pred_flatten)        
        else: 
            print("Returned data is None")        

        torch.cuda.synchronize() 

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    p_pred = np.concatenate(p_pred, axis=0)

    #Add all the confidence metrics here  
    auc, acc, prec, recall, f1 =  calculate_classification_metrics(y_true, y_pred, 0.90)    
    print("classification metrics are done")

    return auc

def finetune(model, loader, val_data_loader, test_data_loader, device, criterion, optimizer, config): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)   
    # Run training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0 # save highest mean auc

    pp_texts = preprocess_text(model.module.text_pair, model.module.local_clip)
    fig, ax =  plt.subplots(1,1,figsize=(10,8))

    if config.evaluate:
        if dist.get_rank()==0:
            auc = evaluate(model, val_data_loader, device, criterion, 0, config, ax, config.val_csvpath)

    
    else: #proceed to finetuning
        for epoch in tqdm(range(config.epochs)):
            for data in tqdm(loader):
              if data is not None:
                # get the images
                images = data['img'] #this is proportional to batch size
                labels = data['lbl'] #this is batch x 1 0 = noPE, 1 = PE
                patch_labels = data['plbl']  # this is list of patch numbers that are the center of effusion

                # perform step for a single batch
                loss, loss_global, loss_local, loss_patches = finetune_batch(images, pp_texts, labels, patch_labels, model, device, criterion, optimizer)
            
            if dist.get_rank()==0 and epoch >=0:      
                val_auc = evaluate(model, val_data_loader, device, criterion, epoch, config, ax, config.val_csvpath, isval=True)    #This works because we dont use a distributed data sampler, only a weighted sampler would be fine       
                current_auc = val_metrics[1]
                if current_auc > highest_val_auc:
                    highest_val_auc = current_auc
                    test_auc = evaluate(model, test_data_loader, device, criterion, epoch, config, ax , config.test_csvpath,isval=False) 
                    

                model_path = os.path.join(model_save_dir, "ckpt_finetune_{}.pt".format(epoch+1))
                save(model, model_path)   
    
    
def finetune_batch(images, texts, labels, patch_labels, model, device, criterion, optimizer):

    model.train()
    
    torch.autograd.set_detect_anomaly(True) 
    images, texts, labels = images.to(device, non_blocking=True), texts.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    for m in range(len(patch_labels)):
        patch_labels[m] = patch_labels[m].to(device, non_blocking=True)
    
    if model_version==0 or model_version==3 or model_version==5:
        # Forward pass ➡
        logits_per_image, logits_per_text, logits_from_patch_all, sparse_logits_from_patch_all, logits_from_each_patch = model(images, texts, patch_labels)
        loss, loss_global, loss_local, loss_patches = criterion(logits_per_image, logits_from_patch_all, sparse_logits_from_patch_all, logits_from_each_patch, labels, patch_labels, device)
    elif model_version==1:
        # Forward pass ➡
        logits_per_image, lr_logits_per_image, p_logits_per_image = model(images, texts)
        # Compute loss
        loss, loss_global, loss_local, loss_patches = criterion(logits_per_image, lr_logits_per_image, p_logits_per_image, labels, patch_labels, device)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
        
    return loss, loss_global, loss_local, loss_patches

def train_log(loss, example_ct, epoch, text='Loss'):
    loss = float(loss)
    print(f"{text} after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def save(model, path): 
    torch.save(model.module.state_dict(), path)

import csv
if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)




