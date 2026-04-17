
# Patch-CLIP

## 🏥 Overview

Patch-CLIP is based on CLIP open source implementation. It introduces additional loss correlating patch token embeddings to text, similar to the global image-text contrastive loss. 

## To finetune 

torchrun --nnodes=1 --nproc_per_node=1 finetune.py \
    --epochs 10 \
    --batch_size 64 \
    --optimizer "sgd" \
    --lr 0.00001 \
    --context_length 77 \
    --save_interval 100 --log_interval 10 \
    --train_datapath "" \
    --train_csvpath "" \
    --val_datapath "" \
    --val_csvpath "" \
    --test_datapath "" \
    --test_csvpath "" \
    --checkpoint "" \
    --arch 'vitl14' \
    --summary_file "" \
    --model_name  \
    --concept "Pleural Effusion"

## To evaluate 

torchrun --nnodes=1 --nproc_per_node=1 finetune.py \
    --epochs 10 \
    --batch_size 64 \
    --evaluate \
    --optimizer "sgd" \
    --lr 0.00001 \
    --context_length 77 \
    --save_interval 100 --log_interval 10 \
    --train_datapath "" \
    --train_csvpath "" \
    --val_datapath "" \
    --val_csvpath "" \
    --test_datapath "" \
    --test_csvpath "" \
    --checkpoint "" \
    --arch 'vitl14' \
    --summary_file "" \
    --model_name  \
    --concept "Pleural Effusion"


### Prerequisites

- Install packages in the requirements.txt

### Getting Help
- Please email sheethal.bhat@fau.de

## License

This software is licensed under the GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 license.

## Acknowledgments

We thank the authors of CheXZero https://github.com/rajpurkarlab/CheXzero/tree/main
Tiu, E., Talius, E., Patel, P. et al. Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning. Nat. Biomed. Eng (2022). https://doi.org/10.1038/s41551-022-00936-9
---

**Ready to get started?** Run the installation script and visit the setup wizard to configure your installation!

