import torch
from path import Path
import fastai
from fastai import *
from fastai.vision import *

# file path within the project
data_fp = Path('data')
data_train = data_fp/'train'
data_test = data_fp/'test'
sub_fp = Path('submission') 

def mapr(input: torch.Tensor, targs: torch.LongTensor, mapn: int):
    """
        Compute the mean average precision
    
        > map5 = partial(mapr, mapn=5)
    """
    n = targs.shape[0]  # number for samples
    input = input.argsort(dim=-1, descending=True)[:,:mapn]
    targs = targs.view(n, -1)
    return ((input == targs).float()/torch.arange(1,mapn+1, device=input.device).float()).sum(dim=-1).mean()




def get_data(size, bs, csv, num_workers=0, padding_mode='reflection'):
    """
        Consistent way to get data for experimenting
    """
    labels = pd.read_csv(data_fp/csv)  # obtain the index for validation set
    val_idx = labels.index[labels.validation].to_list()
    
    tfms = get_transforms(flip_vert=False, max_zoom=1)  ## remove vertical and zooming
    
    src = (ImageList.from_df(path=data_fp, df=labels, cols='Image', folder='train')
                     # images' filepath are in a dataframe with column name 'Image'
                    .split_by_idx(val_idx)
                    # validations are not random and determined by the row indices
                    .label_from_df(cols='Id')
                    # classes for the images are in a dataframe with column name 'Id'
                    .add_test_folder())
                    # images to be use for inferences to the kaggle competition
        
    return (src.transform(tfms, 
                          size=size,
                          resize_method=ResizeMethod.PAD,
                          padding_mode=padding_mode)
                .databunch(bs=bs, num_workers=num_workers)
                # creates a dataloader
                .normalize(imagenet_stats))
                # normalize the whale images with imagenet's mean and std because we are using a pretrained model