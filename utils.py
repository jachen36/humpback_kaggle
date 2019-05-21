import torch
from path import Path

# file path within the project
data_fp = Path('data')
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