"https://github.com/leido/pytorch-prednet/blob/master/kitti_data.py"

import hickle as hkl
import torch
import torch.utils.data as data

class KITTI(data.Dataset):
    def __init__(self, datafile, sourcefile, size, mode='all'):
        self.datafile = datafile
        self.sourcefile = sourcefile
        self.X = hkl.load(self.datafile)  
        self.sources = hkl.load(self.sourcefile) 
        self.size = size   # batch_size
        self.mode = mode  
        cur_loc = 0
        possible_starts = []
        while cur_loc < self.X.shape[0] - self.size + 1:
            if self.sources[cur_loc] == self.sources[cur_loc + self.size - 1]:
                possible_starts.append(cur_loc)
                if mode=='all':
                    cur_loc += 1
                elif mode=='unique':
                    cur_loc += self.size
                else:
                    prisize('Error. Wrong Mode')
                    break
            else:
                cur_loc += 1
        self.possible_starts = possible_starts
    def __getitem__(self, index):
        loc = self.possible_starts[index]
        return self.X[loc:loc+self.size]
    def __len__(self):
        return len(self.possible_starts)
