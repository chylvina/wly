from nodule_class.conv6 import Net
import torch
import pandas as pd
import collections
import numpy as np
from nodule_class.dataset import TestCls_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from func_timeout import func_set_timeout
from torch.nn import DataParallel


def collate(batch):
    if torch.is_tensor(batch[0]):
        out = None
        return torch.cat(batch, 0, out=out)
    if isinstance(batch[0], pd.DataFrame):
        return pd.concat(batch)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

class LungIsncls(object):
    def __init__(self, model_path, index):
        self.index = index
        isn_net = Net()
        isn_net.load_state_dict(torch.load(model_path))
        isn_net = isn_net.cuda(self.index)
        isn_net.eval()
        self.isn_net = isn_net

    @func_set_timeout(10)
    def nodule_cls(self, nodule_df, case, spacing):
        dataset = TestCls_Dataset(nodule_df, case, spacing, sample_num=32)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=False)
        softmax = torch.nn.Softmax()
        probabilities_list = []
        candidates_list = []
        for i, (data, cands) in enumerate(data_loader):
            # torchvision.utils.save_image(data[:, :, :, :, 10], 'batch_%d.png' % i)
            data = Variable(data).cuda(self.index)
            output = self.isn_net(data)
            probs = softmax(output).data[:, 1].cpu().numpy()
            probabilities_list.append(probs)
            candidates_list.append(cands)
            del probs, cands, output

        probabilities = np.concatenate(probabilities_list)
        candidates = pd.concat(candidates_list)
        if 'probability' in candidates.columns:
            p_1 = candidates['probability'].values
        else:
            p_1 = 1
        candidates['probability2'] = probabilities
        candidates['probability3'] = probabilities * p_1

        candidates = candidates[candidates.probability2 > 0.12]
        candidates = candidates[candidates.probability3 > 0.25]

        if len(candidates) > 6:
            candidates = candidates.iloc[0:6, :]

        del dataset, data_loader,probabilities
        return candidates
