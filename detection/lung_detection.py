from detection import res18
from detection import split_combine
import torch
from torch.nn import DataParallel
from torch.autograd import Variable
import numpy as np
from utils.det_utils import nms
from nodule_class.lung_isncls import pbb_to_df
from func_timeout import func_set_timeout
from detection.data import data_loader
import time


class LungDetection(object):
    def __init__(self, model_path):
        max_stride = 16
        margin = 32
        stride = 4
        sidelen = 144
        pad_value = 170
        self.split_comber = split_combine.SplitComb(sidelen, max_stride, stride, margin, pad_value)

        # detection net
        config1, nod_net, loss, get_pbb = res18.get_model()
        checkpoint = torch.load(model_path)
        nod_net.load_state_dict(checkpoint)
        # chylvina
        nod_net = DataParallel(nod_net).cuda()
        nod_net.eval()
        self.nod_net = nod_net
        self.get_pbb = get_pbb

    @func_set_timeout(20)
    def prediction(self, imgs, spacing, endbox, mask):
        stride = 4
        pad_value = 170
        imgs, coord, nzhw = data_loader(imgs, stride, pad_value, self.split_comber)

        splitlist = range(0, len(imgs) + 1, 1)
        if splitlist[-1] != len(imgs):
            splitlist.append(len(imgs))
        outputlist = []
        print(time.ctime(), ' Lung Detection has ',len(splitlist),' Imgs')
        for i in range(len(splitlist) - 1):
            # img: torch.Size([1, 1, 208, 208, 208])
            input = Variable(imgs[splitlist[i]:splitlist[i + 1]]).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
            output = self.nod_net(input, inputcoord)
            # print('out_put:',output.data.cpu().numpy().shape)
            outputlist.append(output.data.cpu().numpy())
            # chylvina
            torch.cuda.empty_cache()
            del output
        output = np.concatenate(outputlist, 0)

        output = self.split_comber.combine(output, nzhw=nzhw)
        thresh = -3
        pbb, _ = self.get_pbb(output, thresh, ismask=True)
        torch.cuda.empty_cache()
        pbb = nms(pbb, 0.05)
        nodule_df = pbb_to_df(pbb, spacing, endbox, mask)
        nodule_df = nodule_df[nodule_df.probability > 0.25]
        # print(nodule_df)
        return nodule_df
