import threading
import time
# import numpy as np
import pickle
from utils.ret_utils import error_info
from nodule_class.isnodule import LungIsncls
from preprocessing.location import lobe_locate_gmm
from detection.lung_detection import LungDetection
from func_timeout import FunctionTimedOut
from func_timeout import func_set_timeout
import weakref


class GpuThread(threading.Thread):
    def __init__(self, que_det, que_ret):
        threading.Thread.__init__(self)
        self.que_det = que_det
        self.que_ret = que_ret
        self.lung_dete = LungDetection('./model/det.ckpt')

        # is nodule cls
        self.lung_isnc = LungIsncls('./model/isn.ckpt')

        l_u = pickle._Unpickler(open('./model/left_gmm.pkl', 'rb'))
        l_u.encoding = 'latin1'
        self.left_gmm = l_u.load()

        r_u = pickle._Unpickler(open('./model/right_gmm.pkl', 'rb'))
        r_u.encoding = 'latin1'
        self.right_gmm = r_u.load()
        # cudnn.benchmark = True

    def run(self):
        i = 0
        while True:
            result_dict = self.que_det.get(block=True)
            try:
                t_s = time.time()
                nodule_df = self.lung_dete.prediction(
                    result_dict['prep_data'], result_dict['prep_spac'],
                    result_dict['prep_ebox'], result_dict['prep_mask'])
                print(time.ctime(), ' ', i,
                      'GPU DOING USE TIME(lung dete prediction):',
                      time.time() - t_s)
                t_s = time.time()
                preb = self.lung_isnc.nodule_cls(nodule_df,
                                                 result_dict['prep_case'],
                                                 result_dict['prep_spac'])
                print(time.ctime(), ' ', i,
                      'GPU DOING USE TIME(lung isnc nodule cls):',
                      time.time() - t_s)
                # preb = lung_isnc.nodule_cls(nodule_df, result_dict['prep_case'], result_dict['prep_spac'])
                # del lung_isnc
                t_s = time.time()
                preb = self.lung_lobe(preb, result_dict['prep_mask'])
                result_dict['nodule_preb'] = preb
                self.que_ret.put(result_dict, timeout=2)
                print(time.ctime(), ' ', i, 'GPU DOING US TIME(lung lobe):',
                      time.time() - t_s)
                i += 1
                del result_dict, nodule_df, preb

            except FunctionTimedOut:
                print(time.ctime() + 'FUN TIMEOUT ')
            except Exception as e:
                print(time.ctime() + 'GPU ERROR : {} '.format(e))
                error_info(200, result_dict)

    @func_set_timeout(5)
    def lung_lobe(self, nodule_df, mask):
        nodule_df_values = nodule_df[['coordX', 'coordY', 'coordZ']].values
        lungs = []
        lobes = []

        lobel_info = []
        for nodule in nodule_df_values:
            lung, lobe = lobe_locate_gmm(nodule, mask, self.left_gmm,
                                         self.right_gmm)
            lungs.append(lung)
            lobes.append(lobe)
            lobel_info.append(lung + '肺' +
                              (lobe + '叶' if not lobe == '' else ''))
        nodule_df['lung'] = lungs
        nodule_df['lobe'] = lobes

        nodule_df['lobel_info'] = lobel_info
        return nodule_df
