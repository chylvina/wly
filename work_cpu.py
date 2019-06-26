import threading
import time
from preprocessing.prepare import prepare_data2
from preprocessing import read_dicom
from utils.ret_utils import error_info
from preprocessing.lung_segment import LungSegmentUnet
from multiprocessing import Pool
from func_timeout import FunctionTimedOut


class CpuThread(threading.Thread):
    def __init__(self, que_pre, que_det, cpu_thread_type):
        threading.Thread.__init__(self)
        self.que_pre = que_pre
        self.que_det = que_det
        self.cpu_thread_type = cpu_thread_type
        self.lung_segm = LungSegmentUnet('./model/lung_segment.ckpt', self.cpu_thread_type)
        self.pool = Pool(processes=1)
        # self.pool = Pool(processes=1, maxtasksperchild=10)

    def run(self):
        i = 0
        while True:
            result_dict = self.que_pre.get(block=True)
            try:
                t_s = time.time()
                print(result_dict['data_path'])
                # res = self.pool.apply_async(cpu_preprocess_1, (result_dict,))
                # case, spacing, instances = res.get()
                case, spacing, instances = read_dicom.load_dicom2(
                    result_dict['data_path'])
                print(time.ctime(), ' ', i, ' load dicom use :',
                      time.time() - t_s)
                # assert 40 < case.shape[0] < 80
                # lung_segm = LungSegmentUnet('./model/lung_segment.ckpt')
                t_s = time.time()
                prep_mask = self.lung_segm.cut(case, 20)
                print(time.ctime(), ' ', i, ' lung segment cut use :',
                      time.time() - t_s)
                # prep_mask = lung_segm.cut(case, 30)
                # del lung_segm
                # print('cat  us :', time.time() - t_s)
                t_s = time.time()
                res = self.pool.apply_async(prepare_data2,
                                            (case, spacing, prep_mask))
                prep_data, extendbox = res.get(timeout=20)

                # prep_data, extendbox = prepare_data2(case, spacing, prep_mask)
                result_dict['prep_case'] = case
                result_dict['prep_spac'] = spacing
                result_dict['prep_inst'] = instances
                result_dict['prep_data'] = prep_data
                result_dict['prep_mask'] = prep_mask
                result_dict['prep_ebox'] = extendbox
                print(
                    time.ctime(), ' ', i,
                    ' cpu process task us time: {}.{}'.format(
                        time.time() - t_s, ' result dict: ',
                        result_dict['data_path']))
                self.que_det.put(result_dict)
                i += 1
                del result_dict, case, spacing, prep_mask, prep_data, extendbox
            except FunctionTimedOut:
                print(time.ctime() + 'GPU FUN TIMEOUT ')
            except Exception as e:
                print("CPU ERROR:", " ", i, e)
                error_info(100, result_dict)


#
# if __name__ == '__main__':
#     case, spacing, instances = read_dicom.load_dicom2('/home/wly/log/1.2.392.200036.9116.2.6.1.3268.2060219908.1561080241.620462')
#     # case, spacing, instances = read_dicom.load_dicom2('/home/wly/log/1.2.392.200036.9116.2.5.1.3268.2054276094.1561070139.404229')
#     lung_segm = LungSegmentUnet('./model/lung_segment.ckpt')
#     prep_mask = lung_segm.cut(case, 2)
#     # prepare_data2(case, spacing, prep_mask)
#     pool = Pool(processes=1)
#     res = pool.apply_async(prepare_data2, (case, spacing, prep_mask))
#     prep_data, extendbox = res.get(timeout=20)
#     print(prep_data.shape)
#     from matplotlib import pyplot as plt
#
#     plt.subplot(211)
#     plt.title('INPUT IMAGE')
#     plt.imshow(prep_data[0,200])
#     plt.subplot(212)
#     plt.title('INPUT IMAGE')
#     plt.imshow(prep_data[0,10])
#     plt.show()
