import threading
import time
import numpy as np
import os
import shutil
from utils.ret_utils import success_ret_info
from utils.ret_utils import error_info
import requests
from multiprocessing import Pool
from functools import partial
import json

requests.adapters.DEFAULT_RETRIES = 5


class DownThread(threading.Thread):
    def __init__(self, que_get, que_pre):
        threading.Thread.__init__(self)
        self.que_get = que_get
        self.data_path = '/tmp/data/'
        os.makedirs(self.data_path, exist_ok=True)
        self.que_pre = que_pre
        self.pool = Pool(processes=10)
        # these variable create for log
        self.queueGetErrCnt = 0
        self.queuePutErrCnt = 0

    def run(self):
        i = 0
        while True:
            que_gsize = self.que_get.qsize()
            que_psize = self.que_pre.qsize()

            if que_psize < 3 and que_gsize > 0:
                try:
                    print(time.ctime(), ' ', i, ' Download operation start')
                    result_dict = self.que_get.get(block=True)
                    series = result_dict['series']
                    # print([ser['windowCenter'] for ser in series])
                    windC = [
                        int(ser['windowCenter'].split('\\')[0]) < 0
                        for ser in series
                    ]
                    # print('windC:',windC)
                    if sum(windC):
                        ser = series[windC.index(True)]
                        s_path = os.path.join(self.data_path, ser['seriesUid'])
                        if os.path.exists(s_path):
                            shutil.rmtree(s_path)
                            os.mkdir(s_path)
                        else:
                            os.mkdir(s_path)
                        # down_pool = Pool(processes=5)
                        # for i, file in enumerate(ser['files']):
                        #     # img_name = os.path.join(s_path, file['imageUid'])
                        #     # downloading(file['url'], img_name)
                        #     img_name = os.path.join(s_path, file['imageUid'])
                        print(time.ctime(), ' ', i,
                              ' Download multiprocessing start')
                        t_s = time.time()
                        self.pool.map(
                            partial(downloading, path=s_path), ser['files'])
                        print(time.ctime(), ' Download operation cost',
                              time.time() - t_s)
                        # down_pool.close()
                        # down_pool.join()
                        result_dict['seriesUid'] = ser['seriesUid']
                        result_dict['data_path'] = s_path
                        try:
                            self.que_pre.put(result_dict)
                        except Exception as e:
                            self.queuePutErrCnt += 1
                            print(time.ctime(), ' ', i, ' Get',
                                  self.queuePutErrCnt, ' Errors ;',
                                  'Download put queue_pre: ', e)
                        i += 1
                        del result_dict, series, windC, ser, s_path
                    else:
                        assert False, 101
                except Exception as e:
                    print(" download data error {}".format(e))
                    self.queueGetErrCnt += 1
                    print(time.ctime(), ' ', i, ' Get', self.queueGetErrCnt,
                          ' Errors ;', 'Download get queue_get: ', e)
                    error_info(101, result_dict)
            else:
                time.sleep(1)


def downloading(file, path):
    save_path = os.path.join(path, file['imageUid'])
    try:
        r = requests.get(file['url'], timeout=3)
    except requests.exceptions.RequestException as e:
        print(time.ctime(), 'Download operation;', e)
    assert r.status_code == 200, 111
    if not r.content:
        print(time.ctime(), 'Download operation result is None')
    with open(save_path + '.dcm', "wb") as code:
        code.write(r.content)


class PullThread(threading.Thread):
    def __init__(self, que_get, que_pre, que_det, que_ret):
        threading.Thread.__init__(self)
        self.que_get = que_get
        self.que_pre = que_pre
        self.que_det = que_det
        self.que_ret = que_ret
        self.pull_data_url = 'http://39.96.243.14:9191/api/gpu/next?modality=CT&st=5'
        # these variable create for log
        self.queuePutErrCnt = 0

    def run(self):
        i = 0
        while True:
            que_gsize = self.que_get.qsize()
            que_psize = self.que_pre.qsize()
            que_dsize = self.que_det.qsize()
            que_rsize = self.que_ret.qsize()
            try:
                print('{} : {} {}- {} - {} - {} '.format(
                    time.ctime(), i, que_gsize, que_psize, que_dsize,
                    que_rsize))
                # if que_dsize < 2 and que_gsize < 2 and que_rsize < 2 and que_psize < 2:
                if que_gsize < 4:

                    # if que_gsize < 100:
                    status, req_result = pull_task_http(self.pull_data_url, i)
                    if not status:
                        print(time.ctime(), ' ', i, 'Pull operation status: ',
                              status)
                    if req_result == {}:
                        print(time.ctime(), ' ', i,
                              ' Pull operation result is None')
                    assert status, 110
                    if status:
                        if req_result['errCode'] == 0:
                            val = req_result['val']
                            if val == {}:
                                print(time.ctime(), ' ', i,
                                      'Pull operation result val is None')
                            assert not val == {}
                            result_dict = {}
                            result_dict['studyInstanceUid'] = val[
                                'studyInstanceUid']
                            result_dict['customStudyInstanceUid'] = val[
                                'customStudyInstanceUid']
                            result_dict['series'] = val['series']
                            try:
                                self.que_get.put(result_dict)
                            except Exception as e:
                                self.queuePutErrCnt += 1
                                print(time.ctime(), ' ', i, ' Get',
                                      self.queuePutErrCnt, ' Errors ;',
                                      ' Download put queue_get: ', e)
                            i += 1
                            del result_dict, status, req_result, val
                else:
                    time.sleep(1)
            except AssertionError as e:
                time.sleep(10)
            except Exception as e:
                print(time.ctime(), ' ', i, 'PULL ERROR: {}'.format(e))


def pull_task_http(pull_data_url, i):
    try:
        result = requests.get(pull_data_url, timeout=3)

        if i > 100:
            return False, {}

        if result.status_code == 200:
            result_json = result.json()
            with open(
                    "/code/json3/" + str(i) + ".json", 'w',
                    encoding='utf-8') as json_file:
                json.dump(result_json, json_file)
            return True, result_json
        else:
            return False, {}
    except requests.exceptions.RequestException as e:
        print(time.ctime(), 'Pull operation;', e)


class PushThread(threading.Thread):
    def __init__(self, que_ret):
        threading.Thread.__init__(self)
        self.que_ret = que_ret
        self.push_data_url = 'http://39.96.243.14:9191/api/gpu/submit'
        # these variable create for log
        self.queueGetErrCnt = 0

    def run(self):
        i = 0
        while True:
            try:
                result_dict = self.que_ret.get(block=True)
            except Exception as e:
                self.queueGetErrCnt += 1
                print(time.ctime(), ' ', i, ' Get', self.queueGetErrCnt,
                      ' Errors ;', 'Push get queue_ret: ', e)
            try:
                json_info = success_ret_info(result_dict)
                url = self.push_data_url + '?customStudyUid=' + result_dict[
                    'customStudyInstanceUid']
                result = requests.post(url, json_info, timeout=2)
                result_json = result.json()
                if not result_json:
                    print(time.ctime(), ' ', i,
                          'Push operation result is None.')
                i += 1
                if os.path.exists(result_dict['data_path']):
                    shutil.rmtree(result_dict['data_path'])
                print(time.ctime(), i, ' {} : {} '.format(
                    result_json, json_info))
                del result_dict, result, result_json, url
            except Exception as e:
                print(time.ctime(), 'Push operation; ', e)


def pull_from_json(i):
    try:
        fileList = find_all_json()
        with open(fileList[i], 'r') as load_f:
            load_dict = json.load(load_f)
            print(load_dict)
        if load_dict:
            status = 200
        if status == 200:
            result_json = load_dict
            return True, result_json
        else:
            return False, {}
    except Exception as e:
        print(time.ctime(), 'Pull operation;', e)


def find_all_json():
    jsonPaths = []
    dirList = os.listdir('/code/')
    for jsonPath in dirList:
        if 'json' in jsonPath:
            for jsonFile in os.listdir(os.path.join('/code', jsonPath)):
                jsonPaths.append(os.path.join('/code', jsonPath, jsonFile))
    return jsonPaths
