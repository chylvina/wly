import json
import requests
import time
import os
import shutil

def error_info(error_code, result_dict):
    error = {
        '100': 'The cpu process is abnormal',
        '101': 'Read DICOM file exception',
        '102': 'Preprocessed data is abnormal',
        '111': 'File download exception',
        '110': 'Get task exception',
        '112': 'Information processing exception',
        '200': 'The GPU process is abnormal',
        '201': 'Target detection is abnormal',
        '202': 'Abnormal data classification'
    }
    msg = {'errorCode': error_code, 'errorMsg': error[str(error_code)], 'creator':'xuelang'}
    error_ret_info(json.dumps(msg), result_dict)
    # if os.path.exists(result_dict['data_path']):
    #     shutil.rmtree(result_dict['data_path'])


def success_ret_info(result_dict):
    preb = result_dict['nodule_preb']
    series = []
    # long_axis = result_dict['long_axis']
    # lobe_info = result_dict['lung_lobe']
    dicomName = result_dict['prep_inst']
    for index, row in preb.iterrows():
        ser = {}
        ser['srsUid'] = result_dict['seriesUid']
        ser['coordX'] = row.coordX
        ser['coordY'] = row.coordY
        ser['coordZ'] = dicomName[int(row.coordZ)]
        # print('Z:', row.coordZ, '-->', ser['coordZ'])
        ser['diameter_mm'] = row.diameter_mm
        ser['lungLob'] = row.lobel_info
        # ser['lungSeg'] = None
        # ser['contour'] = None  # lun kuo
        # ser['bm'] = None  # liang e xing
        # if row.long == 0:
        #     ser['lAxis'] = row.diameter_mm
        # else:
        #     ser['lAxis'] = row.long
        # if row.long == 0:
        #     ser['sAxis'] = row.diameter_mm * 0.8
        # else:
        #     ser['sAxis'] = row.short
        # ser['hu'] = None
        # ser['kpi'] = None
        series.append(ser)
    info = {"type": "AI", "format": "DCM",
            "sdyUid": result_dict['studyInstanceUid'],
            'series': series,
            'creator':'xuelang'}

    return json.dumps(info)


def error_ret_info(msg, result_dict):
    print('error', msg, result_dict)
    # if result_dict:
    #     try:
    #         push_data_url = 'http://39.96.243.14:9191/api/gpu/submit'
    #         url = push_data_url + '?customStudyUid=' + result_dict['customStudyInstanceUid']
    #         result = requests.post(url, msg)
    #         print(time.ctime(),msg, result.json())
    #         del result_dict,result
    #         s = requests.session()
    #         s.keep_alive = False
    #         s.close()
    #     except Exception as e:
    #         print("error_ret_info error {}".format(e))
    #         pass

if __name__ == '__main__':
    result_dict = {'errorCode': 100, 'errorMsg': 'The cpu process is abnormal.',
                   'customStudyInstanceUid': '1.3.6.1.4.1.46677.0.600268.47419611.1904020116.2336'}
    json = error_ret_info(result_dict, result_dict)
    print(json)
