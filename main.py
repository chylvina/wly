from flask import Flask
import queue
from work_cpu import CpuThread
from work_gpu import GpuThread
from work_oth import PullThread, PushThread
import time
import warnings
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

warnings.filterwarnings("ignore")

app = Flask(__name__)


# http://192.168.1.222:9595/api/gpu/next?modality=CT&st=5


def hello():
    print("Wan Li Yun Provide Services \n Dete Servcie ")


if __name__ == "__main__":
    hello()
    que_pre = queue.Queue()
    que_det = queue.Queue()
    que_ret = queue.Queue()

    pull_thread = PullThread(que_pre, que_det, que_ret)
    pull_thread.setDaemon(True)
    pull_thread.start()

    # down_thread = DownThread(que_get, que_pre)
    # down_thread.setDaemon(True)
    # down_thread.start()

    # down_thread_nums = 2
    # for i in range(down_thread_nums):
    #     down_thread = DownThread(que_get, que_pre)
    #     down_thread.setDaemon(True)
    #     down_thread.start()
    # use gpu or cpu

    cpu_thread_nums = 4
    for i in range(cpu_thread_nums):
        cpu_thread = CpuThread(que_pre, que_det, "gpu", i % 2)
        cpu_thread.setDaemon(True)
        cpu_thread.start()

    # for i in range(1):
    #     cpu_thread = CpuThread(que_pre, que_det, "cpu")
    #     cpu_thread.setDaemon(True)
    #     cpu_thread.start()

    gpu_thread_nums = 2
    for i in range(gpu_thread_nums):
        gpu_thread = GpuThread(que_det, que_ret, i % 2)
        gpu_thread.setDaemon(True)
        gpu_thread.start()

    # gpu_thread = GpuThread(que_det, que_ret)
    # gpu_thread.setDaemon(True)
    # gpu_thread.start()
    #
    push_thread = PushThread(que_ret)
    push_thread.setDaemon(True)
    push_thread.start()

    try:
        while push_thread.isAlive():
            time.sleep(2)
    except KeyboardInterrupt:
        print("stopped by keyboard")
    # add MAX HU
    # add AUG HU

