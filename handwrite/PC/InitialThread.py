import threading
import time


class InitialThread(threading.Thread):
    def __init__(self, conn, data_controller):
        threading.Thread.__init__(self)
        self.conn = conn
        self.dataController = data_controller
        self.exitFlag = False
        t = time.time()
        print("Initial start timestamp:%d" % int(round(t*1000)))

    def run(self):
        while not self.exitFlag:
            data_str = self.conn.recv(1024).decode('utf-8')
            print("data:%s" % data_str)
            if "" == data_str:
                self.exitFlag = True
                break
            self.exitFlag = self.dataController.decode_str(data_str)
        t = time.time()
        print("Initial end timestamp:%d" % int(round(t * 1000)))
        self.dataController.initial()
