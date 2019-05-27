from MyFrame import *
import socket
from InitialThread import *
from HWController import *
from SensorDetectThread import *
import queue
from HWDetectorThread import *
from RecognizeHelperUI import *
import _thread

class HandwriteFrame(MyFrame):
    def __init__(self, parent):
        MyFrame.__init__(self, parent)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()
        port = 8124
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.ip, port))
        self.s.listen(5)
        self.m_staticText_ip.SetLabel(self.ip)

    def start(self, event):
        self.m_staticText_model.SetLabel('Load model...')
        _thread.start_new_thread(start_game_thread, (self, self.s))
        self.m_button_start.Disable()
        self.m_button_stop.Enable()

    def stop(self, event):
        self.m_textCtrl.SetValue('')


def start_game_thread(frame, s):
    hwController = HWController()
    recognizeHelper = RecognizeHelper(frame)
    try:
        while True:
            print("Listenning...")
            conn, add = s.accept()
            print("A client connecting...")
            pr = conn.recv(128).decode('utf-8')
            print("pr:%s" % pr)
            if "ISS" == pr:
                # 初始化阈值
                initialThread = InitialThread(conn, hwController)
                initialThread.start()
            elif "PC" == pr:
                if hwController.h_thre == 0:
                    hwController.load_threholds(hwController.threholdFilePath)
                print("h_thre:%s l_thre:%s thre:%s" % (
                hwController.h_thre, hwController.l_thre, hwController.thre))
                dataQueue = queue.Queue(maxsize=0)
                hwDetectorThread = HWDetectorThread(dataQueue, hwController, recognizeHelper)
                hwDetectorThread.start()
                sensorDetectThread = SensorDetectThread(conn, dataQueue, hwDetectorThread)
                sensorDetectThread.start()
            else:
                print("Error program flag!")
        print("MInput Server end!")

    except KeyboardInterrupt:
        print("you have CTRL+C, Now quit")
        s.close()


if __name__ == '__main__':
    app = wx.App(False)
    frame = HandwriteFrame(None)
    frame.Center()
    frame.Show(True)
    app.MainLoop()
