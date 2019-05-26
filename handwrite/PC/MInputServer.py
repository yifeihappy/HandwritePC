import socket
from InitialThread import *
from HWController import *
from SensorDetectThread import *
import queue
from HWDetectorThread import *
from RecognizeHelper import *
# 实时keystroke 识别系统

# hostname = socket.gethostname()
# ip = socket.gethostbyname(hostname)
# ip = "192.168.0.125"
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
finally:
    s.close()
print(ip)
port = 8124
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, port))
s.listen(5)
print("Handwriting Server starts:%s:%s" % (ip, port))
hwController = HWController()
recognizeHelper = RecognizeHelper()
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
            print("h_thre:%s l_thre:%s thre:%s" % (hwController.h_thre, hwController.l_thre, hwController.thre))
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

