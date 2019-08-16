from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import numpy  as np
import time
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication,QWidget
from PyQt5.QtCore import (QEvent,QTimer,Qt)
from PyQt5.QtGui import QPainter
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model
import mltools
import pyqtgraph as pg

class Myfigure(FigureCanvas):
    def __init__(self):
        self.fig = plt.figure()
        FigureCanvas.__init__(self,self.fig)
    def Draw(self):
        # 画频谱图
        weight_file = 'weights-ResNet-norm.h5'

        test_filename="F:/PyQT5学习/01.hdf5"
        test_file = h5py.File(test_filename, 'r')
        X = test_file['X'][:, :, :]
        Y = test_file['Y'][:]
        Z = test_file['Z'][:]


        point_num = 1024
        x = np.linspace(0, 5, point_num)
        y1 = X[0][:, 0][:point_num]
        y2 = X[0][:, 1][:point_num]
        fs = 2000
        y = y1 + y2 * 1j
        NFFT = 256  # the length of the windowing segments
        dt = 0.0005
        Fs = int(1.0 / dt)  # the sampling frequency
        Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=Fs, mode='psd')
        position = self.fig .add_axes()  # 位置[左,下,右,上]
        plt.colorbar(im, cax=position, orientation='horizontal')  # 方向horizontal or vertical
        self.draw()


    def Draw1(self):
        weight_file = 'weights-ResNet-norm.h5'

        test_filename = "F:/PyQT5学习/01.hdf5"
        test_file = h5py.File(test_filename, 'r')
        X = test_file['X'][:, :, :]
        Y = test_file['Y'][:]
        Z = test_file['Z'][:]

        classes = ['2FSK',
                   '4FSK',
                   '4ASK',
                   '16QAM',
                   'MSK',
                   'OQPSK',
                   'BPSK',
                   'QPSK',
                   '8PSK',
                   'LFM'
                   ]

        if weight_file is not None:
            model = load_model(weight_file)
        Y_hat = model.predict(X, batch_size=1024, verbose=1)
        test_file.close()
        # plot confusion matrix
        cm, right, wrong = mltools.calculate_confusion_matrix(Y, Y_hat, classes)
        acc = round(1.0 * right / (right + wrong), 4)
        print('CNN网络模型在测试集上的总准确率:%.2f%s / (%d + %d)' % (100 * acc, '%', right, wrong))

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        self.tick_marks = np.arange(len(classes))
        plt.xticks(self.tick_marks, classes, rotation=45)
        plt.yticks(self.tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('正确的标签')
        plt.xlabel('预测的标签')
        self.draw()

class Main_window(QWidget):
    def __init__(self):
        super(Main_window, self).__init__()

        hbox = QHBoxLayout()


        splitter1=QSplitter(Qt.Horizontal)
        self.figure1 = Myfigure()
        splitter1.addWidget(self.figure1)
        self.figure1.Draw1()


        self.figure=Myfigure()
        splitter1.addWidget(self.figure)
        self.figure.Draw()



        f = h5py.File("F:/PyQT5学习/01.hdf5", 'r')
        self.a = f['X'][:]
        self.b = f['Y'][:]
        self.c = f['Z'][:]
        point_num = 1024
        x = np.linspace(0, 5, point_num)
        y1 = self.a[0][:, 0][:point_num]
        y2 = self.a[0][:, 1][:point_num]

        self.plt1 = pg.PlotWidget(title="时域图")
        self.plt1.addLegend()
        self.curve1 = self.plt1.plot(pen=pg.mkPen("r", width=2), name="I")  # 设置pen 格式
        self.curve2 = self.plt1.plot(pen=pg.mkPen("g", width=2), name="Q")  # 设置pen 格式
        self.curve1.setData(x, y1)
        self.curve2.setData(x, y2)


        splitter2 = QSplitter(Qt.Vertical)
        splitter2.addWidget(self.plt1)
        splitter2.addWidget(splitter1)

        splitter2.setSizes([200, 400])

        #设置窗体全局布局以及子布局的添加
        hbox.addWidget(splitter2)
        self.setLayout(hbox)



if __name__ =='__main__':
    app=QApplication(sys.argv)
    ui=Main_window()
    ui.show()
    sys.exit(app.exec_())
