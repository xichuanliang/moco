import matplotlib.pyplot as plt


class util():
    def __init__(self, loss_list, accuracy_list1, accuracy_list2, epoch, jpgName):
        self.loss_list = loss_list
        self.epoch = epoch
        self.jpgName = jpgName
        self.accuracy_list1 = accuracy_list1
        self.accuracy_list2 = accuracy_list2

    def drawSingle(self, loss_list, accuracy_list1, epoch, jpgName):
        x1 = range(0, epoch)
        x2 = range(0, epoch)
        y1 = accuracy_list1
        y2 = loss_list
        plt.subplot(2, 1, 1)
        plt.plot(x1, y1, 'o-')
        plt.title('Train accuracy vs. epoches')
        plt.ylabel('Train accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, '.-')
        plt.xlabel('Train loss vs. epoches')
        plt.ylabel('Train loss')
        plt.savefig(jpgName)
        plt.show()

    def drawDouble(self, accuracy_list1, accuracy_list2, epoch, jpgName):
        x = range(0, epoch)
        y1 = accuracy_list1
        y2 = accuracy_list2
        plt.title('different accuracy')
        plt.plot(x, y1, 'b', label='Training accuracy')
        plt.plot(x, y2, 'r', label='validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(jpgName)
        plt.figure()
