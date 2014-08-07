import sys
from PyQt4 import QtCore,QtGui

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
	QtGui.QMainWindow.__init__(self, parent)
	self.configureMainWindow()
        self.initLayout()
        self.initInputs()

    def configureMainWindow(self):
        self.title="Clustering"
        self.x=600.0
        self.y=600.0
	self.setWindowTitle(self.title)
        self.resize(self.x, self.y)

    def initLayout(self):
        direction=QtGui.QBoxLayout.TopToBottom
        self.layout=QtGui.QBoxLayout(direction)

    def initInputs(self):
        inputs = QtGui.QWidget(self)
        inputs.resize(500,400)
        self.textFields={}
        formLayout=QtGui.QFormLayout()
        self.addField("Mean X","0.0",formLayout)
        self.addField("Mean Y","0.0",formLayout)
        self.addField("Var X", "1.0",formLayout)
        self.addField("Var Y", "1.0",formLayout)
        inputs.setLayout(formLayout)
        self.layout.addWidget(inputs)

    def addField(self,name,default,layout):
        text = QtGui.QTextEdit()
        text.append(default)
        text.setMaximumSize(400,40);
        self.textFields[name]=text
        layout.addRow(name,text)

def main():
    app = QtGui.QApplication(sys.argv)
    w = MainWindow()
    w.move(300, 300)
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
