from Tkinter import *
from DrawingBoard import *
from MNISTDetector import *
from PIL import Image
import time

root = Tk()
root.wm_title("MNIST Demo")

detector = MNISTDetector()
drawingBoard = DrawingBoard(root)
v = StringVar()
v.set("Detect Result: N/A")
resultLabel = Label(root, textvariable=v)

def detect():
    global resultLabel
    print "Detecting"
    image = drawingBoard.getImage()
    size = 28, 28
    image.thumbnail(size, Image.ANTIALIAS)
    tv = list(image.getdata())
    tva = [(255-x)*1.0/255.0 for x in tv]
    start = time.time()
    result = list(detector.detect(image=tva)[0])
    elapsed = time.time() - start
    v.set("Detect Result: "+str(result.index(max(result)))+", spend time :"+str(elapsed)+" s")

def clear():
    print "Clearing"
    drawingBoard.clear()

def save():
    print "Save"
    image = drawingBoard.getImage()
    size = 28, 28
    image.thumbnail(size, Image.ANTIALIAS)
    image.save('image.jpg')

drawingBoard.getDrawingArea().grid(row=0, columnspan=3)

resultLabel.grid(row=1, columnspan=3)

detectBtn = Button(root, text="Detect", command=detect)
detectBtn.grid(column=0, row=2)

clearBtn = Button(root, text="Clear", command=clear)
clearBtn.grid(column=1, row=2)

saveBtn = Button(root, text="Save Image", command=save)
saveBtn.grid(column=2, row=2)

root.mainloop()
