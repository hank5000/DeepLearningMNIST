import Tkinter as tk
from PIL import Image, ImageDraw

class DrawingBoard:
    def getDrawingArea(self):
        return self.drawing_area

    def __init__(self,parent):
        self.parent = parent
        self.sizex = 280
        self.sizey = 280
        self.b1 = "up"
        self.xold = None
        self.yold = None
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.image=Image.new("L",(280,280),"White")
        self.draw=ImageDraw.Draw(self.image)

    def save(self):
        filename = "temp.jpg"
        self.image.save(filename)

    def getImage(self):
        return self.image

    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("L",(280,280),"White")
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self, event):
        self.b1 = "down"

    def b1up(self, event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self, event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,width=40)
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),width=40)

        self.xold = event.x
        self.yold = event.y
