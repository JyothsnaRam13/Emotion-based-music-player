from tkinter import *
import numpy as np
import pandas as pd
import os
from tkinter import ttk

from PIL import ImageTk, Image  

root = Tk()
root.title("Welcome")
img =Image.open('doc.jpg')
bg = ImageTk.PhotoImage(img)
##root.geometry("650x450")

# Add image
label = Label(root, image=bg)
label.place(x = 0,y = 0)

root.geometry("960x603")
w2 = Label(root, justify=LEFT, text="Emotion based music player", fg="black", bg="#e6921e")
w2.config(font=("times", 30,"bold"))
w2.place(x=250 , y=75)




def rms():
    os.system('TestEmotionDetector.py')


def sym():
    os.system('python Emoplayerfinal.py')
    




pms = Button(root, text="       Emotion Detection      ",font=("times", 25,"bold"),activebackground="#0de6ff", command=rms,bg="#89c5cc",fg="black",width=20)
pms.place(x=300 , y=300)
sb = Button(root,  text="          Emotion Based Music          ",font=("times", 25,"bold"),activebackground="#0de6ff", command=sym,bg="#89c5cc",fg="black",width=20)
sb.place(x=300 , y=400)
