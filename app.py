import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import word_recognition as wr

IMG_WIDTH = 480
IMG_HEIGHT = 320
img_path = ""


def choose_photo(event):
    global img_path
    img_path = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(),"input_data"),
                                          title="Select file",
                                          filetypes=(('image files', '.png'), ('image files', '.jpg')))
    load = Image.open(img_path)
    resized = load.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(resized)
    photoContainer.config(image=render)
    photoContainer.image = render


def recognize_word():
    global img_path, model
    recognized_word = wr.recognize(img_path, model)
    lblRecognitionResult.config(text=recognized_word)


# MAIN
root = Tk()
root.title = "Word recognition"
root.geometry("500x500")

model = wr.load_model_data()

photoContainer = Label(root, image=PhotoImage(width=IMG_WIDTH, height=IMG_HEIGHT), text="Click bellow to select image",
                       compound=BOTTOM)
photoContainer.pack(fill=X, side=TOP, pady=10)
photoContainer.bind('<Button-1>', choose_photo)

btnRecognize = Button(root, text="Recognize", fg="green", command=recognize_word)
btnRecognize.pack(pady=10)

lblRecognitionResult = Label(root, text="Test", font=("Times", 36, "bold italic"))
lblRecognitionResult.pack(fill=X, pady=10)

root.mainloop()