from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

filename = "/Users/markporter/Documents/PHOTO-2019-01-28-11-32-33.jpg"
def UploadAction():
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    path = filename

    # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(path))
    image.configure(image=img)
    image.img = img
    return


main_window = Tk()
main_window.geometry("800x800")

label_1 = Label(main_window, text="Choose a file:")
label_1.pack()

button_1 = Button(main_window, text="Choose a File", command=UploadAction).pack()

# Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img = ImageTk.PhotoImage(Image.open(filename))

# The Label widget is a standard Tkinter widget used to display a text or image on the screen.
image = Label(main_window, image=img)
image.pack()
main_window.mainloop()

