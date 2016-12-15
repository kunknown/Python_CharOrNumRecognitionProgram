#Date: 12/05/17
#Team Members: Kaushal Patel (section 1), Zhe Zhou (section 2), Zhaoyang Xie (section 2)

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import PhotoImage
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

class Uploader(Tk):
    def __init__(self,*args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.root = Canvas()
        self.root.pack()

        #creating buttons & labels
        uploadButton = Button(self.root, text='UPLOAD IMAGE', command=self.upload_image).pack(fill=X)
        convertButton = Button(self.root, text='CHECK', command=self.check_for_match).pack(fill=X)
        imageLabel = Label(self.root, text='Selected Image (shown below):').pack()

    #uploading an image
    def upload_image(self):
        file_name = askopenfilename(filetypes=[('PNG files', '*.png'), ('JPEG FILES', '*.jpg')])
        image = Image.open(file_name)
        image = image.resize((50, 50), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        displayImage = Label(image=photo)
        displayImage.image = photo
        displayImage.pack()

        self.file_name = file_name

    #read & convert the uploaded image
    def check_for_match(self):
        textLabel = Label(text='Number or Character (shown below):').pack()

        #reading the training image
        img = cv2.imread('digits.png')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        plt.imshow(gray, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

        #split the gray image to 5000 cells into 50 rows and, each 20*20 size
        cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

        #Make it into a Numpy array. It size will be (50,100,20,20)
        x = np.array (cells)

        #training AI
        train = x[:,:100].reshape(-1,400).astype(np.float32)
        k = np.arange(10)
        train_labels = np.repeat(k,500)[:,np.newaxis]

        #input sample
        sample = cv2.imread(self.file_name)
        gray2 = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        #resize input image to 20x20
        resizedSample = cv2.resize(gray2, (20, 20), interpolation = cv2.INTER_CUBIC)
        plt.imshow(resizedSample, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])

        #put image in 1 cell
        sampleCells = [np.hsplit(row, 1) for row in np.vsplit(resizedSample, 1)]
        y = np.array(sampleCells)
        test = y[0][0].reshape(-1, 400).astype(np.float32)

        #knn algorithm
        knn = cv2.ml.KNearest_create()
        knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
        #test it with test data for k = 5
        ret,result,neigobours,dist = knn.findNearest(test,k = 1)

        #check whether there's a match
        matches = result == test
        correct = np.count_nonzero(matches)
        if correct == 1:
            outputText = Label(text='It\'s a number').pack()
        else:
            outputText = Label(text='It\'s a character').pack()

root = Uploader()
root.geometry("500x500+700+100")
root.title('"Number or Character" - Recognition Program')
root.mainloop()
