import tkinter
import cv2
from matplotlib import pyplot as plt
from PIL import ImageTk, Image
from math import floor
import numpy as np
from scipy.interpolate import griddata


class ImageEditor:
    def __init__(self, obj):
        self.obj = obj
        self.image = obj.image
        self.grey_image = obj.grey_image
        self.height, self.width = self.grey_image.shape[:2]
    
    # Функція для нормалізації гістограми 8-бітового зображення
    def normalization(self):
        copy_image = self.grey_image.copy()
        intensity = self.obj.intensity_distribution()
        first = Histogram.first_above_zero(intensity)
        last = Histogram.last_above_zero(intensity) 
        for i in range(self.height):
            for j in range(self.width):
                if self.grey_image[i][j] == 0:
                    copy_image[i][j] = 0
                else:
                    copy_image[i][j] = floor(255 * ((self.grey_image[i][j] - first) / (last-first)))
        return copy_image
    
    def normalization_rgb(self):
        copy_image = self.image.copy()
        red, green, blue = self.obj.intensity_distribution_rgb()
        first, last = [], []
        first.extend([Histogram.first_above_zero(blue),
                      Histogram.first_above_zero(green),
                      Histogram.first_above_zero(red)])
        
        last.extend([Histogram.last_above_zero(blue),
                      Histogram.last_above_zero(green),
                      Histogram.last_above_zero(red)])
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3):
                    if self.image[i][j][k] == 0:
                        copy_image[i][j][k] = 0
                    else:
                        copy_image[i][j][k] = floor(255 * ((self.image[i][j][k] - first[k])
                                                        / (last[k]-first[k])))
        return copy_image
        
                    
    
    def equalization(self):
        copy_image = self.grey_image.copy()
        intensity = self.obj.intensity_distribution()
        cdf = self.obj.cumulate_sum(intensity)
        for i in range(self.height):
            for j in range(self.width):
                copy_image[i][j] = floor(cdf[self.grey_image[i][j]] * 255)
        return copy_image

    def equalization_rgb(self):
        copy_image = self.image.copy()
        red, green, blue = self.obj.intensity_distribution_rgb()
        cdf_red = self.obj.cumulate_sum(red)
        cdf_green = self.obj.cumulate_sum(green)
        cdf_blue = self.obj.cumulate_sum(blue)
        for i in range(self.height):
            for j in range(self.width):
                copy_image[i][j][0] = floor(cdf_blue[self.image[i][j][0]] * 255)
                copy_image[i][j][1] = floor(cdf_green[self.image[i][j][1]] * 255)
                copy_image[i][j][2] = floor(cdf_red[self.image[i][j][2]] * 255)
        return copy_image
    
    def perspective_transformation(self):
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x,",",y)
                inmap.append([x,y])
                X.append(y); Y.append(x)
                strXY = str(x)+", "+str(y)
                cv2.circle(self.image, (x, y), 3, (0,0,255))
                cv2.imshow("image", self.image)
                if len(inmap) == 4:
                    source = np.float32(inmap)
                   
                    destination = np.float32([[0, 0],
                                        [0, max(X)-1],
                                        [max(Y) - 1, max(X) - 1],
                                        [max(Y) - 1, 0]])
                    in_ = cv2.getPerspectiveTransform(src=source, dst=destination)
                    out = cv2.warpPerspective(self.image, in_, (max(Y), max(X)),
                                              flags=cv2.INTER_LINEAR)
                    cv2.imshow("image", out)
        
        inmap = []
        X, Y = [], []
        cv2.imshow("image", self.image)
        cv2.setMouseCallback("image", click_event)
        
    def bilinear_transformation(self):
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x,",",y)
                inmap.append([x,y])
                X.append(y); Y.append(x)
                strXY = str(x)+", "+str(y)
                cv2.circle(self.image, (x, y), 3, (0,0,255))
                cv2.imshow("image", self.image)  
            if len(inmap) == 4:
                destination = np.float32(inmap)
                wh_complex = complex(self.image.shape[1]-1, self.image.shape[1])
                source = np.float32([[0, 0],
                                     [0, self.image.shape[1]],
                                     [self.image.shape[0], self.image.shape[1]],
                                     [self.image.shape[0] - 1, 0]]) 
                grid_x, grid_y = np.mgrid[0:wh_complex.real:wh_complex-wh_complex.real,
                                          0:wh_complex.real:wh_complex-wh_complex.real] 
                grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
                map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(self.image.shape[1], 
                                                                          self.image.shape[0])
                map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(self.image.shape[1], 
                                                                          self.image.shape[0])
                map_x_32 = map_x.astype('float32')
                map_y_32 = map_y.astype('float32') 
                warped = cv2.remap(self.image, map_x_32, map_y_32, cv2.INTER_CUBIC)   
                cv2.imshow("image", warped) 
        inmap = []
        X, Y = [], []
        cv2.imshow("image", self.image)
        cv2.setMouseCallback("image", click_event)  

    def erosion(self):
        kernel = np.ones((5,5), np.uint8)
        result = cv2.erode(self.image,kernel,iterations = 1)
        return result
    
    def dilation(self):
        kernel = np.ones((5,5), np.uint8)
        result = cv2.dilate(self.image, kernel, iterations = 1)
        return result
    
    def opening(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        return result

    def closing(self):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        return result


class Histogram:
    def __init__(self, path=None, image=None):
        self.path = path or None
        if self.path:
            self.image = cv2.imread(self.path, cv2.COLOR_BGR2GRAY)
            self.grey_image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        elif image is not None:
            self.image = image
            if self.image.dtype == 'uint8':
                self.grey_image = self.image
            else:
                self.grey_image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    
    def cumulate_sum(self, hist):
        return [sum(hist[:i+1]) for i in range(len(hist))]
    
    # Фукнція для обчислення гістрограми для 8-бітового зображення
    def intensity_distribution(self):
        height, width = self.grey_image.shape[:2]
        intensity = [0] * 256
        for i in range(height):
            for j in range(width):
                intensity[self.grey_image[i][j]] += 1
        return [i / (width * height) for i in intensity]
    

    # Функція для виведення гістограми для RGB-зображення
    def intensity_distribution_rgb(self):
        height, width = self.image.shape[:2]
        red, green, blue = [0] * 256, [0] * 256, [0] * 256
        for i in range(height):
            for j in range(width):
                red[self.image[i][j][2]] += 1
                green[self.image[i][j][1]] += 1
                blue[self.image[i][j][0]] += 1
        red = [i / (width * height) for i in red]
        green = [i / (width * height) for i in green]
        blue = [i / (width * height) for i in blue]
        return [red, green, blue]
    
    # Функція для виведення гістограми
    def build_histogram_rgb(self):
        red, green, blue = self.intensity_distribution_rgb()
        plt.plot(red, 'r')
        plt.plot(green, 'g')
        plt.plot(blue, 'b')
        plt.show()
    
    def build_histogram(self):
        intensity = self.intensity_distribution()
        plt.plot(intensity)
        plt.show()
    
    # Ця фукнція повертає індекс першого невід'много елемента гістограми
    @staticmethod
    def first_above_zero(array):
        for i in range(len(array)):
            if array[i] > 0:
                return i

    # Ця фукнція повертає індекс останнього невід'много елемента гістограми
    @staticmethod
    def last_above_zero(array):
        for i in range(len(array)-1, -1, -1):
            if array[i] > 0:
                return i


class Gui(tkinter.Tk):
    def __init__(self, title, screen_size):
        tkinter.Tk.__init__(self)
        self.title(title)
        self.geometry(screen_size)
        self.button_open_image = tkinter.Button(self, text="Choose a picture",
                                                command=self.open_image)
        self.button_histogram = tkinter.Button(self, text='Histogram',
                                               command=self.open_histogram)
        self.button_rgb_histogram = tkinter.Button(self, text='RGB Histogram',
                                                    command=self.open_rgb_histogram)
        self.button_normalization = tkinter.Button(self, text='Normalization',
                                                    command=self.open_normalized_img)
        self.button_normalization_rgb = tkinter.Button(self, text='Normalization RGB',
                                                    command=self.open_normalized_img_rgb)
        self.button_equalization = tkinter.Button(self, text='Equalization',
                                                  command=self.open_equalized_img)
        self.button_equalization_rgb = tkinter.Button(self, text='Equalization RGB',
                                                  command=self.open_equalized_img_rgb)
        self.button_perspective_transformation = tkinter.Button(self, text='Perspective transformatin',
                                                  command=self.open_perpective_transformation)
        self.button_bilinear_transformation = tkinter.Button(self, text='Bilinear transformation',
                                                   command=self.open_bilinear_transformation)
        self.button_erosion = tkinter.Button(self, text='Erosion',
                                             command=self.open_erosion_transformation)
        self.button_dilation = tkinter.Button(self, text='Dilation',
                                              command=self.open_dilation_transformation)
        self.button_opening = tkinter.Button(self, text='Opening',
                                             command=self.open_opening_transformation)
        self.button_closing = tkinter.Button(self, text='Closing',
                                             command=self.open_closing_transformation)
        self.button_open_image.place(x=1000, y=0, width=150)
        self.button_histogram.place(x=1000, y=25, width=150)
        self.button_rgb_histogram.place(x=1000, y=50, width=150)
        self.button_normalization.place(x=1000, y=75, width=150)
        self.button_normalization_rgb.place(x=1000, y=100, width=150)
        self.button_equalization.place(x=1000, y=125, width=150)
        self.button_equalization_rgb.place(x=1000, y=150, width=150)
        self.button_perspective_transformation.place(x=1000, y=175, width=150)
        self.button_bilinear_transformation.place(x=1000, y=200, width=150)
        self.button_erosion.place(x=1000, y=225, width=150)
        self.button_dilation.place(x=1000, y=250, width=150)
        self.button_opening.place(x=1000, y=275, width=150)
        self.button_closing.place(x=1000, y=300, width=150)
        self.canvas = tkinter.Canvas(self, width=700, height=600)
        self.canvas.place(x=0)

    def open_image(self):
        self.filename = tkinter.filedialog.askopenfilename(initialdir='./images',
                                                          title='Choose a picture',
                                                          filetypes=(('png', '*.png'),
                                                                     ('jpg', '*.jpg')))
        self.image = ImageTk.PhotoImage(Image.open(self.filename))
        self.canvas.delete("all")
        self.canvas.background = self.image
        self.canvas.create_image((5, 5), anchor='nw', image=self.canvas.background)
    
    def open_histogram(self):
        histogram = Histogram(path=self.filename)
        histogram.build_histogram()
    
    def open_rgb_histogram(self):
        histogram = Histogram(path=self.filename)
        histogram.build_histogram_rgb()
    
    def open_normalized_img(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram)
        edited_image = image.normalization()
        cv2.imshow('Normalization', edited_image)
        Histogram(image=edited_image).build_histogram()
        cv2.waitKey(0)
    
    def open_normalized_img_rgb(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram)
        edited_image = image.normalization_rgb()
        cv2.imshow('Normalization RGB', edited_image)
        Histogram(image=edited_image).build_histogram_rgb()
        cv2.waitKey(0)
    
    def open_equalized_img(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram)
        edited_image = image.equalization()
        cv2.imshow('Equalization', edited_image)
        Histogram(image=edited_image).build_histogram()
        cv2.waitKey()
    
    def open_equalized_img_rgb(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram)
        edited_image = image.equalization_rgb()
        cv2.imshow('Equalization RGB', edited_image)
        Histogram(image=edited_image).build_histogram_rgb()
        cv2.waitKey(0)
    
    def open_perpective_transformation(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram)
        image.perspective_transformation()
    
    def open_bilinear_transformation(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram)
        image.bilinear_transformation()
    
    def open_erosion_transformation(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram) 
        result = image.erosion()
        cv2.imshow('Erosion', result) 
        cv2.waitKey(0)  

    def open_dilation_transformation(self):
        histogram = Histogram(path=self.filename)
        image = ImageEditor(histogram) 
        result = image.dilation()
        cv2.imshow('Dilation', result) 
        cv2.waitKey(0)  

    def open_opening_transformation(self):
        histogram = Histogram(path=self.filename)         
        image = ImageEditor(histogram)
        result = image.opening()
        cv2.imshow('Opening', result)
        cv2.waitKey(0)
        
    def open_closing_transformation(self):
        histogram = Histogram(path=self.filename)         
        image = ImageEditor(histogram)
        result = image.closing()
        cv2.imshow('Closing', result)
        cv2.waitKey(0)
    


gui = Gui('opencv', '1150x700')
gui.mainloop()

