from Tkinter import *
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageEnhance

class ImageButcher(Tk):
    def __init__(self):
        Tk.__init__(self)

        #create ui
        f = Frame(self, bd=2)

        self.colour = StringVar(self)
        self.colourMenu = OptionMenu(f, self.colour,
                                        *('red', 'green', 'blue', 'white'))
        self.colourMenu.config(width=5)
        self.colour.set('red')
        self.colourMenu.pack(side='left')

        self.rectangleButton = Button(f, text='Rectangle',
                                    command=self.draw_rectangle)
        self.rectangleButton.pack(side='left')

        self.brightenButton = Button(f, text='Brighten',
                                    command=self.on_brighten)
        self.brightenButton.pack(side='left')

        self.mirrorButton = Button(f, text='Mirror',
                                    command=self.on_mirror)
        self.mirrorButton.pack(side='left')
        f.pack(fill='x')

        self.c = Canvas(self, bd=0, highlightthickness=0,
                        width=100, height=100)
        self.c.pack(fill='both', expand=1)

        #load image
        im = Image.open('/home/sjb/Desktop/python/python-study/img/brain.jpg')
        im.thumbnail((512,512))

        self.tkphoto = ImageTk.PhotoImage(im)
        self.canvasItem = self.c.create_image(0,0,anchor='nw',image=self.tkphoto)
        self.c.config(width=im.size[0], height=im.size[1])

        self.img = im
        self.temp = im.copy() # 'working' image

    def display_image(self, aImage):
        self.tkphoto = pic = ImageTk.PhotoImage(aImage)
        self.c.itemconfigure(self.canvasItem, image=pic)

    def on_mirror(self):
        im = ImageOps.mirror(self.temp)
        self.display_image(im)
        self.temp = im

    def on_brighten(self):
        brightener = ImageEnhance.Brightness(self.temp)
        self.temp = brightener.enhance(1.1) # +10%
        self.display_image(self.temp)

    def draw_rectangle(self):
        bbox = 9, 9, self.temp.size[0] - 11, self.temp.size[1] - 11        
        draw = ImageDraw.Draw(self.temp)
        draw.rectangle(bbox, outline=self.colour.get())
        self.display_image(self.temp)


app = ImageButcher()
app.mainloop()