import asip_v2 as modl
import os
import tkinter as tk
from PIL import ImageTk,Image
from tkinter import ttk,filedialog
import numpy as np
import cv2
import pandas as pd

class GUI(object):
    def __init__(self):
        self.img_dict = dict()
        # self.restart = False
        self.root = tk.Tk()
        # self.times = 0
        self.start()

    def start(self):
        self.root.title('Contextual classification')
        self.btn_open = tk.Button(self.root, text="OPEN",command=lambda: self.open(),width=14,bg='green',fg='black')
        self.btn_open.grid(column=0,row=0,columnspan=9999,padx=4,pady=4)
        self.image_label = tk.Label(self.root)
        self.image_label.grid(column=0, row=4, columnspan=9999, padx=4, pady=4)
        self.btn_exit = tk.Button(self.root,text="EXIT",width=14,bg='red',fg='black',command=lambda: self.root.destroy())
        self.btn_exit.grid(column=2, row=7, padx=4, pady=4)
        self.root.mainloop()

    def savefile(self, image):
        files = [("JPG", "*.jpg"),("PNG","*.png"),("TIFF","*.tiff"),("all files","*.*")]
        file_name = filedialog.asksaveasfile(
                                                initialdir=os.path.dirname(os.path.abspath(__file__)),
                                                title="Save as",mode='w', 
                                                filetypes=files,
                                                defaultextension=files
                                             )
        if file_name == None:
            return
        cv2.imwrite(file_name.name, np.array(image, dtype=np.uint8))

    def open(self):
        self.root.filename = filedialog.askopenfilename(
                                                            initialdir=os.path.dirname(os.path.abspath(__file__)),
                                                            title="Select file",
                                                            filetypes=[("JPG", "*.jpg"),("PNG","*.png"),("TIFF","*.tiff"),("all files","*.*")]
                                                        )
        try:
            self.color = cv2.imread(self.root.filename)
        except:
            GUI.start(self)  

        size = self.color.shape
        self.img_list = []
        self.display_list = []
        self.origin = []
        self.clas = []
        self.index = 2
        self.mat_ind = 0
        self.p = 3
        self.img_list.append(self.color)
        self.btn_color = 'green'
        GUI.my_win(self)


    def my_win(self):
        self.btn_restart = tk.Button(self.root, text="RESTART",command=lambda: [self.root.destroy(),GUI()],width=14,bg='black',fg='white')
        self.btn_restart.grid(column=0,row=0,columnspan=9999,padx=4,pady=4)
        
        self.image_label.grid_forget()
        self.view(self.img_list[0])
        self.txt_k = tk.ttk.Label(self.root, width=17, text="K value for K-Means :", font=("Times New Roman", 10))
        self.txt_k.grid(column=0, row=1, padx=4, pady=4) 
        self.ks = tk.Text(self.root, height=1, width=8, bg="light yellow")
        self.ks.grid(column=1, row=1, padx=4, pady=4)
        
        self.img_name = tk.ttk.Label(self.root)
        self.img_name.grid(column=5, row=4, padx=8, pady=4)

        self.btn0 = tk.Button(self.root, text="Perform Soft K-means", width=16, bg='blue violet', command=lambda: self.contextual())
        self.btn0.grid(column=2, row=1, padx=4, pady=4)
    
    def contextual(self):
        self.txt_k.grid_forget()
        self.ks.grid_forget()
        self.btn0.grid_forget()

        self.k = int(self.ks.get('1.0','end-1c')) 
        self.hidden_mat, final_belongs_to, final_centroid = modl.k_means(self.img_list[0],self.k)
        # self.img_list.append(self.hidden_mat)
        self.clas.append(final_belongs_to)

        self.btn1 = tk.Button(self.root, text="Show K-means output", width=16, bg='blue violet', command=lambda: self.view_K(final_belongs_to))
        self.btn1.grid(column=0, row=1, padx=4, pady=4)
        self.btn2 = tk.Button(self.root, text="Save probabilities", width=14, bg='blue violet', command=lambda: self.save_mat(self.mat_ind))
        self.btn2.grid(column=0, row=7, padx=4, pady=4)
        self.btn3 = tk.Button(self.root, text="Enter compatibility matrix", width=19, bg='blue violet', command=lambda: self.comp_mat())
        self.btn3.grid(column=1, row=1, padx=4, pady=4)
    


    def context_calc(self,neigh,iterations,comp_tx):
        self.btn1.grid_forget()
        self.btn3.grid_forget()

        com = comp_tx.get('1.0','end-1c')
        com = com.split(',')
        j = 0
        for i in com:
            com[j] = i.split()
            j += 1
        self.comp = np.array(com,dtype=np.float)
        # print(self.comp, self.comp.shape)
        self.Probs = modl.relax(self.hidden_mat, self.comp, int(neigh.get('1.0','end-1c')), self.k, int(iterations.get('1.0','end-1c')))
        # self.img_list.append(self.Probs)
        self.clas.append(np.argmax(self.Probs,axis=2))
        # self.img_list[2] = np.argmin(self.Probs,axis=2)
        self.btn31 = tk.Button(self.root, text="Show Final output", width=14, bg='red', command=lambda: self.view_K(np.argmax(self.Probs,axis=2),True))
        self.btn31.grid(column=0, row=1, padx=4, pady=4)

    def comp_mat(self):
        self.sub = tk.Toplevel()
        comp_txt = tk.ttk.Label(self.sub, wraplength = 500, text="Give space between elements of same \n column and a comma before start of new row", font=("Times New Roman", 10))
        comp_txt.grid(column=0, row=0, padx=4, pady=4) 
        comp_tx = tk.Text(self.sub, height=5, width=25, bg="light yellow")
        comp_tx.grid(column=0, row=1, padx=2, pady=2)
        
    
        neigh_txt = tk.ttk.Label(self.sub, width=10, text="Neighbors:", font=("Times New Roman", 10))
        neigh_txt.grid(column=0, row=2, padx=2, pady=2) 
        neigh = tk.Text(self.sub, height=1, width=5, bg="light yellow")
        neigh.grid(column=1, row=2, padx=4, pady=4)
        
        iterations_txt = tk.ttk.Label(self.sub, width=10, text="Iterations:", font=("Times New Roman", 10))
        iterations_txt.grid(column=0, row=3, padx=4, pady=4) 
        iterations = tk.Text(self.sub, height=1, width=5, bg="light yellow")
        iterations.grid(column=1, row=3, padx=4, pady=4)

        btn3 = tk.Button(self.sub, text="OK", width=14, bg='blue violet', command=lambda: [self.context_calc(neigh,iterations,comp_tx), self.sub.destroy()])
        btn3.grid(column=0, row=4, padx=4, pady=4)

    def save_mat(self,ind): 
        initialdir = os.path.dirname(os.path.abspath(__file__))
        if ind == 0:
            dic = {}
            for i in range(self.k):
                dic[f'Class_{i}'] = self.hidden_mat[:,:,i].ravel()
            pd.DataFrame(dic).to_csv(f"{initialdir}/hidden_mat.csv")
            # print(x)
            self.mat_ind += 1
        elif ind == 1:
            dic = {}
            for i in range(self.k):
                dic[f'Class_{i}'] = self.Probs[:,:,i].ravel()
            pd.DataFrame(dic).to_csv(f"{initialdir}/Probs.csv")
            # print(x)

    def view_K(self,final_belongs_to, jj=False):
        j = 255//self.k-1
        s= final_belongs_to.shape
        img = np.empty((s[0],s[1],3))
        new = final_belongs_to[:,:]
        for i in range(1,self.k):
            new = np.where(new==i,j*i,new)

        for i in range(s[0]):
            for j in range(s[1]):
                a = '{:03b}'.format(int(final_belongs_to[i,j]))
                img[i,j,:] = new[i,j]*np.array([int(a[0]), int(a[1]), int(a[2])])

        self.img_list.append(img)
        self.view(img)

        if jj == True:        
            self.btn31.grid_forget()
        
            btn4 = tk.Button(self.root, text="Prev", width=14, bg='green', command=lambda: self.view_new(-1))
            btn4.grid(column=0, row=1, padx=4, pady=4)

            btn5 = tk.Button(self.root, text="New Win", width=14, bg='red', command=lambda: self.new_window())
            btn5.grid(column=1, row=1, padx=4, pady=4)

            btn6 = tk.Button(self.root, text="Next", width=14, bg='green', command=lambda: self.view_new(1))
            btn6.grid(column=2, row=1, padx=4, pady=4)

            btn7 = tk.Button(self.root, text="Save image", width=14, bg='pink', command=lambda: self.savefile(self.img_list[self.index]))
            btn7.grid(column=1, row=7, padx=4, pady=4)

            self.btn8 = tk.Button(self.root, text="Difference", width=14, bg='brown', command=lambda: [self.difference(),self.btn8.destroy()])
            self.btn8.grid(column=4, row=1, padx=4, pady=4)

    def difference(self):
        diff = np.where(self.clas[0] == self.clas[1],0,255)
        # diff = diff.astype('np.uint8')
        self.img_list.append(diff)
        self.p += 1
        self.view(diff)

    def view_new(self, clic):
        self.img_name.grid_forget()
        self.image_label.grid_forget()
        self.index += clic
        # self.index = max(0,self.index)
        self.index = self.index % self.p
        self.image_label = tk.Label(self.root, image=self.display_list[self.index])
        self.image_label.grid(column=0, row=5, columnspan=9999, padx=4, pady=4)
        self.mat_ind = self.index-1
        if self.index == 0:
            self.title = 'Original'
            self.btn2.destroy()
            self.img_name = tk.ttk.Label(self.root, text=self.title, font=("Times New Roman", 10))
            self.img_name.grid(column=0, row=4, padx=8, pady=4)
        elif self.index == 1:
            self.title = 'K-Means'
            self.btn2 = tk.Button(self.root, text="Save probabilities", width=14, bg='blue violet', command=lambda: self.save_mat(self.mat_ind))
            self.btn2.grid(column=0, row=7, padx=4, pady=4)
            self.img_name = tk.ttk.Label(self.root, text=self.title, font=("Times New Roman", 10))
            self.img_name.grid(column=0, row=4, padx=8, pady=4)
        elif self.index == 2:
            self.title = 'Contextual'
            self.btn2 = tk.Button(self.root, text="Save probabilities", width=14, bg='blue violet', command=lambda: self.save_mat(self.mat_ind))
            self.btn2.grid(column=0, row=7, padx=4, pady=4)
            self.img_name = tk.ttk.Label(self.root, text=self.title, font=("Times New Roman", 10))
            self.img_name.grid(column=0, row=4, padx=8, pady=4)
        else:
            self.title = 'Difference: 0 = no change, 1 = change'
            self.btn2.destroy()
            self.img_name = tk.ttk.Label(self.root, text=self.title, font=("Times New Roman", 10))
            self.img_name.grid(column=0, row=4, padx=8, pady=4)
        
    def view(self, image):
        self.display = self.list_to_PIL(image)
        self.display_list.append(self.display)
        self.image_label = tk.Label(self.root, image=self.display)
        self.image_label.grid(column=0, row=5, columnspan=9999, padx=4, pady=4)

    def list_to_PIL(self, image, original=False):
        if len(image.shape) == 3:
            image = np.flip(image,axis=-1)
        image = Image.fromarray(np.uint8(image))
        self.origin.append(ImageTk.PhotoImage(image))
        size = image.size
        if size[1]>=size[0] and size[1]>580 and original==False:
            s1, s2 = (580*size[0])//size[1], 580
            image = image.resize((s1, s2, size[2]), resample=0) if len(size) == 3 else image.resize((s1, s2), resample=0)
        elif size[0]>size[1] and size[0]>750 and original==False:
            s1, s2 = 750, (750*size[1])//size[0]
            image = image.resize((s1, s2, size[2]), resample=0) if len(size) == 3 else image.resize((s1, s2), resample=0)
        return ImageTk.PhotoImage(image) 

    def new_window(self):
        new_win = tk.Toplevel()
        new_win.title(self.title)
        new_img = self.origin[self.index]
        globals()["self.new_display"+f"_{self.index}"] = new_img
        width, height, depth = self.img_list[0].shape
        canv = tk.Canvas(new_win, relief=tk.SUNKEN)
        canv.config(width=min(width, 1750), height=min(height, 750))
        canv.config(highlightthickness=0)
        sbarV = tk.Scrollbar(new_win, orient=tk.VERTICAL)
        sbarH = tk.Scrollbar(new_win, orient=tk.HORIZONTAL)
        sbarV.config(command=canv.yview)
        sbarH.config(command=canv.xview)
        canv.config(yscrollcommand=sbarV.set)
        canv.config(xscrollcommand=sbarH.set)
        sbarV.pack(side=tk.RIGHT, fill=tk.Y)
        sbarH.pack(side=tk.BOTTOM, fill=tk.X)
        canv.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
        canv.config(scrollregion=(0,0,width,height))
        self.imgtag = canv.create_image(0,0, anchor="nw", image=globals()["self.new_display"+f"_{self.index}"])

GUI()