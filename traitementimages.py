import tensorflow as tf
import cv2
import os
import numpy as np
from tqdm import tqdm

def load(filename):
    image=cv2.imread(filename,1)
    return(image)


def afficher(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)



def load_sans_classe(dataset):
    L=[]
    l=os.listdir(dataset)
    for elt in tqdm(l):
        img=load(dataset+ "/" + elt)
        L.append(img*(1/255))
    return(L)
A=load_sans_classe("Dataset_Test")
def load_avec_classes(dataset):
    L=[]
    l=os.listdir(dataset)
    if ("png" in dataset) or ("JPEG" in dataset):
        img=load(dataset)
        return(img)
    for elt in l:
        L=L+load_avec_classes(dataset+"/"+elt)
    return(L)

def resize_image(img,new_dim):
    res=tf.image.resize([img],new_dim)[0]
    return(res)
def resize_list(l,new_dim):
    L=[]
    for elt in l:
        L.append(resize_image(elt,new_dim))
    return L
def crop_image(img,new_dim):
    crop=tf.image.random_crop(img,new_dim)
    return(crop)
def min_dim(dataset):
    l=A
    besta,bestb,lol=l[0].shape
    for elt in l:
        a,b,c=elt.shape
        besta=min(a,besta)
        bestb=min(b,bestb)
    if a<128 or b<128:
        print("IMAGE DE TAILLE INFERIEURE A 128x128 !!!")
    return (besta-besta%4,bestb-bestb%4)
        


def crop_dataset_upscale(dataset):
    L=A
    lres=[]
    new_dim=min_dim(dataset)
    a,b=new_dim

    for elt in tqdm(L):
        lres.append(crop_image(elt,(a,b,3)))
    
    return(lres)



def resize_dataset(dataset):
    L=crop_dataset_upscale(dataset)
    a,b=min_dim(dataset)
    new_dim=(a//4,b//4)
    return tf.image.resize(L,new_dim)

