from model import data_loader
from keras.models import Model, load_model
import numpy as np
from PIL import Image
import shutil
import cv2
import os
cwd = os.getcwd()

def videos_to_imgs_skip(path, dest, skip=30):
    filenum = 0
    for video in os.listdir(path):
        if not video.startswith('.'):
            #do something
            vidcap = cv2.VideoCapture(path+video)
            success,image = vidcap.read()
            count = 0
            while success:
                if count%skip == 1:
                    cv2.imwrite(dest+'{0}.png'.format(count), image)     # save frame as JPEG file
                count += 1     
                success,image = vidcap.read()
            filenum +=1
    return

def videos_to_imgs(path, dest):
    filenum = 0
    for video in os.listdir(path):
        if not video.startswith('.'):
            #do something
            vidcap = cv2.VideoCapture(path+video)
            success,image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(dest+'{0}.png'.format(count), image)     # save frame as JPEG file
                count += 1     
                success,image = vidcap.read()
            filenum +=1
    return

def train_to_test(path, dest):
    count = 0
    for img in os.listdir(path):
        if not img.startswith('.'):
            #do something
            skip = 20
            if count%skip == 0:
                shutil.copyfile(path+img, dest+img)  
            count += 1
    return

def convert_frames_to_video(pathIn,pathOut,fps):
    img=[]
    for image in sorted(os.listdir(pathIn), key=lambda x: int(x.replace(".png", ""))):
        if not image.startswith('.'):
            img.append(cv2.imread(pathIn+image))

    height,width,layers=img[1].shape

    video=cv2.VideoWriter(pathOut+'video.avi',0,fps,(width,height))

    for j in range(0,len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()
    return

def evaluate_video(videopath, temppath, videodestpath, abmodelpath):
    model = load_model(abmodelpath)
    #convert video into list of images
    videos_to_imgs(videopath, temppath+'/A/')
    #feed image through AB
    dl = data_loader.DataLoader('bullshitlol')
    imgs_A = []      
    for image in sorted(os.listdir(temppath+'/A/'),key=lambda x: int(x.replace(".png", ""))):
        if not image.startswith('.'):
            imgs_A.append(dl.load_img(temppath+'/A/'+image))
    imgs_A = np.array(imgs_A)
    imgs_A = np.squeeze(imgs_A)
    #delete temp
    # for f in temppath:
    #     os.remove(f)
    #predict
    fake_B = model.predict(imgs_A)
    #rescale
    fake_B = 80*(fake_B)+80
    #convert back to list of images in file
    for i in range(np.size(fake_B,0)):
        cv2.imwrite(temppath+'/B/'+'{0}.png'.format(i), fake_B[i])  
    #convert to video
    convert_frames_to_video(temppath+'/B/', videodestpath+'/video.avi',30)
    return