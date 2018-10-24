from model import cyclegan
from model import data_loader
from keras.models import Model, load_model
import shutil
import cv2
import os
cwd = os.getcwd()

def videos_to_imgs(path, dest):
    filenum = 0
    for video in os.listdir(path):
        if not video.startswith('.'):
            #do something
            skip = 40
            vidcap = cv2.VideoCapture(path+video)
            success,image = vidcap.read()
            count = 0
            while success:
                if count%skip == 0:
                    cv2.imwrite(dest+'{0}frame{1}.jpg'.format(filenum,count), image)     # save frame as JPEG file      
                success,image = vidcap.read()
                count += 1
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

def evaluate_video(videopath, videodestpath, modelAtoBpath):
    model = load_model(modelAtoBpath)
    data_loader.
    #convert video into list of images
    for video in os.listdir(videopath):
        if not video.startswith('.'):
            #do something
            input_imgs = []
            vidcap = cv2.VideoCapture(videopath+video)
            success,image = vidcap.read()
            while success:
                input_imgs.append(image)
                success,image = vidcap.read()


    return

# videos_to_imgs(cwd+'\\A_Vids\\',cwd+'\\datasets\\wildlands\\trainA\\') #trainA
# videos_to_imgs(cwd+'\\B_Vids\\',cwd+'\\datasets\\wildlands\\trainB\\') #trainB
# train_to_test(cwd+'\\datasets\\wildlands\\trainA\\',cwd+'\\datasets\\wildlands\\testA\\') #trainA
# train_to_test(cwd+'\\datasets\\wildlands\\trainB\\',cwd+'\\datasets\\wildlands\\testB\\') #trainB




gan =cyclegan.CycleGAN()
gan.train(epochs=60, batch_size=1, sample_interval=200)