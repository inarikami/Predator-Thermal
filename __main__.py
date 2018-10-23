from model import cyclegan
import cv2
import os

def videos_to_imgs(path, dest):
    for video in os.listdir(path):
        filenum = 0
        if not video.startswith('.'):
            #do something
            vidcap = cv2.VideoCapture(video)
            success,image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite('{0}frame{1}.jpg'.format(filenum,count), image)     # save frame as JPEG file      
                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
            filenum +=1
    return

videos_to_imgs('','') #trainA
videos_to_imgs('','') #trainB
gan =cyclegan.CycleGAN()
gan.train(epochs=200, batch_size=1, sample_interval=200)