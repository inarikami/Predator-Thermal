from model import cyclegan
import cv2
import os
cwd = os.getcwd()

def videos_to_imgs(path, dest):
    filenum = 0
    for video in os.listdir(path):
        if not video.startswith('.'):
            #do something
            skip = 8
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

# videos_to_imgs(cwd+'\\A_Vids\\',cwd+'\\datasets\\wildlands\\trainA\\') #trainA
# videos_to_imgs(cwd+'\\B_Vids\\',cwd+'\\datasets\\wildlands\\trainB\\') #trainB
gan =cyclegan.CycleGAN()
gan.train(epochs=200, batch_size=1, sample_interval=200)