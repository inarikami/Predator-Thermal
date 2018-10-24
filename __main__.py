from model import cyclegan
from model import data_loader
from keras.models import Model, load_model
from PIL import Image
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
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def evaluate_video(videopath, temppath, videodestpath, abmodelpath):
    model = load_model(abmodelpath)
    #convert video into list of images
    videos_to_imgs(videopath, temppath)
    #feed image through AB
    dl = data_loader.DataLoader('bullshitlol')
    imgs_A = []
    for image in os.listdir(temppath):
        if not image.startswith('.'):
            imgs_A.append(dl.load_img(temppath+image))
    #predict
    #delete temp
    for f in temppath:
        os.remove(f)
    fake_B = model.predict(imgs_A)
    #convert back to list of images in file
    counter = 0
    for image in fake_B:
        im = Image.fromarray(image)
        im.save(temppath+'img{0}.jpeg'.format(counter))    
        counter += 1  
    #convert to video
    convert_frames_to_video(temppath, abmodelpath+'video.avi',60)
    return

# videos_to_imgs(cwd+'\\A_Vids\\',cwd+'\\datasets\\wildlands\\trainA\\') #trainA
# videos_to_imgs(cwd+'\\B_Vids\\',cwd+'\\datasets\\wildlands\\trainB\\') #trainB
# train_to_test(cwd+'\\datasets\\wildlands\\trainA\\',cwd+'\\datasets\\wildlands\\testA\\') #trainA
# train_to_test(cwd+'\\datasets\\wildlands\\trainB\\',cwd+'\\datasets\\wildlands\\testB\\') #trainB

# gan =cyclegan.CycleGAN()
# gan.train(epochs=60, batch_size=1, sample_interval=200)

evaluate_video()