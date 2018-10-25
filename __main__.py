from model import utils
from model import cyclegan
import os
cwd = os.getcwd()

utils.videos_to_imgs_skip(cwd+'\\A_Vids\\',cwd+'\\datasets\\wildlands\\trainA\\', 40) #trainA
utils.videos_to_imgs_skip(cwd+'\\B_Vids\\',cwd+'\\datasets\\wildlands\\trainB\\', 40) #trainB
utils.train_to_test(cwd+'\\datasets\\wildlands\\trainA\\',cwd+'\\datasets\\wildlands\\testA\\') #trainA
utils.train_to_test(cwd+'\\datasets\\wildlands\\trainB\\',cwd+'\\datasets\\wildlands\\testB\\') #trainB

gan =cyclegan.CycleGAN()
gan.train(epochs=100, batch_size=1, sample_interval=200)

utils.evaluate_video(cwd+'/eval/',cwd+'/temp/',cwd,cwd+'/ABmodel_gen1.h5')

# convert_frames_to_video(cwd+'/temp/B/', cwd+'/video.avi',10)
