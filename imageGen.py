from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from tqdm import tqdm


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.1, 0.9]
)

for imageFolder in os.listdir('Images/'):
    try: os.mkdir('newImages')
    except: pass

    for file in os.listdir('Images/'+imageFolder+'/'):
        os.mkdir('newImages/'+imageFolder)
        img = load_img('Images/'+imageFolder+'/'+file)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape) # 1,3,WIDTH, HEIGHT - > 3 is because the image is RGB. [0, 255, 255], if was grey 3 would be 1 [60] 
        i = 0

        for batch in datagen.flow(x, batch_size = 1, save_to_dir='newImages/'+imageFolder,save_prefix=imageFolder, save_format='jpg'):
            i += 1 
            if i > 3: 
                break



