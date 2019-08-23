import os
from Config import*
import random
from PIL import Image
# import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class data_prepare(object):

    def __init__(self, train_path, test_path, val_path, batch_size):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.batch_size = batch_size

        class_name = os.listdir(train_path)
        file_name = os.listdir(train_path + '//' + class_name[0])
        print(class_name)
        file_path = train_path + '//' + class_name[0] + '//' + file_name[0]
        img = Image.open(class_name)

        self.img_width,  self.img_height = img.size
        

    def get_data(self, path):
        data_datagen = ImageDataGenerator(rescale=1. / 255)
        data_generator = data_datagen.flow_from_directory(
            path,
            target_size=(self.img_width, self.img_height),
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=random.randint(0, 100))

        return data_generator

    def data_prepare(self):
        train_generator = self.get_data(self.train_path)
        test_generator = self.get_data(self.test_path)
        val_generator = self.get_data(self.val_path)

        return train_generator, test_generator, val_generator


    # def preprocession_image(self, file_path):
    #     for train_test_val in os.listdir(file_path):
    #         train_test_val = os.path.join(file_path, train_test_val)
    #         for car_house_file in os.listdir(train_test_val):
    #             car_house_file = os.path.join(train_test_val, car_house_file)
    #             for file_name in os.listdir(car_house_file):
    #                 image_path = os.path.join(car_house_file, file_name)
    #                 img = cv2.imread(image_path)
    #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                 img = cv2.resize(img, (255, 255))
    #                 cv2.imwrite(image_path, img)