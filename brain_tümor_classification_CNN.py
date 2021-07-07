#%%

"""
BRAIN TUMOR CLASSIFICATION
"""

#%% Import Libraries
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from glob import glob

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
#%% Read the dataset as a training and testing

train_path = "Training/"
test_path = "Testing/"

img = load_img(train_path + "pituitary_tumor/p (1).jpg")
plt.imshow(img)
plt.axis("off")
plt.show()


x = img_to_array(img)
print(x.shape) # (512, 512, 3)


className = glob(train_path + "/*")
numberOfClass = len(className)

print("Number of classes: {}".format(numberOfClass))  # Number of classes: 4


#%% Convolutional Neural Network (CNN)

model = Sequential()
model.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = x.shape))
model.add(Activation("relu")) # activation function as relu
model.add(MaxPooling2D())


model.add(Conv2D(filters = 32,kernel_size = (3,3)))
model.add(Activation("relu")) # activation function as relu
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32,kernel_size = (3,3)))
model.add(Activation("relu")) # activation function as relu
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32,kernel_size = (3,3)))
model.add(Activation("relu")) # activation function as relu
model.add(MaxPooling2D())


# Flatten Operation
model.add(Flatten())
model.add(Dense((1024))) # 1024 is a number of neurons in the model
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass)) # output must be number of classes as a quanttiy
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = Adam(learning_rate=0.0001),
              metrics = ["accuracy"]
              )



batch_size = 32

#%% Data Generator Train-Test Split
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.3,
                                   horizontal_flip=True,
                                   zoom_range = 0.3
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size = x.shape[:2],
                                                    batch_size = batch_size,
                                                    color_mode = "rgb",
                                                    class_mode = "categorical"
                                                    )


test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size = x.shape[:2],
                                                    batch_size = batch_size,
                                                    color_mode = "rgb",
                                                    class_mode = "categorical"
                                                    )


#%% Training the Model (Fit)

hist = model.fit_generator(generator = train_generator,
                           steps_per_epoch=1600 // batch_size,
                           epochs = 100,
                           validation_data= test_generator,
                           validation_steps=800//batch_size
                           )


#%% Save Model

model.save_weights("Classifier_Of_Brain_Tümor.h5")





#%% Model Evaluation
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()