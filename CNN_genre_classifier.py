import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from pathlib import Path
import h5py


DATASET_PATH = Path(r"C:\Users\User\OneDrive - University of Canberra - STAFF\PhD\Studies\HRI\Audio-Detection\data.h5")


#load data function
#def load_data(dataset_path):
#    with open(dataset_path, "r") as fp:
#        data = h5.load(fp)
    
    #covert lists into numpy arrays
#    x = np.array(data["mfcc"])
#    y = np.array(data["labels"])

#    return x, y

def load_data(dataset_path):
    with h5py.File(dataset_path, "r") as hf:
        mfccs = np.array(hf['mfcc'])
        labels = np.array(hf['labels'])
        mappings = list(map(lambda x: x.decode('utf-8'), hf['mappings'][:]))  # Convert bytes to string
    
    return mfccs, labels, mappings

def prepare_datasets(test_size, validation_size):
    
    #load data
    x, y, _ = load_data(DATASET_PATH)
    
    #create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    
    #create train/validation split - split the train set
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)
    
    # For a CNN, tensor flow expects a 3D array for each sample
    #3D array -> (130, 13, 1) <- (no of time been's, mfcc, depth)
    x_train = x_train[..., np.newaxis] #4D array -> (num_samples, 130, 13, 1)
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape):
    
    #create model
    model = keras.Sequential()
    
    #1st convolution layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape)) #(# of kernels, grid size, activation fn, input shape)
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same')) #grid size, stride, padding
    model.add(keras.layers.BatchNormalization())
    
    #2nd convolution layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    #3rd convolution layer
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    #flatten the output and feed it to dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    
    #output layer
    model.add(keras.layers.Dense(10, activation='softmax')) #neurons as many as the genres
    
    return model

def predict(model, x, y):
    
    x = x[np.newaxis, ...]
    
    # prediction is a 2D array = [[0.1,0.2, ...]]
    prediction = model.predict(x) #x -> (1, 130, 13, 1)
    
    #extract index with max value
    predicted_index = np.argmax(prediction, axis=1) #[4]
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

if __name__ == "__main__":
    
    #create train, validation and test sets #validation set is used to evaluate before testing so that the model has never seen the test set
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2) #(test_size, validation_size)
    
    #build the CNN
    input_shape = (x_train.shape[1], x_train.shape[2],x_train.shape[3])
    model = build_model(input_shape)
    
    
    #Compile the CNN
    optimizer = keras.optimizers.Adam(learning_rate=0.0001) 
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    
    
    #train the CNN
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=30, batch_size=32)
    
    
    #evaluate the CNN on the test tet
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    
    #make prediction on a sample
    x = x_test[100] 
    y = y_test[100]
    
    predict(model, x, y)
    
    
    