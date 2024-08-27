# Audio-Detection
Binary Audio Detection module for Self Modulating System


Dataset:


I have combined multiple datasets to create the dataset for this project. 
Combined datasets were: esc-50, Asif et al (2022), and freesound
With 971 audio files for "Emergency" situation, 2006 audio files for "Non-Emergency" situation, in total I had 2977 data.
Out of which 25% were used as test data and 20% were used as validation data.




Model:


I have used a CNN model to detect the context based on audio data. 

For the CNN model: I have considered one input layer, 3 convolutional layers, and one output layer. It is a Sequential model with below layers,

1st convolution layer - Has 32 kernels of 3*3 grids, Activation function is RELU, Used Max Pooling of 3*3 grids with 2*2 strides

2nd convolution layer - Has 32 kernels of 3*3 grids, Activation function is RELU, Used Max Pooling of 3*3 grids with 2*2 strides

3rd convolution layer - Has 32 kernels of 2*2 grids, Activation function is RELU, Used Max Pooling of 2*2 grids with 2*2 strides

Then flattened the output and fed it into a dense layer. 30% dropouts were considered.       

Output layer was a Dense layer with 2 neurons, and Softmax activation.

Accuracy: ![image](https://github.com/user-attachments/assets/aeedb148-bb8f-452c-a54a-e6974df2781c)

