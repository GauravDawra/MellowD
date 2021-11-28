# MellowD
Machine learning project analysing music

### CNN Method
In this method, we try to predict the genre of a song using CNN over a MEL-spectogram image of a song. The dataset used GTZAN. To know how we converted a song into mel spectogram, please look at the suitable section. The dimensions of a mel-spectogram was taken to be 100 pixels X 150 pixels. We could achieve a test accuracy of 70.5%. The train test split was 80-20. The model with the mentioned accuracy is stored as a *.mdl* file.
To run the *CNN* model, do the following:
- Run `python3 cnn_mel.py`
- If you want to use the already trained model stored in cnn_mel.mdl, type 'y' when prompted, otherwise type 'n'.
- It will train the model on the hyperparameters defined at the top of the file
- By default, learning rate = 0.001, batch size = 4, number of epochs = 100 (change these in file if needed)
- Let it run. When the training ends, you will be asked if you want to save the model. Type 'y' if you want to save the model and 'n' otherwise.
- While the process runs, it creates two graphs:
    1. **train_loss_mel_cnn.jpg**: training loss vs iteration graph
    2. **genre_wise_accuracy_prediction.jpg**: genre wise accuracy achieved after training

### For MEL-spectogram formation
Here is the link for the google colab notebook that contains the preprocessing and image formation code: https://colab.research.google.com/drive/11WdI9c6J3hFqAMU_CnDWgzsm8cnuyktR?usp=sharing. This code converts a 30 seconds song into a MEL-spectogram using STFT. The image length is 100 pixels by 150 pixels. *MEL-spectogram.ipynb* (in the repository) also contains a copy of the colab notebook.
You will also need access to the GTZAN dataset which is stored in my drive. Here is the link for our MellowD folder: https://drive.google.com/drive/folders/1AOuteJ0NNRGR9ke_bXv9ajvOTSpH8EyB?usp=sharing
Make sure this MellowD folder is a part of your drive too before running the code.
- The fourth and fifth cells were just used to visualize the data and the expected output, so they won't run unless you provide the specific files. 
- Run the rest of the cells in order to produce the output which will be saved in your *MellowD/genres/mel_data folder* in your drive.

### ANN method
In this method, we try to predict the genre of a song using ANN of a song. The dataset used is GTZAN. we have used MFCC matrix to convert audio data to matrix.  
The ANN_Genre.ipynb file has been shared which can be tested. The model has an accuracy of nearly 59% as can be seen through the code. we have split the data into Train, Test by 8 : - 2 split. Then we trained the ANN model, used ’tanh’ activation function and learning rate to be 0.8, maximum iteration as 150. At the hidden layer we used ’softmax’ activation function. Batch size was taken to 64.

### For Popularity prediction
Run `python3 Popularity_prediction.py` to get the regression results on popularity prediction dataset (spotify dataset). Please include tracks.csv file in suitable directory.
This will show results of Lasso regression, Ridge regression etc on the dataset. We can see that the accuracy achieved is very high even with these simple models.
