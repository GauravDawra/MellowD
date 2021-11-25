# MellowD
Machine learning project analysing music

### CNN Method
To run the CNN model, do the following:
- Run python3 cnn_mel.py
- If you want to use the already trained model stored in cnn_mel.mdl, type 'y' when prompted, otherwise type 'n'.
- It will train the model on the hyperparameters defined at the top of the file
- By default, learning rate = 0.001, batch size = 4, number of epochs = 100 (change these in file if needed)
- Let it run. When the training ends, you will be asked if you want to save the model. Type 'y' if you want to save the model and 'n' otherwise.
- While the process runs, it creates two graphs:
    1. train_loss_mel_cnn.jpg: training loss vs iteration graph
    2. genre_wise_accuracy_prediction.jpg: genre wise accuracy achieved after training
