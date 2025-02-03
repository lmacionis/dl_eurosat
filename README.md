CNN Image Classification

Tasks to complete:

  Download the data and examine the general information (how many data points are in each class, what the example images look like)
  Prepare the data (split it into separate training, validation, and testing datasets, augment, normalize, etc.)
  Design a model architecture or several (it must have at least several convolutional layers, at least several pooling layers, insert a Dropout layer (s), apply L2 normalization).
  Search for CNN architectures online and try to adapt them to this task. Compare the results.
  Try different hyperparameters (learning rate, number of epochs, batch size).
  Evaluate the model's accuracy, display the Confusion matrix and other metrics.
  Write a short summary about the results.

Results:

Custom CNN Model

    Training Duration: 30 epochs
    Performance Metrics:
      Train loss: 0.1413, Accuracy: 95.37%
       Validation loss: 0.3434, Accuracy: 91.13%
        Testing Accuracy: 91%

Visualizations
Training Loss and Accuracy Graph:

![image](https://github.com/user-attachments/assets/bcf0dcdb-04bf-4e91-b37a-af3db8e2dcae)


Confusion matrix:

![image](https://github.com/user-attachments/assets/538c5cf7-9f0d-4cb0-a850-a731509cc76e)



Summary:

Trainable data accuracy are highly dependent on amount of convolutional layers, batch size, and number of transformations done.
Best result was achieved with low amount of transformations and batch size of 30, bigger or smaller batch size already drops a value of accuracy.
