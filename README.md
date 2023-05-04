Download Link: https://assignmentchef.com/product/solved-cs69012-assignment10-cloth-classification-using-rnn-and-cnn
<br>



The objective of this assignment is to train a Recurrent Neural Network (RNN) and a Convolutional Neural Network (CNN) using Tensorflow, for the task of image classification. You may use Tensorflow 2, but for this assignment we recommend using v1 functionality as follows:

<table width="602">

 <tbody>

  <tr>

   <td width="602">import tensorflow.compat.v1 as tf</td>

  </tr>

  <tr>

   <td width="602">tf.disable_v2_behavior()</td>

  </tr>

 </tbody>

</table>

<strong>Dataset : </strong>Fashion MNIST Dataset. You can load the dataset from tf.keras directly as shown in the following tutorial: <a href="https://www.tensorflow.org/tutorials/keras/classification">https://www.tensorflow.org/tutorials/keras/classification</a> with the following lines of code:

<table width="602">

 <tbody>

  <tr>

   <td width="602">fashion_mnist = tf.keras.datasets.fashion_mnist</td>

  </tr>

  <tr>

   <td width="602"> </td>

  </tr>

  <tr>

   <td width="602">(train_data, train_labels), (test_data, test_labels) =</td>

  </tr>

  <tr>

   <td width="602">fashion_mnist.load_data()</td>

  </tr>

 </tbody>

</table>

Loading the dataset returns four NumPy arrays:

<ul>

 <li>The train_data and train_labels arrays are the <em>training set</em>—the data the model uses to learn.</li>

 <li>The model is tested against the <em>test set</em>, the test_data, and test_labels</li>

</ul>

The images are 28×28 NumPy arrays, with pixel values ranging from 0 to 255.

<strong>Class Labels: </strong>Each training and test example is assigned to one of the following labels:

<ul>

 <li>0 T-shirt/top</li>

 <li>1 Trouser</li>

 <li>2 Pullover</li>

 <li>3 Dress</li>

 <li>4 Coat</li>

 <li>5 Sandal</li>

 <li>6 Shirt</li>

 <li>7 Sneaker</li>

 <li>8 Bag</li>

 <li>9 Ankle boot</li>

</ul>

You should further split the training data into training and validation sets using Scikit Learn’s train_test_split method:

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.htm </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">l</a>

You should record both the training and validation errors at the end of each epoch (do not train on the validation set, just record the error).

You don’t have to submit any data files in your final submission.

You may use the <strong>early stopping </strong>technique to decide when to finish training. A simple algorithm for this would be as follows:

<ol>

 <li>Train on training data.</li>

 <li>Find validation set performance at regular intervals.</li>

 <li>If validation set performance doesn’t improve over k iterations (called patience), stop training. This is called <strong>early stopping.</strong></li>

 <li>Load best model weights based on validation set.</li>

 <li>Find performance on the test set.</li>

</ol>

<h1>Task 1</h1>

You will create a recurrent neural network with a tensorflow.compat.v1.nn.rnn_cell.RNNCell as a basic component. Use tensorflow.compat.v1.nn.static_rnn to create the RNN. Train it end to end.

RNN specifics:

<ol>

 <li>Hidden state size: 100</li>

 <li>Timesteps for unfolding the RNN: 28</li>

 <li>Non-linear activation: tanh</li>

 <li>Loss function: categorical softmax cross entropy with logits</li>

 <li>Regularization: Use dropout with retention probability of 0.8</li>

 <li>Optimizer: Adam</li>

</ol>

Save the weights corresponding to the best validation accuracy in a folder ‘logs_rnn/’ and report the results on the validation and test set.

<h1>Task 2</h1>

You will create a 2D convolutional neural network using tensorflow.compat.v1.layers.conv2d as a basic component. Train it end to end.

CNN specifics:

<ol>

 <li>1st convolutional layer: 32 filters of dimension 5×5 followed by batch normalization, relu activation and max pooling (with 2×2 subsampling).</li>

 <li>2nd convolutional layer: 64 filters of dimension 5×5 followed by batch normalization, relu activation and max pooling (with 2×2 subsampling).</li>

 <li>Flatten above layer and feed it to a densely connected layer with 1024 hidden units</li>

 <li>Add an output dense layer with 10 logits followed by batch normalization and softmax activation.</li>

 <li>Loss function: categorical softmax cross entropy with logits</li>

 <li>Regularization: Use dropout with retention probability of 0.6</li>

 <li>Optimizer: Adam</li>

</ol>

Save the weights corresponding to the best validation accuracy in a folder ‘logs_cnn/’ and report the results on the validation and test set.

Submit a brief report on what inferences you could draw from the experiments (in pdf only). The report should contain information on model hyperparameters and the training, validation and test accuracies.

<strong>Constraints and Considerations:</strong>

<ol>

 <li>You must NOT use tensorflow.keras functions other than for loading the dataset. Define the RNN using the functions in tensorflow.compat.v1.nn only. Similarly use only the functions in tensorflow.compat.v1.layers for the CNN.</li>

 <li>Apart from the above restriction you have the freedom to choose the details of your architecture e.g. learning rate, variable initializers, strides, zero padding etc.</li>

 <li>You should create a train-validation split on the training set and train and test on appropriate data only. You must train properly with a proper stopping criterion. All decisions you take for model parameters must be logical.</li>

 <li>Do NOT hardcode any paths.</li>

 <li>Include all training and testing codes. Two files named ‘rnn.py’ and ‘cnn.py’  must be present. They will take some command line arguments. ‘python rnn.py  –train’ should train the RNN model, ‘python rnn.py –test’ should load all model weights, test the model and report the accuracy on the test data. Similarly for the file ‘cnn.py’.</li>

 <li>Name the ZIP file as well the folder that you compressed to get the zip as</li>

</ol>

“Assignment10_&lt;Roll No.&gt;.zip” and “Assignment10_&lt;Roll No.&gt;” respectively. Upload the zip file to moodle. Note that the zip file should contain your python files, model weights (i.e. the ‘logs_rnn/’ and the ‘logs_cnn/’ folders), and a pdf file explaining your model hyperparameters and results/inferences. Please note that all explanations should be short (1-3 sentences).

<ol start="7">

 <li>Small differences in your method’s accuracy would not be penalized (larger may be; remember to set a fixed random seed for reproducing your result). Your experimentation technique should be sound (like, do not test on training data or train on validation data).</li>

</ol>