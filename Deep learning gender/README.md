Deep learning gender from name -LSTM Recurrent Neural Networks
------------------------------------------------------------
Deep learning neural networks have shown promising results in problems related to vision, speech and text with varying degrees of success. I have tried looking at a text problem here, where we are trying to predict gender from name of the person. RNNs are a good fit for this as it involves learning from sequences (in this case sequence of characters).we will use character sequences which make up the name as our X variable, with Y variable as m/f indicating the gender. we use a stacked LSTM model and a final dense layer with softmax activation (many-to-one setup). categorical cross-entropy loss is used with adam optimizer. A 20% dropout layer is added for regularization to avoid over-fitting.
More here 
https://medium.com/@prdeepak.babu/deep-learning-gender-from-name-lstm-recurrent-neural-networks-448d64553044 <br>
https://prdeepakbabu.wordpress.com/
