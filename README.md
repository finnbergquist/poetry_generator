# AI Poetry Generator App by Finn Bergquist

### Description

This intelligent poetry generator uses a five-layer recurrent neural network,
trained on over 17,000 lines of poetry from a large assortment of famous poets.
The network in this repository pretrained, but it can be trained differently as
with other hyperparameters. (see How to Run Program section)
The neural network predicts the sequential order of words, allowing for
a user to ask for the next word when generating poetry. The frontend of this
project utilizes that capability, combining the autonomous way to add a next
word with a manual way as well. The user could theoretically write a few words,
then ask the AI for its predictions for the following word and then respond
back to the AI. Noise was added for the next word selection process to ensure
that similar results were not repeated due to the deterministic nature of the
pretrained network.

The network is trained, loaded, and queried from the poetry_agent class which
is implemented in the Flask app backend. The frontend is written in html/css,
sending backend calls for each different functionality of the app(ex. store the
poetry information, evaluate saved poems, speak poems, ask for a new AI word,
etc.)

The saved poetry can be viewed in the Recorded Poems tab, where they can be
audibly read aloud. Additionally, this is where the evaluative RMSE
(root mean squared error) metric can be seen. The purpose of this metric is to
show how far the generated poetry deviates from the deterministic policy of
the neural network. If the highest rated output word was always chosen, the
RMSE score would be zero. For example, say the seed text is:

```
This is ___
```

The neural network output might look like:

```
[0.234, 0.546, 0.456]
```

corresponding to:

```
[cheese, sparta, fun]
```

With added noise, the system might still select the word "fun", and then this
words contribution to the RMSE would be

```
(y'-y)^2 = (0.546 - 0.456)^2
```

where y = the selected word and y'= the argmax word

### How to Run the Program

### Install Dependencies

* numpy
* keras
* tensorflow
* sklearn
* flask

For each of these, run:
```
pip install dependency_name
```

### Start a local server running the app:
```
flask run
```
This will start a local development server with a link in the terminal where the web app can be accessed

### Re-training the Model:

Adjust the hyperparameters in the poetry_agent __init__() method however you like and run
```
pip train.py
```


