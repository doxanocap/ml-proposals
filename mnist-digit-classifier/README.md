# Digits Recognition Neural Network

This repository implements a classification solution for hand-written digits using the MNIST dataset.

## Project Overview

### Initial Implementation
The initial phase involved manual implementation inspired by the book "Neural Networks and Deep Learning" [link](http://neuralnetworksanddeeplearning.com/index.html) and educational videos on Deep Learning by 3Blue1Brown [video link](https://www.youtube.com/watch?v=aircAruvnKk&t=754s&pp=ygUNZGVlcCBsZWFybmluZw%3D%3D). The model was built using an analytical approach to Logistic Regression, employing numpy, Stochastic Gradient Descent (SGD), backpropagation, and a custom loss function.

### Advanced Implementation
Subsequently, the methodology advanced through insights from lectures at MIPT [link](http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%28%D0%BA%D1%83%D1%80%D1%81_%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D0%B9%2C_%D0%9A.%D0%92.%D0%92%D0%BE%D1%80%D0%BE%D0%BD%D1%86%D0%BE%D0%B2%29). The refined solution employed the Maximum Likelihood method, utilizing pytorch, torchvision, log_loss, backpropagation, cross_entropy_loss, and the ADAM optimizer.

## Running the Server

To initiate the server, execute the following command:

```bash
python main.py
```

After server initiation, access the HTML page located at "cd client/index.html" in a web browser.