# multitask training of RNN models

Pytorch implementation of multitask RNN training (original TensorFlow code [here](https://github.com/gyyang/multitask)):

> "Task representations in neural networks trained to perform many cognitive tasks." Guangyu Robert Yang, Madhura R. Joglekar, H. Francis Song, William T. Newsome & Xiao-Jing Wang (2019) [*Nature Neuroscience* Volume 22, pp. 297–306](https://www.nature.com/articles/s41593-018-0310-2)


This code trains a RNN model for multiple types of cognitive tasks. `RNN_rate_dynamics.py` is a custom RNN implementation of continuous-time **rate-neuron network dynamics**, which is commonly used in neuroscience models:

![equation](https://latex.codecogs.com/gif.latex?\bg_white&space;\tau&space;\dot{h}&space;=&space;-&space;h&space;&plus;&space;\sigma(W_{hh}&space;h&space;&plus;&space;W_{ih}&space;s))

where $h$ is neural (hidden) state,  $s$ is (sensory) input, and $W_{hh}, W_{ih}$ are synaptic weight parameters (recurrent and input weights).
$\tau$ is the synaptic integration time constant. 
Note that this model is defined in continuous-time, $\dot{h} = f(h(t))$, 
whereas most deep-learning models use discrete-time descriptions: $h_{t+1} = f(h_{t})$. Euler-integration is then used to simulate the model with discrete time steps. 

Here's a sample code for running a RNN model:

```
import torch
from RNN_rate_dynamics import RNNLayer

T, batch = 1000, 100
n_input, n_rnn, n_output = 10, 500, 5

rnn  = RNNLayer(n_input, n_rnn, torch.nn.ReLU(), 0.9, True)   # input_size, hidden_size, nonlinearity, decay, bias
```

The main training code is defined in `multitask/train.py`: 
Here is a sample code to train the model

```
import multitask

hp, log, optimizer = multitask.set_hyperparameters(model_dir='debug', hp={'learning_rate': 0.001}, ruleset='mante') #, rich_output=True)
run_model = multitask.Run_Model(hp, RNNLayer)
multitask.train(run_model, optimizer, hp, log)
```

Finally, let's visualize the results. The analysis functions are in `multitask/standard_analysis.py`

```
rule = 'contextdm1'
multitask.standard_analysis.easy_activity_plot(run_model, rule)
```


This repo is prepared for the [Harvard-MIT Theoretical and Computational Neuroscience Journal Club](https://compneurojc.github.io/).
Check out the full RNN tutorial repo [here](https://github.com/jennhu/rnn-tutorial).

