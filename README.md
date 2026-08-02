# 4-Layer Deep Neural Network from Scratch

### 1. Overview
This implementation features a 4-layer deep neural network (Input Layer, 2 Hidden Layers, and Output Layer) built entirely from scratch using Python and NumPy. This is designed to demonstrate the fundamental mechanics of machine learning without relying on high-level frameworks like TensorFlow or PyTorch. It features manual weight initialization, matrix multiplications, forward propagation, and backpropagation via gradient descent.

![Architecture](https://raw.githubusercontent.com/Mazinghorab/Deep-Neural-Network/main/The_architecture.png "The architecture")

### 2. Mathematical Foundations
The network learns by pushing data through successive layers, calculating the error, and propagating that error backward to adjust the weights. Here is the mathematical engine driving the model:

### Forward Propagation
The network utilizes the `tanh` activation function for the hidden layers to introduce non-linearity, and a `softmax` activation function at the output layer to generate class probabilities.

$$
\begin{aligned} 
z_1 &= XW_1 + b_1 \\ 
a_1 &= \tanh(z_1) \\ 
z_2 &= a_1W_2 + b_2 \\ 
a_2 &= \tanh(z_2) \\ 
z_3 &= a_2W_3 + b_3 \\ 
\hat{y} &= \text{softmax}(z_3) 
\end{aligned}
$$

### Loss Function
To evaluate the model's performance, *the Categorical Cross-Entropy* Loss function calculates the divergence between the predicted probability distribution ($\hat{y}$) and the true one-hot encoded labels ($y$). 

$$
L(y, \hat{y}) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i \in C} y_{n,i} \log \hat{y}_{n,i}
$$

### Backpropagation
The model computes the error at the output layer and backpropagates it to update the weights and biases across the network.

*Error Terms*:

$$
\begin{aligned} 
\delta_3 &= \hat{y} - y \\ 
\delta_2 &= (1 - \tanh^2 z_2) \circ \delta_3 W_3^T \\ 
\delta_1 &= (1 - \tanh^2 z_1) \circ \delta_2 W_2^T 
\end{aligned}
$$

*Gradients*:

$$
\begin{aligned} 
\frac{\partial L}{\partial W_3} &= a_2^T \delta_3 \\ 
\frac{\partial L}{\partial W_2} &= a_1^T \delta_2 \\ 
\frac{\partial L}{\partial W_1} &= X^T \delta_1 
\end{aligned}
$$

### 3. Architecture & Matrix Dimensions
The network architecture requires precise dimensional scaling:

**$X$ (Input Data):** Shape `(num_examples, nn_input_dim)`

**Layer 1 ($W_1$ & $b_1$):** $W_1$ is initialized as `(nn_input_dim, nn_hdim)`. This maps the raw input features to the first hidden layer. $b_1$ is `(1, nn_hdim)`.

**Layer 2 ($W_2$ & $b_2$):** $W_2$ is initialized as `(nn_hdim, nn_hdim)`. This maps the output of the first hidden layer to the second hidden layer. $b_2$ is `(1, nn_hdim)`.

**Layer 3 / Output ($W_3$ & $b_3$):** $W_3$ is initialized as `(nn_hdim, nn_output_dim)`. This maps the final hidden representation to the output classes. $b_3$ is `(1, nn_output_dim)`.

## 4. Setup and Usage
To run the neural network on your local machine, follow these steps:

*1. Clone the repository:*
```bash
git clone https://github.com/Mazinghorab/Deep-Neural-Network.git
cd Deep-Neural-Network
```

*2. Install Dependencies*:
The code relies on standard scientific computing libraries.

```bash
pip install numpy matplotlib scikit-learn
```

*3. Run the Code*:
Excute the Jupyter Notebook or Python script to view the training process and the generated decision boundary visualizations.
