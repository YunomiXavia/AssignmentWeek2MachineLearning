import numpy as np
class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        - W1: First layer weights; has shape (D, H)
        - b1: First layer biases; has shape (H,)
        - W2: Second layer weights; has shape (H, C)
        - b2: Second layer biases; has shape (C,)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two-layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        hidden_layer = np.maximum(0, X.dot(W1) + b1)  # ReLU activation
        scores = hidden_layer.dot(W2) + b2

        if y is None:
            return scores

        # Compute the loss
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # Compute the gradients
        grads = {}
        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        grads['W2'] = np.dot(hidden_layer.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)
        dhidden = np.dot(dscores, W2.T)
        dhidden[hidden_layer <= 0] = 0

        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)

        grads['W1'] += reg * W1
        grads['W2'] += reg * W2

        return loss, grads

    def train(self, X, y, X_val, y_val, learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array of shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.

        Returns:
        A dictionary containing the following:
        - loss_history: List containing the value of the loss at each training iteration.
        - train_acc_history: List containing the training accuracy at each epoch.
        - val_acc_history: List containing the validation accuracy at each epoch.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.params
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Update the parameters
            for param in self.params:
                self.params[param] -= learning_rate * grads[param]

            if verbose and it % 100 == 0:
                print(f'iteration {it} / {num_iters}: loss {loss}')

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        hidden_layer = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        scores = hidden_layer.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        return y_pred
