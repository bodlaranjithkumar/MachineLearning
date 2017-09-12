function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add 1's column to the input layer as a bias unit
a1 = [ones(m,1) X]; % dim : 5000 * 401

a2 = sigmoid(a1 * Theta1'); % dim: 5000 * 25

% add 1's column to the first hidden layer as the bias unit
a2 = [ones(m,1) a2]; % dim : 5000 * 26

% calculate the hypthosis i.e., the output layer
a3 = sigmoid(a2 * Theta2'); % dim : 5000 * 10

% create a temp y matrix
tempy = zeros(m,num_labels); % dim : 5000 * 10 zeros

% set the values of tempy at index = y values
% TO DO: Can I simplify by not using the for loop?
for row = 1 : m
   y_value = y(row);
   tempy(row,y_value) = 1;
end

% https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA
% Use sum(sum()) for matrix addition. Basically, this is sum of rows and
% then sum of columns to get the final scalar value.
J = (1/m) * sum(sum(-tempy .* log(a3) - (1 - tempy) .* log(1 - a3)));


Theta1_Without_BiasUnit = Theta1;
Theta1_Without_BiasUnit(:,1) = [];

Theta2_Without_BiasUnit = Theta2;
Theta2_Without_BiasUnit(:,1) = [];

J = J + (lambda/(2*m)) * (sum(sum(Theta1_Without_BiasUnit.^2)) + sum(sum(Theta2_Without_BiasUnit.^2)));

%************** part 2 - backpropagation algrotihm ************************
for t = 1 : m
   a1 = [1; X(t,:)'];   % add 1 bias unit to input. Dim: 401 * 1
   
   z2 = Theta1 * a1;    % dim: 25 * 1;
   a2 = sigmoid(z2);
   a2 = [1; a2];        % add 1 bias unit to the 2nd activation layer. Dim: 26 * 1
   
   a3 =  sigmoid(Theta2 * a2);  % dim: 10 * 1
   
   delta3 = a3 - tempy(t,:)';   % dim: 10 * 1
   
   delta2 = (Theta2'*delta3) .* a2 .* (1-a2);
   delta2 = delta2(2:end);          % remove the bias unit
   
   Theta2_grad = Theta2_grad + delta3 * a2'; % dim: 10 * 26
   Theta1_grad = Theta1_grad + delta2 * a1';
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;

% Part 3 - Regularization
Theta2_grad = Theta2_grad + (lambda/m)*([zeros(size(Theta2, 1), 1) Theta2(:,2:end)]);   % set the first column to 0s.
Theta1_grad = Theta1_grad + (lambda/m)*([zeros(size(Theta1, 1), 1) Theta1(:,2:end)]);   % set the first column to 0s.

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
