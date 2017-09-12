function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias unit i.e., 1 column in the beginning.
X = [ones(m,1) X];
% fprintf("x %f \n", size(X,2));

% Compute hidden layer activation units.
a = sigmoid(X * Theta1'); % Dim: 5000 x 25

% Add additional bias unit at the beginning to the above activation unit
% matrix.
a = [ones(size(a,1),1) a];  % Dim: 5000 x 26

% Compute output layer.
h = sigmoid(a * Theta2');   % Dim: 5000 x 10

% Get the maximum probability and index (this is the digit) values for each
% row (i.e. input digit) in the X.
[max_probablity, max_index] = max(h,[],2); % Dim: 5000 x 1

% Return the digit prediction for all the rows in X using the index value of
% theta2.
p = max_index;

% =========================================================================


end
