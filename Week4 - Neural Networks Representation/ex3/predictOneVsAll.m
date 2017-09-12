function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);    % # of rows = 10 ie number of digits to predict with 0 represented as 10.
  
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];         % Changes X dimensions to 5000 x 401.

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%  Dimensions 5000 x 401 * 401 x 10 = 5000 x 10. 
%  Given max syntax returns the maximum value in each row resulting in 5000 x 1 matrix. 
%  Each value in the column denotes the number prediction of the image.

h = sigmoid(X * all_theta');
[max_probability, max_index] = max(h,[],2);

% Mistake was to use the max_probability. 
% Use index of the theta matrix which returns the value at that index with maximum probability.
p = max_index;   

% =========================================================================


end
