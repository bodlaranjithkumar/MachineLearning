function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% sigmoid([1; 0])
% sigmoid([1 0])
% sigmoid([1 -111111; 0 11111])

% exp(x) = e to the power x
g = 1 ./ (1 + exp(-z));



% =============================================================

end
