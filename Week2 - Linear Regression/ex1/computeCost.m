function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% y dimensions = m * 1;
% X dimensions = m * 2;
% theta dimensions = 2 * 1;
h = (theta' * X')';   % or X * theta;
meanSquaredSum = sum((h - y) .^ 2); % element wise square and then sum of the m * 1 vector
J = (1/(2*m)) * meanSquaredSum;

% =========================================================================

end
