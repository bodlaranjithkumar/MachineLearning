function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
% theta2 = theta(2,:);
% theta3 = theta(3,:);
J = (1/m) * sum((-y .* log(h)) - ((1 - y).*log(1 - h))) + (lambda/ (2 * m)) * sum(theta([2,3],:) .^ 2) ;

% Calculate the gradient for each feature j
% grad1 = (1/m)*sum((h - y)'*X(1,:));

% grad2 = (1/m)*sum((h - y)'*X(2,:)) + (lambda / m) * theta(2,:);
% grad3 = (1/m)*sum((h - y)'*X(3,:)) + (lambda / m) * theta(3,:);

% gradient vector consists of the values with feature gradient
grad(1:1) = (1/m)*sum((h - y)'*X(:,1));

for iter = 2 : length(theta)
   grad(iter,1) =  (1/m)*sum((h - y)'*X(:,iter)) + (lambda / m) * theta(iter,:);
end


% =============================================================

end
