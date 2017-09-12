function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

theta0 = theta(1,:);
theta1 = theta(2,:);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %==== Calculate theta0, theta1 =========
    h = X * theta;
    theta0 = theta0 - alpha * (1/m) * sum((h - y));
    theta1 = theta1 - alpha * (1/m) * sum((h - y) .* X(:,2));

    %========================================

    J = computeCost(X, y, [theta0; theta1]);    % Calculate Cost for the latest theta0, theta1

    if iter > 1 && J_history(iter - 1) < J      % Exit the loop and the function if the current cost is > previous cost i.e., divergence
        %fprintf('iteration: %f\n', iter);
        break;
    end
    
    theta = [theta0; theta1];                   % Update the theta vector

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = J;

end

end
