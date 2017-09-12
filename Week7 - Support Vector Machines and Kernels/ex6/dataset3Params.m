function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
[row col] = size(values);

% Time Complexity is O(n ^2) where n is the number of values. 
for i=1:row
    current_C = values(i);
    for j=1:row
      current_sigma = values(j);  
      
      % Note: we need to train the model for X, y values for the current_C,
      % current_sigma values.
      model= svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
      predictions = svmPredict(model, Xval);
      error = mean(double(predictions ~= yval));
      %fprintf('error for c = %f, sigma = %f is : %f',current_C, current_sigma, error);
      
      % Initialize lowest_error with the error for the first error.
      if(i==1 && j==1)
        lowest_error = error;
      end
      
      if(error < lowest_error)
          lowest_error = error;
          C = current_C;
          sigma = current_sigma;
      end      
    end
end




% =========================================================================

end
