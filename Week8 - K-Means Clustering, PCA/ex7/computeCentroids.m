function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% assigned_examples_indices = idx == 1;
% assigned_examples = X(assigned_examples_indices,:);
% assigned_examples_mean = sum(assigned_examples);
% [r c] = size(assigned_examples);
% average_x_subset_sum = assigned_examples_mean ./ r;
% fprintf("assigned_examples_indices:%f",assigned_examples_indices);
% fprintf("example_value:%f",x_subset);
% fprintf("x_subset_sum:%f",x_subset_sum);
% fprintf("mean_x_subset_sum:%f",average_x_subset_sum);

for i = 1 : K
    assigned_examples_indices = idx == i;   % find example number assigned to centroid i.
    assigned_examples = X(assigned_examples_indices,:);  %Get the examples assigned to centroid i using indices from above.
    assigned_examples_mean = sum(assigned_examples); % calculate the column wise mean of the examples.
    [r c] = size(assigned_examples);    % find the number of examples assigned to centroid i.
    centroids(i,:) = assigned_examples_mean ./ r; % calculate the average of the examples which is the new coordinates for centroid i.   
end







% =============================================================


end

