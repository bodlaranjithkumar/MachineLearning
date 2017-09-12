function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

total_examples = size(X,1);

% Time Complexity is O(total_examples * K);
for i = 1 : total_examples
    currentX = X(i,:);
    currentX = currentX(:); % convert current example to column vector to compute the norm.
    
    idx(i) = 1;
    %min_norm = 0;
    
    for j = 1 : K
        current_centroid = centroids(j,:);
        current_centroid = current_centroid(:); % convert current centroid to column vector to compute the norm.
        current_norm = sum((currentX - current_centroid) .^2);
        
        if(j == 1)
            min_norm = current_norm;
        end
        
        if(j > 1 && current_norm < min_norm)
%             fprintf("current_norm: %f\tmin_norm:%f\tj:%f",current_norm, min_norm, j);
            min_norm = current_norm;
            idx(i) = j;
        end
    end    
end





% =============================================================

end

