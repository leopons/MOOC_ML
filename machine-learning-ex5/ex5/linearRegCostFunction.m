function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


htheta = theta' * X';
squarredErrors = (htheta-y').^2;
J = ( 1 / (2*m) ) * ( sum(squarredErrors) + lambda * (theta.^2)'*[0;ones(size(theta,1)-1,1)] );

thetaSans0 = theta(2:end);
reg = [ 0; (lambda/m)*thetaSans0 ];
grad = (1/m)*(htheta-y')*X + reg';

% =========================================================================

grad = grad(:);

end
