function [nll,g,H] = LogisticLossSimple_N(w,X,y,weights)
% Negative log likelihood for binary logistic regression
% w: d*1
% X: n*d
% y: n*1, should be -1 or 1

% This file is from pmtk3.googlecode.com

[n, d] = size(X);
if nargin < 4, weights = ones(n,1); end
y01 = (y+1)/2;
mu = sigmoid(X*w);
mu = max(mu, eps); % bound away from 0
mu = min(1-eps, mu); % bound away from 1
nll = -sum(weights .* (y01 .* log(mu) + (1-y01) .* log(1-mu))) / n;
Xw = X .* repmat(colvec(weights), 1, d);
if nargout > 1
  g = Xw'*(mu-y01) / n;
end

end
