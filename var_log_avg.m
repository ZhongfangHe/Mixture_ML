% Let ui be a K-by-1 vector, a = (1/K) *[1 1 ... 1] * log((u1+u2+...+um)/m)
% What is the variance of a?

function avar = var_log_avg(umat) 
% Inputs:
%   umat: a m-by-K matrix where each row is a draw of u
% Outputs:
%   avar: a scalar of the variance of (1/K) *[1 1 ... 1] * log((u1+u2+...+um)/m) 

[m,K] = size(umat);

nw_cov = Newey_West_longRun_cov(umat);
ucov = nw_cov/m;

umean = mean(umat)';
L = (1/K)*ones(K,1);
scale_vec = L./umean;

avar = scale_vec' * ucov * scale_vec;
