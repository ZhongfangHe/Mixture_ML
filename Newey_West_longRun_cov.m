% Given a time series x, compute its long-run covariance matrix by Newey-West method
% sqrt(n) * mean(x) ~ N(0, cov), where cov is the long-run variance
% that is, mean(x) ~ N(0,s/n)


function nw_cov = Newey_West_longRun_cov(x)
% Inputs:
%   x: a n-by-K vector of time series data
% Outputs:
%   nw_cov: a K-by-K matrix of the long-run variance 

n = size(x,1);
xmean = mean(x);
u = x - repmat(xmean,n,1);

nlag = floor(4 * ((n/100)^(2/9)));

cov0 = u'*u/n;   
nw_cov = cov0;
for j = 1:nlag
    tmp = u(1:(n-j),:)'*u((1+j):n,:)/(n-j);
    weight = 1 - j / (nlag + 1); 
    nw_cov = nw_cov +  weight* (tmp + tmp');
end

