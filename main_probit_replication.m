% Consider the probit model:
% yt = I{zt>0}, zt = xt'*b + etat, etat~N(0,1)
%
% Compute its marginal likelihood

clear;
dbstop if warning;
dbstop if error;
rng(123456); %reported in text
% rng(987); %validation 1
% rng(3721); %validation 2


%% Gather data for a probit model   
read_file = 'Data_Recession.xlsx';
read_sheet = 'Data'; 
data_y = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:B273'); %1954Q3 to 2021Q3
data_x = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'C2:M273'); %1954Q3 to 2021Q3,11 non-constant regressors
data = [data_y  data_x];
[ntotal,ndata] = size(data);
h = 4; %forecast horizon in quarters
y = data(h+1:ntotal, 1); %recession indicator
x = [ones(ntotal-h,1) data(1:ntotal-h, 2:ndata)]; %lagged regressors
[n,K] = size(x);   
minNum = 1e-100;


%% MCMC
bvar0 = 100;
b_prior_cov = bvar0 * eye(K);
b_prior_cov_inv = b_prior_cov\eye(K);

xtimesx = x' * x;
Ainv = b_prior_cov_inv + xtimesx;
A = Ainv \ eye(K);
Ax = A*x'; 
pmat = eye(n) - x * A * x'; %maybe used repeatedly

burnin = 2000;
ndraws = 5000*2;
ntotal = burnin + ndraws;
disp('PXDA scale starts:');
draws_pxda2.b = zeros(ndraws,K);
draws_pxda2.z = zeros(ndraws,n);
draws_pxda2.s = zeros(ndraws,1);
tic;
z = zeros(n,1);
b = mvnrnd(zeros(K,1),b_prior_cov)';
for drawi = 1:ntotal
    % draw z from p(z|y,b)
    for t = 1:n
        zt_mean = x(t,:) * b;
        if y(t) == 1
            z(t) = zt_mean + trandn(-zt_mean, Inf);
        else
            z(t) = zt_mean + trandn(-Inf, -zt_mean);
        end
    end

    % draw s from p(s|y,z) integrating out b (easier than conditioning on b which will need MH)
    rss = z' * pmat * z;
    s2 = 1/gamrnd(0.5*n, 2/rss);
    s = sqrt(s2);


    % draw b from p(b|y,z,s)
    Ainv = b_prior_cov_inv + xtimesx;
    A = xtimesx \ eye(K);
    zz = z/s;
    a = A * x' * zz;
    b = mvnrnd(a, A)';

    if drawi > burnin
        i = drawi - burnin;
        draws_pxda2.b(i,:) = b';
        draws_pxda2.z(i,:) = z';
        draws_pxda2.s(i) = s;
    end  

    if round(drawi/5000) == (drawi/5000)
        disp([num2str(drawi),' draws out of ', num2str(ntotal), ' have completed!']);
        toc;
    end    
end
disp('PXDA scale is completed!');
toc;
disp(' ');
draws = draws_pxda2;



%% 1. IS: direct Gaussian
tic;
bmean = mean(draws.b)';
bcov = cov(draws.b);
bcov_inv = bcov\eye(K);
b_covec = bcov_inv * bmean;
b_sw = bmean'*bcov_inv*bmean;
bcov_half = chol(bcov)';
logdet_bcov = 2*sum(log(diag(bcov_half)));
logw = zeros(ndraws,1);
eps = randn(ndraws,K);
for drawi = 1:ndraws
    bj = bmean + bcov_half * eps(drawi,:)';
    logpj = -0.5*K*log(2*pi*bvar0)-0.5*sum(bj.^2)/bvar0; %log prior
    zp = normcdf(x*bj);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    logyj = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
    logqj = -0.5*K*log(2*pi)-0.5*logdet_bcov-0.5*bj'*bcov_inv*bj-0.5*b_sw+bj'*b_covec; %log q
    wj = -logqj + logpj + logyj;
    logw(drawi) = wj;
end
wlevel = max(logw);
ew = exp(logw - wlevel);
tmp_mean = mean(ew);
tmp_std = std(ew);
disp('IS: direct Gaussian');
disp(['LML = ',num2str(wlevel+log(tmp_mean)),' with scaled std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');



%% 2. GD: direct Gaussian
tic;
constvec = ones(ndraws,1);
bmean = mean(draws.b)';
bcov = cov(draws.b);
bcov_inv = bcov\eye(K);
b_covec = bcov_inv * bmean;
b_sw = bmean'*bcov_inv*bmean;
bcov_half = chol(bcov)';
logdet_bcov = 2*sum(log(diag(bcov_half)));
logw_gd = zeros(ndraws,1);
logy_gd = zeros(ndraws,1);
for drawi = 1:ndraws
    bj = draws.b(drawi,:)'; 
    logpj = -0.5*K*log(2*pi*bvar0)-0.5*sum(bj.^2)/bvar0; %log prior
    zp = normcdf(x*bj);
    zp1 = zp;
    zp0 = 1-zp;
    idx = find(zp1<minNum);
    tmp = (zp1(idx)-minNum)/minNum; 
    zp1(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    idx = find(zp0<minNum);
    tmp = (zp0(idx)-minNum)/minNum; 
    zp0(idx) = minNum*exp(tmp - 0.5*(tmp.^2)); 
    logyj = sum(y.*log(zp1) + (1-y).*log(zp0)); %log likelihood
    logqj = -0.5*K*log(2*pi)-0.5*logdet_bcov-0.5*bj'*bcov_inv*bj-0.5*b_sw+bj'*b_covec; %log q
    wj = -logqj + logpj + logyj;
    logw_gd(drawi) = wj;
    logy_gd(drawi) = logyj;
end
wtmp = -logw_gd;
wlevel = max(wtmp); %mean(wtmp);
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
[~, hac_std, ~] = HAC_regression(ew,constvec); 
tmp_std = sqrt(ndraws)*hac_std; %tmp_std = std(ew);
disp('GD: direct Gaussian');
disp(['LML = ',num2str(-wlevel-log(tmp_mean)),' with scaled std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');


%% 3. Bridge Sampling: direct Gaussian
tic;
niter = 10;
rho = corr(logy_gd(1:ndraws-1),logy_gd(2:ndraws));
sc = (1-rho)/(1+rho);
logpy_bridge_vec = zeros(niter,1);
for j = 1:niter
    if j == 1
        logpy_last = log(mean(exp(logw)));
    else
        logpy_last = logpy_bridge_vec(j-1);
    end
    fis = logw;
    tmpis = exp(fis)./(1+sc*exp(fis-logpy_last));
    fgd = logw_gd;
    tmpgd = 1./(1+sc*exp(fgd-logpy_last));
    logpy_bridge_vec(j) = log(mean(tmpis)) - log(mean(tmpgd));
end
logpy = logpy_bridge_vec(niter);
vr1 = var(tmpis)/ndraws;
[~, st2, ~] = HAC_regression(tmpgd,constvec);
vr2 = st2^2;
logpy_var = vr1/(mean(tmpis)^2) + vr2/(mean(tmpgd)^2);
logpy_std = sqrt(logpy_var);
disp('Bridge Sampling: direct Gaussian');
disp(['LML = ',num2str(logpy),' with scaled std = ',num2str(sqrt(ndraws)*logpy_std)]);
toc;
disp(' ');


%% 4. Mixture: direct Gaussian
tic;
nw = 51; %should be multiples of 10 plus 1
wgrid = (0:(1/(nw-1)):1)'; %wgrid = rand(nw,1);

umat_IS = logw * wgrid'; 
umat_GD = logw_gd * (wgrid-1)';
wlevel_IS = max(umat_IS);
wlevel_GD = max(umat_GD);
tmpmat_IS = exp(umat_IS-repmat(wlevel_IS,ndraws,1)); 
tmpmat_GD = exp(umat_GD-repmat(wlevel_GD,ndraws,1)); 
logpy_vec = (wlevel_IS + log(mean(tmpmat_IS)) - wlevel_GD - log(mean(tmpmat_GD)))';
logpyis_cov = cov(tmpmat_IS);
logpygd_cov = Newey_West_longRun_cov(tmpmat_GD);
Ais = diag(1./(mean(tmpmat_IS)));
Agd = diag(1./(mean(tmpmat_GD)));
logpy_cov = (Ais*logpyis_cov*Ais + Agd*logpygd_cov*Agd)/ndraws;

weivec = ones(nw,1)/nw;
logpy = weivec' * logpy_vec;
logpy_var = weivec' * logpy_cov * weivec;
logpy_std = sqrt(logpy_var);
disp('Geometric Mixture: direct Gaussian, OLS');
disp(['LML = ',num2str(logpy),' with scaled std = ',num2str(sqrt(ndraws)*logpy_std)]);

logpyvar_vec = diag(logpy_cov);
logpystd_vec = sqrt(logpyvar_vec);
[~,idx] = min(logpyvar_vec);
disp(['Geometric Mixture: direct Gaussian, MinVar at w = ',num2str(wgrid(idx))]);
disp(['LML = ',num2str(logpy_vec(idx)),' with scaled std = ',num2str(sqrt(ndraws)*logpystd_vec(idx))]);

eps = 1e-10;
logpy_cov_inv = (eps*eye(nw)+logpy_cov)\eye(nw);
lvec = ones(nw,1);
weivec = logpy_cov_inv*lvec/sum(sum(logpy_cov_inv));
logpy = weivec' * logpy_vec;
logpy_var = weivec' * logpy_cov * weivec;
logpy_std = sqrt(logpy_var);
disp('Geometric Mixture: direct Gaussian, GLS');
disp(['LML = ',num2str(logpy),' with scaled std = ',num2str(sqrt(ndraws)*logpy_std)]);
toc;
disp(' ');






