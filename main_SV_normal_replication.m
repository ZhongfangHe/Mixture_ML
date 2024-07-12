% Consider the linear regression model with SV:
% yt = xt'*b + N(0,exp(zt)), zt = (1-phi)*u + phi*ztm1 + etat, etat~N(0,s)
% compute its marginal likelihood

clear;
dbstop if warning;
dbstop if error;
rng(123456); %reported in text
% rng(987); %validation 1
% rng(3721); %validation 2


%% Gather data 
read_file = 'Data_Inflation.xlsx';
read_sheet = 'Data2'; %change of inflation rate
data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B3:V222');    
[ng,nr] = size(data);
inflation = data(:,1);
reg = data(:,2:nr);    
y = inflation(2:ng); %change
uset = 1:(nr-1);
x = [ones(ng-1,1) inflation(1:(ng-1)) reg(1:(ng-1),uset)]; %full 
[n,K] = size(x);
disp(['n = ',num2str(n), ', K = ', num2str(K)]);
    

%% MCMC
tic;
ndraws = 5000*2; 
burnin = 2000;
disp(['burnin = ',num2str(burnin),', ndraws = ',num2str(ndraws)]);
ntotal = burnin + ndraws;

b0_mean = zeros(K,1);
b0_var = 100*ones(K,1); %prior: reg coefficients

muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
phih_mean = 0.95; phiha = 100; phihb = phiha*(1-phih_mean)/phih_mean; % AR(1): phi = 2* betarnd(a,b))-1
priorSV = [muh0 invVmuh phiha phihb]'; 
muh = muh0 + sqrt(1/invVmuh) * randn;
phih = 0.5*(1+betarnd(phiha,phihb)); 
sigh2_s = 1;
sigh2 = gamrnd(0.5,2*sigh2_s);
sigh = sqrt(sigh2); %prior: SV

b0_OLS = regress(y,x);
resid_OLS = y - x*b0_OLS;
hSV = log(var(resid_OLS))*ones(n,1); %initialize SV by log OLS residual variance.

draws.b = zeros(ndraws,K);
draws.SVpara = zeros(ndraws,4); % [mu phi sig2 sig]
draws.z = zeros(ndraws,n); 
for drawi = 1:ntotal
    yvar = exp(hSV);
    ystd = sqrt(yvar); 
    yy = y./ystd;
    xx = x./repmat(ystd,1,K);
    Binv = diag(1./b0_var) + xx'*xx;    
    Binvb = xx'*yy; 
    tmp = mvnrnd(Binvb,Binv)';
    b = Binv\tmp;
    
    resid = y - x*b;
    logz2 = log(resid.^2 + 1e-100);
    [hSV, muh, phih, sigh] = SV_update2(logz2, hSV, ...
        muh, phih, sigh, sigh2_s, priorSV);     
    if drawi > burnin
        i = drawi-burnin;
        draws.b(i,:) = b';
        draws.z(i,:) = hSV';
        draws.SVpara(i,:) = [muh phih sigh^2 sigh];
    end   
end
disp('MCMC is completed!');
toc;
disp(' ');


%% 1. IS: direct Gaussian
tic;
u = draws.SVpara(:,1);
rho = draws.SVpara(:,2);
s = draws.SVpara(:,4);
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

constvec = ones(ndraws,1);
R2vec = zeros(n,1);
ISpara = zeros(n,3+KK); %const, htm1, para_est, var(resid)
resid_mat = zeros(ndraws,n);
for t = 1:n
    ht = draws.z(:,t);
    if t > 1
        htm1 = draws.z(:,t-1);
        xx = [constvec htm1 para_est];
    else
        xx = [constvec para_est];
    end 
    yy = ht;
    Kxx = size(xx,2);
    Binv = eye(Kxx)/10000 + xx'*xx;
    Binvb = xx'*yy;
    coef = Binv\Binvb;
    yyfit = xx*coef;
    resid = yy-yyfit;
    if t > 1
        ISpara(t,:) = [coef' var(resid)];
    else
        ISpara(t,:) = [coef(1) 0 coef(2:Kxx)' var(resid)];
    end
    R2vec(t) = var(yyfit)/var(yy);
    resid_mat(:,t) = resid;
end %calibrate AR(1) IS for latent variable

nsim = ndraws*1;
logh = zeros(nsim,1);
logp = logh;
logy = logh;
logqh = logh;
logqp = logh;
logw = logh;
hj = zeros(n,1);
hj_mean = zeros(n,1);
for drawi = 1:nsim
    paraj = mvnrnd(para_mean, para_cov)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    rhoj2 = rhoj^2; %simulate theta from IS   
    
    for t = 1:n
        if t == 1
            xxt = [1 0 paraj'];
        else
            xxt = [1 hj(t-1) paraj']; 
        end
        hj_mean(t) = xxt * ISpara(t,1:(KK+2))';
        hj(t) = hj_mean(t) + sqrt(ISpara(t,(KK+3)))*randn;
    end %simulate h from IS
    
    logpj_b = -0.5*K*log(2*pi) -0.5*sum(log(b0_var)) -0.5*sum(((bj-b0_mean).^2)./b0_var);
    logpj_SV = -0.5*log(2*pi/invVmuh) - 0.5*((uj-muh0)^2)*invVmuh ...
        +phiha*log(1+rhoj)+phihb*log(1-rhoj)-(phiha+phihb)*log(2)-betaln(phiha,phihb) ...
        -0.5*log(2*pi*sigh2_s) - 0.5*s2j/sigh2_s + 0.5*log(s2j); 
    logpj = logpj_b + logpj_SV; %prior theta
     
    tmp = (hj(2:n)-(1-rhoj)*uj-rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h    
    
    eps = y - x*bj;
    logyj = -0.5*n*log(2*pi) - 0.5*sum(hj+eps.*eps.*exp(-hj)); %likelihood
    
    resid2 = (hj - hj_mean).^2;
    ISd = ISpara(:,KK+3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISd)) -0.5*sum(resid2./ISd); %IS h
    
    tmp = paraj - para_mean;
    logqpj = -0.5*KK*log(2*pi) - 0.5*logdet_para_cov - 0.5*tmp'*para_covinv*tmp; %IS theta
    
    logh(drawi) = loghj;
    logp(drawi) = logpj;
    logy(drawi) = logyj;
    logqh(drawi) = logqhj;
    logqp(drawi) = logqpj; 
    logw(drawi) = logpj+loghj+logyj-logqpj-logqhj;
end
wtmp = logw;
wlevel = max(wtmp); %mean(wtmp);
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
tmp_std = std(ew);
disp('IS: direct Gaussian');
disp(['LML = ',num2str(wlevel+log(tmp_mean)),' with scaled std = ',num2str(tmp_std/tmp_mean)]);
toc;
disp(' ');



%% 2. GD: direct Gaussian
tic;
u = draws.SVpara(:,1);
rho = draws.SVpara(:,2);
s = draws.SVpara(:,4);
para_est = [draws.b  u  log(1+rho)-log(1-rho)  log(s.^2)];
KK = size(para_est,2);
para_mean = mean(para_est)';
para_cov = cov(para_est);
para_covhalf = chol(para_cov)';
para_covinv = para_cov\eye(KK);
logdet_para_cov = 2*sum(log(diag(para_covhalf))); %calibrate Gaussian IS for fixed para

constvec = ones(ndraws,1);
R2vec = zeros(n,1);
ISpara = zeros(n,3+KK); %const, htm1, para_est, var(resid)
resid_mat = zeros(ndraws,n);
for t = 1:n
    ht = draws.z(:,t);
    if t > 1
        htm1 = draws.z(:,t-1);
        xx = [constvec htm1 para_est];
    else
        xx = [constvec para_est];
    end 
    yy = ht;
    Kxx = size(xx,2);
    Binv = eye(Kxx)/10000 + xx'*xx;
    Binvb = xx'*yy;
    coef = Binv\Binvb;
    yyfit = xx*coef;
    resid = yy-yyfit;
    if t > 1
        ISpara(t,:) = [coef' var(resid)];
    else
        ISpara(t,:) = [coef(1) 0 coef(2:Kxx)' var(resid)];
    end
    R2vec(t) = var(yyfit)/var(yy);
    resid_mat(:,t) = resid;
end %calibrate AR(1) IS for latent variable

logh_gd = zeros(ndraws,1);
logp_gd = logh_gd;
logy_gd = logh_gd;
logqh_gd = logh_gd;
logqp_gd = logh_gd;
logw_gd = logh_gd;
hj_mean = zeros(n,1);
for drawi = 1:ndraws
    paraj = para_est(drawi,:)';
    bj = paraj(1:K);
    uj = paraj(K+1);
    rhoj = (exp(paraj(K+2))-1)/(exp(paraj(K+2))+1);
    sj = sqrt(exp(paraj(K+3)));
    s2j = sj^2;
    rhoj2 = rhoj^2; %theta from posterior   
    
    hj = draws.z(drawi,:)';
    for t = 1:n
        if t == 1
            xxt = [1 0 paraj'];
        else
            xxt = [1 hj(t-1) paraj']; 
        end
        hj_mean(t) = xxt * ISpara(t,1:(KK+2))';
    end %h from posterior
    
    logpj_b = -0.5*K*log(2*pi) -0.5*sum(log(b0_var)) -0.5*sum(((bj-b0_mean).^2)./b0_var);
    logpj_SV = -0.5*log(2*pi/invVmuh) - 0.5*((uj-muh0)^2)*invVmuh ...
        +phiha*log(1+rhoj)+phihb*log(1-rhoj)-(phiha+phihb)*log(2)-betaln(phiha,phihb) ...
        -0.5*log(2*pi*sigh2_s) - 0.5*s2j/sigh2_s + 0.5*log(s2j); 
    logpj = logpj_b + logpj_SV; %prior theta
     
    tmp = (hj(2:n)-(1-rhoj)*uj-rhoj*hj(1:n-1)).^2;
    loghj = -0.5*n*log(2*pi*s2j) + 0.5*log(1-rhoj2) - 0.5*(1-rhoj2)*((hj(1)-uj)^2)/s2j ...
        -0.5*sum(tmp)/s2j; %prior h    
    
    eps = y - x*bj;
    logyj = -0.5*n*log(2*pi) - 0.5*sum(hj+eps.*eps.*exp(-hj)); %likelihood
    
    resid2 = (hj - hj_mean).^2;
    ISd = ISpara(:,KK+3);
    logqhj = -0.5*n*log(2*pi) - 0.5*sum(log(ISd)) -0.5*sum(resid2./ISd); %IS h
    
    tmp = paraj - para_mean;
    logqpj = -0.5*KK*log(2*pi) - 0.5*logdet_para_cov - 0.5*tmp'*para_covinv*tmp; %IS theta
    
    logh_gd(drawi) = loghj;
    logp_gd(drawi) = logpj;
    logy_gd(drawi) = logyj;
    logqh_gd(drawi) = logqhj;
    logqp_gd(drawi) = logqpj; 
    logw_gd(drawi) = logpj+loghj+logyj-logqpj-logqhj;
end
wtmp = -logw_gd;
wlevel = max(wtmp); 
ew = exp(wtmp - wlevel);
tmp_mean = mean(ew);
[~, hac_std, ~] = HAC_regression(ew,constvec); 
tmp_std = sqrt(ndraws)*hac_std; 
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


    



