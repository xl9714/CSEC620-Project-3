function [err,mu,sigma,mem] = cw(X,Y,params)
% X is k features by N instances
% Y is 1 label in {-1,1} by N instances
% params is struct containing options

% err: cumulative mistakes after each example
% mu: weight vector 'mu' after learning
% sigma: struct containing covariance/precision (inv. covariance) matrix after learning
% mem: memory consumption
  
  % params
  [k,N] = size(X);
  a = getparam(params,'a',1);
  eta = getparam(params,'eta',0.95);
  update = getparam(params,'update','stdev');
  sparsity = getparam(params,'sparsity','diag_kl');
  FAm = getparam(params, 'FAm', 16);
  bufsize = getparam(params, 'bufsize', FAm);
  average = getparam(params, 'average', 0);
  
  % initialize
  mem = 0;
  mu = getparam(params, 'mu', zeros(k,1));
  if(isfield(params, 'sigma'))
    sigma = getfield(params, 'sigma');
  else
    switch sparsity
     case {'diag_kl','diag_l2'}
      sigma = ones(k,1);
     case 'full'
      sigma = a*eye(k);
     case {'FA', 'FAinv'}
      sigma.FApsi = a * ones(k, 1);
      sigma.FAlam = zeros(k, 0);
      sigma.FAbee = zeros(k, 0);
     case 'buffer'
      sigma.diag = a * ones(k, 1);
      sigma.buff = zeros(k, 0);
    end
  end
  phi = sqrt(2) .* erfinv(2 .* eta - 1);
  psi = 1 + phi^2 / 2;
  xi = 1 + phi^2;
  err = zeros(N,1);
  if average
    v = zeros(size(mu));
  end
    
  % flag for indicate whether covariance was updated
  did_update = 1; % initialized to 1 to force first update

  % iterate
  for i = 1:N

    % progress
    if log2(i) == floor(log2(i))
      disp(sprintf('Processing instance %d...',i));
    end

    % compute stats
    x = X(:,i);
    y = Y(i);
    
    switch sparsity
     case {'diag_kl','diag_l2'}
      sigma_x = my_times(sigma,x);
     case 'full'
      sigma_x = sigma*x;
     case 'FA'
      sigma_x = my_times(sigma.FApsi,x) + sigma.FAlam*(sigma.FAlam'*x) - ...
                sigma.FAbee*(sigma.FAbee'*x);
     case 'FAinv'
      if did_update == 1
        sigma.iPsi = 1 ./ sigma.FApsi;
        sigma.Gam = [sigma.FAlam, sigma.FAbee]; % [Lam, Bee], needed for inverse
        sigma.iPG = bsxfun(@times,sigma.iPsi,sigma.Gam); % iPsi * Gam
        sigma.iIGPG = inv(eye(size(sigma.Gam, 2)) + sigma.Gam' * sigma.iPG);
      end
      sigma_x = my_times(sigma.iPsi, x) - ...  % Woodbury inverse (Psi + Lam Lam' + Bee Bee') * x
                sigma.iPG * (sigma.iIGPG * (sigma.Gam' * my_times(sigma.iPsi, x)));
     case 'buffer'
      sigma_x = my_times(sigma.diag,x) - sigma.buff*(sigma.buff'*x);
    end
    switch update
     case 'weather'
      M = y .* mu;
     otherwise
      M = y*(x'*mu);
    end
    V = my_dot(x,sigma_x);
    
    % make a prediction
    if i > 1
      last_err = err(i-1);
    else
      last_err = 0;
    end
      
    if average
      mistake = (y*(x'*v + i*x'*mu) <= 0);
    else
      mistake = (M <= 0);
    end
    err(i) = last_err + mistake;  
    
    % update
    switch update

     case 'weather'
      mu = y;
      
     case 'perceptron'
      if M <= 0
        mu = mu + y*x;
        if average
          v = v - i*y*x;
        end
      end
      
     case 'pa'
      if M < 1
        if find(x)
          alpha = (1 - M) / (norm(x)^2);
          mu = mu + alpha*y*x;
          if average
            v = v - i*alpha*y*x;
          end
        end
      end
      
     case 'stdev'
      did_update = 0;
      if M < phi*sqrt(V)
        did_update = 1;
        alpha = (-M*psi + sqrt(M^2 * phi^4/4 + V*phi^2*xi)) / (V*xi);
        
        if isreal(alpha) & ~isnan(alpha) & ~isinf(alpha)          
          sqrtU = (-alpha*V*phi + sqrt(alpha^2*V^2*phi^2 + 4*V)) / 2;
          beta = (alpha*phi) / (sqrtU + V*alpha*phi);

          if issparse(sigma_x)
            nz = find(sigma_x);
            mu(nz) = mu(nz) + alpha*y*sigma_x(nz);
            if average
              v(nz) = v(nz) - i*alpha*y*sigma_x(nz);
            end
          else
            mu = mu + alpha*y*sigma_x;
            if average
              v = v - i*alpha*y*sigma_x;
            end
          end
          
          switch sparsity
           case 'diag_kl' % we recommend 'diag_l2' update instead of this one
            if issparse(x)
              nz = find(x);
              sigma(nz) = 1./(1./sigma(nz) + alpha*phi*(1/sqrtU)*(x(nz).^2));
            else
              sigma = 1./(1./sigma + alpha*phi*(1/sqrtU)*(x.^2));
            end
            
           case 'diag_l2'
            sigma = sigma - beta*(sigma_x.^2);
           case 'full'
            sigma = sigma - beta*sigma_x*sigma_x';
           case 'FA' % we recommend using 'FAinv' instead of this one
            [sigma.FApsi, sigma.FAlam, sigma.FAbee] = ...
                FA(sigma.FApsi, sigma.FAlam, sigma.FAbee, FAm, bufsize, beta, sigma_x);
           case 'buffer'
            if size(sigma.buff, 2) >= FAm
              % compress old data if we reach threshold
              sigma.diag = sigma.diag - sum(sigma.buff.^2, 2);
              sigma.buff = zeros(k, 0);
            end
            sigma.buff = [sigma.buff sqrt(beta)*sigma_x];
           case 'FAinv'
            [sigma.FApsi, sigma.FAlam, sigma.FAbee] = ...
                FA(sigma.FApsi, sigma.FAlam, sigma.FAbee, FAm, bufsize, ...
                   alpha * phi / sqrtU, x, 1);
          end
        else
          disp('Warning: bad alpha:');
          alpha
        end
      end
      
     otherwise
      error('I do not know the update method %s',update);
    end
     
    memuse = whos('sigma');
    if(memuse.bytes > mem)
      mem = memuse.bytes;
    end
  end
