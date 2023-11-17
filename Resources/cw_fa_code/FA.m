% Factor Analysis compression function

% Parameters (where 'D' is number of features)
% d: number of factors (width of rectangular matrix Lam)
% bufsize: number of updates in buffer Bee
% Psi [D x 1]: diagonal matrix
% Lam [D x d]: rectangular matrix
% Bee [D x bufsize]: buffer matrix
% beta: update multiplier (see below)
% s: update vector (see below)
% If useInverse == 0
%  In: Psi + Lam Lam' - Bee Bee' - beta s s'
% Out: Psi + Lam Lam' - Bee Bee'
% Else if useInverse == 1
%  In: Psi + Lam Lam' + Bee Bee' + beta s s'
% Out: Psi + Lam Lam' + Bee Bee'

function [Psi, Lam, Bee] = FA(Psi, Lam, Bee, d, bufsize, beta, s, useInverse)
% Parameters for Factor Analysis
EM_ITER = 50; % <-- might have to tweak this
STOP_THRESH = 1e-6; % <-- might have to tweak this
B_THRESH = bufsize;
LamSize = size(Lam, 2);
BeeSize = size(Bee, 2);
if nargin < 8
  USEINVERSE = 0;
else
  USEINVERSE = useInverse;
end

if USEINVERSE == 0
  if LamSize < d
    % Exact covariance update
    LamSize = LamSize + 1;
    BeeSize = BeeSize + 1;
    Lam(:, LamSize) = sqrt(1 - beta) * s;
    Bee(:, BeeSize) = s;
  elseif BeeSize < B_THRESH
    BeeSize = BeeSize + 1;
    Bee(:, BeeSize) = sqrt(beta) * s;
  end
else
  if LamSize < d
    % Exact covariance update
    LamSize = LamSize + 1;
    Lam(:, LamSize) = sqrt(beta) * s;
  elseif BeeSize < B_THRESH
    BeeSize = BeeSize + 1;
    Bee(:, BeeSize) = sqrt(beta) * s;
  end
end

is_approx = 0;
if (BeeSize >= B_THRESH) && (LamSize >= d)
  is_approx = 1;

  % Approximate covariance update
  Psi0 = Psi;
  Lam0 = Lam;

  if USEINVERSE == 0
    Sigdiag = Psi0 + sum(Lam0.^2, 2) - sum(Bee.^2, 2);
  else
    Sigdiag = Psi0 + sum(Lam0.^2, 2) + sum(Bee.^2, 2);
  end
  I = eye(d);
  j = 0;
  diff = STOP_THRESH + 1;
  while (j < EM_ITER) && (diff > STOP_THRESH)
    % E-step
    invPsiLam = Lam;
    invPsi = 1 ./ Psi;
    for k = 1:d
      invPsiLam(:, k) = invPsiLam(:, k) .* invPsi;
    end
    Phi = inv(I + Lam' * invPsiLam); % d x d
    UpsT = invPsiLam * Phi; % D x d

    % Old way (inefficient)...
    %S = repmat(Psi0, 1, d) .* UpsT + Lam0 * (Lam0' * UpsT) - Bee * (Bee' * UpsT); % D x d
    % New way...
    S = UpsT;
    for k = 1:d
      S(:, k) = S(:, k) .* Psi0;
    end
    S = S + Lam0 * (Lam0' * UpsT);
    if USEINVERSE == 0
      S = S - Bee * (Bee' * UpsT);
    else
      S = S + Bee * (Bee' * UpsT);
    end

    % M-step
    prevLam = Lam;
    prevPsi = Psi;
    Lam = S * inv(Phi + UpsT' * S);
    Psi = Sigdiag - sum(Lam .* S, 2);

    j = j + 1;

    % Calculate difference in parameters averaged by number of parameters
    diff = (sum(abs(Psi - prevPsi)) + sum(sum(abs(Lam - prevLam)))) ./ (numel(Psi) + numel(Lam));
  end

  Bee = zeros(size(Lam, 1), 0); % reset Bee
end
