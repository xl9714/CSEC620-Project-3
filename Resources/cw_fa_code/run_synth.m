function run_synth()
% Synthetic experiments
runs = 10 % you can change this to 100 runs to reproduce experiment
N=1000;
D=100;
d=10;
X = {};
Y = {};
for i = 1:runs
  randn('state', 1000 + i);
  rand('state', 1000 + i);
  w_true = randn(d,1);
  V = rand(N,d) > 0.5;
  Y{i} = sign(V*w_true);

  X{i} = repmat(V,1,D);
  flip = rand(size(X{i})) > 0.95;
  X{i}(flip) = 1 - X{i}(flip);
  X{i} = double(X{i});

  X{i} = X{i}';
  Y{i} = Y{i}';
end

% display some output
disp(sprintf('Gathered %d features for %d instances',size(X{1},1),size(X{1},2)));

% learning modes
perceptron.update = 'perceptron';
pa.update = 'pa';
diag.sparsity = 'diag_l2';
buf(1).FAm = 2;
buf(2).FAm = 4;
buf(3).FAm = 8;
buf(4).FAm = 16;
buf(5).FAm = 32;
buf(6).FAm = 64;
[buf(:).sparsity] = deal('buffer');
fa(1).FAm = 2;
fa(2).FAm = 4;
fa(3).FAm = 8;
fa(4).FAm = 16;
fa(5).FAm = 32;
fa(6).FAm = 64;
[fa(:).sparsity] = deal('FAinv');
colors = 'mrygbc';
full.sparsity = 'full';

% go
figure
graph(X,Y,perceptron,'bx--', 'perceptron');
graph(X,Y,pa,'rx--', 'pa');
graph(X,Y,diag,'k', 'cw-diag');
for i = 1:length(fa)
  graph(X,Y,fa(i),colors(i), sprintf('cw-fa%d', fa(i).FAm));
end
graph(X,Y,full,'k--', 'cw-full');

% plotting subroutine
function graph(X,Y,params,format,name)
  params
  runs = numel(X)
  errs = [];
  mems = [];
  times = [];
  for i = 1:runs
    start = cputime;
    [err, mu, sigma, mem] = cw(X{i},Y{i},params);
    time = cputime - start;
    if i == 1
      errs = err;
      mems = mem;
      times = time;
    else
      errs(:, i) = err;
      mems(i) = mem;
      times(i) = time;
    end
  end
  total_err = mean(errs, 2);
  plot(total_err,format);

  save(sprintf('synth_results_%s.mat', name), 'errs', 'mems', 'times', 'format');
  hold on; drawnow
