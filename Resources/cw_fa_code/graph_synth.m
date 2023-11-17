function graph_synth(prefix)
% Synthetic experiments

figure

% figure parameters
colors = 'mrygbc';
asize = 20;
msize = 10;
location = 'NorthOutside';
Paper = [0 0 11 6];
orientation = 'horizontal';


graph(prefix, 'perceptron', 'Perceptron', '--b');
graph(prefix, 'pa', 'PA', '-g');
graph(prefix, 'cw-diag', 'CW-diag', '-.r');
graph(prefix, 'cw-fa2', 'CW-fact2', '--sk');
graph(prefix, 'cw-fa4', 'CW-fact4', '-xm');
graph(prefix, 'cw-fa8', 'CW-fact8', '-.^b');
graph(prefix, 'cw-fa16', 'CW-fact16', '--og');
graph(prefix, 'cw-full', 'CW-full', '-vr');
leg = {'Perceptron', 'PA', 'CW-diag', 'CW-fact2', 'CW-fact4', 'CW-fact8', 'CW-fact16', 'CW-full'};

legend(leg, 'Location', 'NorthWest', 'Orientation', 'vertical');
set(gca, 'FontSize', 18);
ylabel('Cumulative mistakes', 'FontSize', asize);
xlabel('Rounds', 'Fontsize', asize);
set(gcf, 'PaperPosition', Paper);

outpng = sprintf('%s_final.png', prefix);
outeps = sprintf('%s_final.eps', prefix);

print('-dpng', outpng);
print('-depsc', outeps);

% plot table
% Algorithm | Runtime (s) | Memory (b)
% ====================================
% CW-diag
% CW-fact2
% CW-fact4
% CW-fact8
% CW-fact16
% CW-full
fid = fopen(sprintf('%s_bench.tex', prefix), 'w+');
fprintf(fid, '\\begin{tabular}{|l||r|r|}\\hline\n');
fprintf(fid, '{\\bf Algorithm} & {\\bf Runtime (s)} & {\\bf Memory (KB)}\\\\\n');
fprintf(fid, '\\hline\\hline\n');
%fprintf(fid, '%s\n', tablerow(prefix, 'perceptron', 'Perceptron'));
%fprintf(fid, '%s\n', tablerow(prefix, 'pa', 'PA'));
fprintf(fid, '%s\n', tablerow(prefix, 'cw-diag', 'CW-diag'));
fprintf(fid, '%s\n', tablerow(prefix, 'cw-fa2', 'CW-fact2'));
fprintf(fid, '%s\n', tablerow(prefix, 'cw-fa4', 'CW-fact4'));
fprintf(fid, '%s\n', tablerow(prefix, 'cw-fa8', 'CW-fact8'));
fprintf(fid, '%s\n', tablerow(prefix, 'cw-fa16', 'CW-fact16'));
fprintf(fid, '%s\n', tablerow(prefix, 'cw-full', 'CW-full'));
fprintf(fid, '\\hline\\end{tabular}\n');
fclose(fid);

% plotting subroutine
function graph(prefix, name, leg, format)
  errs = [];
  mems = [];
  times = [];
  load(sprintf('%s_results_%s.mat', prefix, name), 'errs', 'mems', 'times');
  total_err = mean(errs, 2);
  std_err = std(errs, 1, 2);
  skip = 50;
  nrounds = numel(total_err);
  idx = [1, skip:skip:nrounds];
  plot(idx, total_err(idx), format, 'LineWidth', 2, 'MarkerSize', 10);
  hold on; drawnow

% table subroutine
function rowtext = tablerow(prefix, name, leg)
  errs = [];
  mems = [];
  times = [];
  load(sprintf('%s_results_%s.mat', prefix, name), 'errs', 'mems', 'times');
  total_time = mean(times);
  total_mem = mean(mems);

  rowtext = sprintf('%s & %.2f & %.2f\\\\', leg, total_time, ceil(total_mem) ./ 1024);
