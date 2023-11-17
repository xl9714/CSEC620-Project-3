README
------

Authors: Alex Kulesza (kulesza@cis.upenn.edu)
         Justin Ma (jtma@cs.ucsd.edu)


To run the example, open a Matlab environment and run the following:

  >> example

The output will be the following files:

  synth_final.png
  synth_final.eps
  synth_bench.tex


Contents
--------
cw.m --- code for running various online algorithms
FA.m --- code for factored approximation routine for "compressing" updates to
         covariance/precision (inv. covariance) matrix
getparam.m --- helper function for fetching parameters in cw.m
my_times.m, my_dot.m --- helper functions for computation in cw.m
run_synth.m --- runs synthetic experiments
graph_synth.m --- produce output graphs/tables
example.m --- a wrapper script for the whole thing
