function x = my_times(f,s)
% computes f.*s about 12x more quickly than the built-in version when f is full
% and s is sparse.
  
  if ~issparse(f) && issparse(s)
    a = find(s);
    [i,j] = find(s);
    x = sparse(i,j,full(s(a)).*f(a),size(s,1),size(s,2));
  else
    x = f.*s;
  end