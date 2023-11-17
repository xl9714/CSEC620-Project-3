function x = my_dot(a,b)
% computes a'*b faster than built-in when a and b have the same sparsity pattern
  
  if issparse(a) && issparse(a)
    i = find(a);
    j = find(b);
    if length(i) == length(j) && all(i == j)
      x = full(a(i))'*full(b(i));
    else
      x = a'*b;
    end    
  else
    x = a'*b;
  end