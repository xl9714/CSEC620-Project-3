function v = get_param(p,k,d)
  if(isfield(p,k))
    v = getfield(p,k);
  else
    v = d;
  end