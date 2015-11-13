function [ x, f, output ] = pnopt(smoothF, affineF, dualproxF, x, z0, nonsmoothF, rho, options )
% pnopt : Proximal Newton-type methods
% 
% [ x, f, output ] = pnopt( smoothF, nonsmoothF, x ) starts at x and seeks a 
%   minimizer of the objective function in composite form. smoothF is a handle
%   to a function that returns the smooth function value and gradient. nonsmoothF
%   is a handle to a function that returns the nonsmooth function value and 
%   proximal mapping. 
%  
% [ x, f, output ] = pnopt( smoothF, nonsmoothF, x, options ) replaces the default
%   optimization parameters with those in options, a structure created using the
%   pnopt_optimset function.
% 
%   $Revision: 0.8.0 $  $Date: 2012/12/01 $

% ============ Process options ============

tfocs_opts = struct(...
'alg', 'N83', ...
'maxIts'     , 500   ,...
'printEvery' , 0     ,...
'tol',         1e-7, ...
'restart'    , -Inf   ...
);

default_options = pnopt_optimset(...
'debug'          , 0          ,... % debug mode 
'desc_param'     , 0.000001     ,... % sufficient descent parameter
'display'        , 1         ,... % display frequency (<= 0 for no display) 
'Lbfgs_mem'      , 50         ,... % L-BFGS memory
'max_fun_evals'  , 500       ,... % max number of function evaluations
'max_iter'       , 100        ,... % max number of iterations
'tfocs_opts'     , tfocs_opts ,... % subproblem solver options
'ftol'           , 1e-9       ,... % stopping tolerance on relative change in the objective function 
'xtol'           , 1e-6        ... % stopping tolerance on solution
);


if nargin > 3
    options = pnopt_optimset( default_options, options );
else
    options = default_options;
end

% ============ Call solver ============

[ x, f, output ] = pnopt_sepqn(smoothF, affineF, dualproxF, x, z0, nonsmoothF, rho, options);


function S3 = merge_struct( S1 ,S2 )
% merge_struct : merge two structures
%   self-explanatory ^
% 
S3 = S1;
S3_names = fieldnames( S2 );
for k = 1:length( S3_names )
    if isfield( S3, S3_names{k} )
        if isstruct( S3.(S3_names{k}) )
            S3.(S3_names{k}) = merge_struct( S3.(S3_names{k}),...
            S2.(S3_names{k}) );
      else
          S3.(S3_names{k}) = S2.(S3_names{k});
      end
  else
      S3.(S3_names{k}) = S2.(S3_names{k});
  end
end

  
