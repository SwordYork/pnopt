function [ x, f_x, output ] = pnopt_sepqn( smoothF, affineF, dualproxF, x, z0, nonsmoothF, rho, options )
% pnopt_pqn : Proximal quasi-Newton methods
%
% [ x, f, output ] = pnopt_pqn( smoothF, nonsmoothF, x, options ) starts at x and
%   seeks a minimizer of the objective function in composite form. smoothF is a
%   handle to a function that returns the smooth function value and gradient.
%   nonsmoothF is a handle to a function that returns the nonsmooth function
%   value and proximal mapping. options is a structure created using the
%   pnopt_optimset function.
%

REVISION = '$Revision: 0.9.2$';
DATE     = '$Date: Mar. 24, 2015$';
REVISION = REVISION(11:end-1);
DATE     = DATE(8:end-1);

% ============ Process options ============

debug          = options.debug;
desc_param     = options.desc_param;
display        = options.display;
max_fun_evals  = options.max_fun_evals;
max_iter       = options.max_iter;
ftol           = options.ftol;
xtol           = options.xtol;
lbfgs_mem      = options.Lbfgs_mem;

% ------------ Set subproblem solver options ------------

tfocs_opts = options.tfocs_opts;
if debug
    tfocs_opts.countOps = 1;
end

% ============ Initialize variables ============

pnopt_flags

iter         = 0;
loop         = 1;
forcing_term = 0.1;

trace.f_x    = zeros( max_iter + 1, 1 );
trace.elps_t    = zeros( max_iter + 1, 1 );
trace.fun_evals  = zeros( max_iter + 1, 1 );
trace.prox_evals = zeros( max_iter + 1, 1 );


if debug
    trace.forcing_term    = zeros( max_iter, 1 );
    trace.subprob_iters   = zeros( max_iter, 1 );
    trace.subprob_optim   = zeros( max_iter, 1 );
end

if display > 0
    fprintf( ' %s\n', repmat( '=', 1, 80 ) );
    fprintf( '                  PNOPT v.%s (%s)\n', REVISION, DATE );
    fprintf( ' %s\n', repmat( '=', 1, 80 ) );
    fprintf( ' %4s   %6s  %6s  %10s  %10s  %10s  %10s \n',...
        '','Fun.', 'Prox', 'Step len.', 'Obj. val.', '|df|/|f|.', 'hDiag' );
    fprintf( ' %s\n', repmat( '-', 1, 80 ) );
end

% ------------ Evaluate objective function at starting x ------------
[ g_x, grad_g_x ] = smoothF( x );
h_x = 0;
for i = 1:length(nonsmoothF)
    h_x = h_x + nonsmoothF{i}(x);
end
f_x         = g_x + h_x;


% ------------ Start collecting data for display and output ------------

fun_evals   = 1;
prox_evals  = 0;


trace.f_x(1)    = f_x;
trace.elps_t(1)    = 0;
trace.fun_evals(1)  = fun_evals;
trace.prox_evals(1) = prox_evals;


if display > 0
    fprintf( ' %4d | %6d  %6d  %f  %f  %12.4e  %f\n', ...
        iter, fun_evals, prox_evals, 0, f_x, 0, 0 );
end


x_old = 0;
backtrack_iters = 0;
grad_g_old = 0;
t_start = tic;

beta = 2;
step = 1;

% ============ Main Loop ============

while loop
    iter = iter + 1;
    
    % ------------ Update Hessian approximation ------------
    % LBFGS method
    if iter > 1
        s =  x - x_old;
        y = grad_g_x - grad_g_old;
        if y'*s > 1e-9
            % larger hDiag produce small ds
            %                     if df / df_old < 0.2
            %                         hDiag = hDiag / 2;
            %                     end
            if backtrack_iters > 1 && iter > 2
                hDiag = hDiag / step;
                if iter > 3
                    beta = 2 / (1+1/beta);
                end
            end
            if iter > 2
                hDiag = hDiag / beta;
            end
            hDiag = min(hDiag, ( y' * y ) / ( y' * s ));
            %hDiag =  ( y' * y ) / ( y' * s );
            %hDiag = 1024*4;
            ys = y'* s;
            ss  = s'* s;
            
            % exceed than lbfgs memory
            if size( sPrev, 2 ) >= lbfgs_mem
                sPrev = sPrev(:,2:lbfgs_mem);
                yPrev = yPrev(:,2:lbfgs_mem);
                YS = YS(:,2:lbfgs_mem);
                SS = SS(:,2:lbfgs_mem);
                SNY = SNY(:,2:lbfgs_mem);
                SNS = SNS(:,2:lbfgs_mem);
                Ninv_y = Ninv_y(:,2:lbfgs_mem);
                Ninv_s = Ninv_s(:,2:lbfgs_mem);
                D = D(:,2:lbfgs_mem);
            end
            
            % update these cached item according the hDiag
            [Ninv_s, Ninv_y, D, SNS, SNY] = quasi_bfgs_update(sPrev, yPrev, YS, SS, rho, hDiag);
            v_ns  = quasi_bfgs(sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, s, rho, hDiag);
            v_ny = quasi_bfgs(sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, y, rho, hDiag);
            
            sny = s' * v_ny;
            sns = s' * v_ns;
            sPrev = [ sPrev, s ];
            yPrev = [ yPrev, y ];
            YS = [ YS, ys];
            SS = [SS, ss];
            SNY = [SNY, sny];
            SNS = [SNS, sns];
            Ninv_y = [ Ninv_y, v_ny];
            Ninv_s = [ Ninv_s, v_ns];
            D = [D, ys + y'*v_ny];
        else
            fprintf('step is bad...\n');
        end
    else
        sPrev = zeros( length(x), 0 );
        yPrev = zeros( length(x), 0 );
        Ninv_y  = zeros( length(x), 0 );
        Ninv_s = zeros( length(x), 0 );
        YS = [];
        SS = [];
        SNY = [];
        SNS = [];
        D = [];
        hDiag = 1;
    end
    H_x = pnopt_bfgs_prod( sPrev, yPrev, hDiag );
    
    
    % ------------ Solve subproblem for a search direction ------------
    
    if iter > 1
        quadF = @(z) pnopt_quad( H_x, grad_g_x, f_x, z - x );
        
        % give the approximate solution
        %                 tfocs_opts.stopFcn = @(f, x) tfocs_stop( x, nonsmoothF,...
        %                     max( 0.0001 * optim_tol, forcing_term * optim ));
        %                 tfocs_opts.errFcn =  @(f,x) tfocs_err() ;
        subprob_iters = 0;
        
        tfocs_opts.continuation = 0;
        x_prox = x;
        tfocs_opts.tol = ftol;
        tfocs_opts.maxIts = 200;
        % tfocs_opts.Lexact = Inf;
        sf = sepqn_smooth_quad(sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, grad_g_x-H_x(x), hDiag, H_x);
        [ x_prox, tfocs_out ] = ...
            sepqn_tfocs_SCD( sf, affineF, dualproxF, 0, x_prox, z0, tfocs_opts);
        if iscell(tfocs_out.dual)
            z0 = cell( tfocs_out.dual );
        else
            z0 = tfocs_out.dual;
        end
        
        subprob_iters = subprob_iters + tfocs_out.niter;
        subprob_prox_evals = subprob_iters;
        subprob_optim = quadF(x_prox);
        for i = 1:length(nonsmoothF)
            subprob_optim = subprob_optim + nonsmoothF{i}(x_prox);
        end
        
        p = x_prox - x;
        
    else
        subprob_iters      = 0;
        subprob_prox_evals = 0;
        subprob_optim      = 0;
        p = - grad_g_x;
    end
    
    % ------------ Conduct line search ------------
    
    x_old      = x;
    f_old      = f_x;
    grad_g_old = grad_g_x;
    
    
    if iter > 1
        [ x, f_x, grad_g_x, step, backtrack_flag, backtrack_iters ] = ...
            pnopt_backtrack( x, p, 1, f_x, h_x, grad_g_x' * p, smoothF, nonsmoothF, ...
            desc_param, xtol, max_fun_evals - fun_evals );
    else
        [ x, f_x, grad_g_x, step, backtrack_flag, backtrack_iters ] = ...
            pnopt_curvtrack( x, p, max( min( 1, 1 / norm( grad_g_x ) ), xtol ), f_x, ...
            grad_g_x'*p, smoothF, nonsmoothF, desc_param, xtol, max_fun_evals - fun_evals );
    end
    if iter < 2
        hDiag =  100;
    end
    
    
    % ------------ Collect data for display and output ------------
    
    fun_evals   =  fun_evals + backtrack_iters ;
    %prox_evals  = prox_evals + subprob_prox_evals;
    prox_evals  =  prox_evals + subprob_prox_evals;
    
    
    trace.f_x(iter+1)        = f_x;
    trace.fun_evals(iter+1)  = fun_evals;
    trace.prox_evals(iter+1) = prox_evals;
    trace.elps_t(iter+1) = toc(t_start);
    
    if debug
        trace.forcing_term(iter)    = forcing_term;
        trace.backtrack_iters(iter) = backtrack_iters;
        trace.subprob_iters(iter)   = subprob_iters;
        trace.subprob_optim(iter)   = subprob_optim;
    end
    
    if display > 0 && mod( iter, display ) == 0
        fprintf( ' %4d | %6d  %6d  %f  %f  %12.4e  %f\n', ...
            iter, fun_evals, prox_evals, step, f_x, abs( f_old - f_x ) / max( 1, abs( f_old ) ), hDiag );
    end
    
    pnopt_stop
end

% ============ Clean up and exit ============

trace.f_x        = trace.f_x(1:iter+1);
trace.fun_evals  = trace.fun_evals(1:iter+1);
trace.prox_evals = trace.prox_evals(1:iter+1);
trace.elps_t     = trace.elps_t(1:iter+1);

if debug
    trace.forcing_term    = trace.forcing_term(1:iter);
    trace.backtrack_iters = trace.backtrack_iters(1:iter);
    trace.subprob_iters   = trace.subprob_iters(1:iter);
    trace.subprob_optim   = trace.subprob_optim(1:iter);
end

if display > 0 && mod( iter, display ) > 0
    fprintf( ' %4d | %6d  %6d  %f  %12.4e  %12.4e %f\n', ...
        iter, fun_evals, prox_evals, step, f_x, optim, hDiag );
    fprintf( ' %s\n', repmat( '-', 1, 64 ) );
end

output = struct( ...
    'flag'       , mflag       ,...
    'fun_evals'  , fun_evals  ,...
    'iters'      , iter       ,...
    'options'    , options    ,...
    'prox_evals' , prox_evals ,...
    'trace'      , trace       ...
    );

clear global subprob_Dg_y subprob_optim

end


function H_x = pnopt_bfgs_prod( S, Y, de )
% pnopt_bfgs_prod : L-BFGS Hessian approximation
%
l = size( S, 2 );
L = zeros( l );
for k = 1:l;
    L(k+1:l,k) = S(:,k+1:l)' * Y(:,k);
end
d1 = sum( S .* Y );
d2 = sqrt( d1 );

R    = chol( de * ( S' * S ) + L * ( diag( 1 ./ d1 ) * L' ), 'lower' );
R1   = [ diag( d2 ), zeros(l); - L*diag( 1 ./ d2 ), R ];
R2   = [- diag( d2 ), diag( 1 ./ d2 ) * L'; zeros( l ), R' ];
YdS  = [ Y, de * S ];
H_x  = @(x) de * x - YdS * ( R2 \ ( R1 \ ( YdS' * x ) ) );

end
