function op = sepqn_smooth_quad(sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, grad_g_x, hDiag, H_x )

%SMOOTH_QUAD   Quadratic function generation.
%   FUNC = SMOOTH_QUAD( P, q, r ) returns a function handle that implements
%
%        FUNC(X) = 0.5 * TFOCS_DOT( P * x, x ) + TFOCS_DOT( q, x ) + r.
%
%   All arguments are optional; the default values are P=I, q=0, r=0. In
%   particular, calling FUNC = SMOOTH_QUAD with no arguments yields
%   
%        FUNC(X) = 0.5 * TFOCS_NORMSQ( X ) = 0.5 * TFOCS_DOT( X, X ).
%
%   If supplied, P must be a scalar, square matrix, or symmetric linear
%   operator. Furthermore, it must be positive semidefinite (convex) or
%   negative semidefinite (concave). TFOCS does not verify operator 
%   symmetry or definiteness; that is your responsibility.
%   If P is a vector, then it assumed this is the diagonal part of P.
%   Note: when P is diagonal, this function can compute a proximity
%   operator.  If P is zero, then smooth_linear should be called instead.
%   When P is an explicit matrix, this can also act as a proimity operator
%   but it may be slow, since it must invert (I+tP). True use_eig mode (see below)
%
%   If P is a scaling matrix, like the identity or a multiple of the identity
%   (say, P*x = 5*x), then specifying the scaling factor is sufficient (in
%   this example, 5). If P is empty, then P=1 is assumed.
%
%   FUNC = SMOOTH_QUAD( P, q, r, use_eig )
%      will perform a one-time (but expensive) eigenvalue decomposition
%       of P if use_eig=true, which will speed up future iterations. In general,
%       this may be significantly faster, especially if you run 
%       an algorithm for many iterations.
%      This mode is only useful when P is a full matrix and when
%      smooth_quad is used as a proximity operator; does not affect
%      it when used as a smooth operator. New in v 1.3.
%
%   See also smooth_linear.m



op = @(varargin) sepqn_smooth_quad_matrix( sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, grad_g_x, hDiag, H_x, varargin{:} );

end


function [ v, g ] = sepqn_smooth_quad_matrix( sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, grad_g_x, hDiag, H_x, x, t)
 g = quasi_bfgs(sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, x-grad_g_x, t, hDiag) ;
% g = quasi_bfgs(sPrev, Ninv_s, Ninv_y, D, YS, SS, SNY, SNS, x-grad_g_x, 0.001, hDiag) ;
 v = 0.5 * g' * H_x(g) + grad_g_x' * g;
end



% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.
