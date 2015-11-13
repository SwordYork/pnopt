addpath(genpath('../'));

%[ty, tX] = libsvmread('dataset/w8a.txt');
%tX = [ones(size(ty)) tX];

%[testy, testX] = libsvmread('dataset/w8a.t');
%testX = [ones(size(testy)) testX];

tX_load = load('dataset/tX.mat');
tX = tX_load.tX;
ty_load = load('dataset/ty.mat');
ty = ty_load.ty;


testX_load = load('dataset/testX.mat');
testX = testX_load.testX;
testy_load = load('dataset/testy.mat');
testy = testy_load.testy;


[n, p] = size(tX);

% Fused Matrix
W = eye(p-1,p);
for i=1:p-1
    W(i,i+1) = -1;
end
W = sparse(W);

% Initial primal
w0 = zeros(p,1);

% Initial dual
z0 = {zeros(size(w0)); zeros(size(w0,1)-1,1)};


[n, p] = size(tX);

% Regularization Parameter
lambda = 2 / n ;
lambda_w = 2 / n ;

% Dual of Regularization
dualproxF = { proj_linf(lambda), proj_linf(lambda_w) };

% Loss function
fun_obj = @(w) FastLogisticLossSimple(w,tX,ty);

% l1 Regularization
l1_pen  = prox_l1(lambda);

% Initial Quasi-LBFGS Strongly Convex Parameter
rho = 0;

t_start = tic;

% Some stop critical
pnopt_options = pnopt_optimset( 'ftol', 1e-6, 'max_iter', 50, 'Lbfgs_mem', 50);

fprintf('train start: \n')
tic;

%  [ w, f, output ] = pnopt( smoothF, nonsmoothF_scalar_matrix, dual_proxF, primal_init, dual_init, nonsmoothF, strongly_convex_parameter, stop_critical)
[ w, f, output ] = pnopt( fun_obj, {1, 0; W, 0}, dualproxF, w0, z0, {l1_pen, @(x) norm(W*x, 1)*lambda_w }, rho, pnopt_options);

toc


py = tX*w;
fprintf('train error rate: %f \n', sum(abs(((py > 0) - (py < 0)) ~=  ty)) / length(py))

testpy = testX*w;
fprintf('test error rate: %f\n', sum(abs(((testpy > 0) - (testpy < 0)) ~=  testy)) / length(testy))

plot(w, 'bo')
title('Fused sparse structural constraints');

