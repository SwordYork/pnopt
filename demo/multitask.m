addpath(genpath('../'));
Xo = load('dataset/usps.mat');
Xo = Xo.all;
Xo = Xo ./ repmat(max(abs(Xo)), 2000, 1);
Xo = double(Xo);
yo = reshape(repmat([1:10],200,1), 2000, 1);


train = [];
test = [];
s = 40;
for i=0:9
    train = [train i*200+2*s+1:i*200+3*s+1];
    test = [test i*200+1:i*200+2*s  i*200+3*s+1:(i+1)*200];
end
X = Xo(train,:);
y = yo(train);
tX = Xo(test,:);
ty = yo(test);

%X = X(1:400,:);
%y = y(1:400);
% y = (y + 1) / 2 + 1;
[n, p] = size(X);

fun_obj = @(w) FastMulLogisticLossSimple(w,X,y);
lambda_s = 0.000018;
lambda_g = 0.000026;
%lambda_g = 0;
%lambda_s = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sparse

r = max(y);
w0 = zeros(r*p,1);
z0 = {zeros(size(w0))};
dualproxF = { proj_linf(lambda_s) };

l1_pen  = prox_l1(lambda_s);
pF = {l1_pen};

aff = {1, 0};

dualproxF{2} = proj_l2group(repmat(lambda_g,p,1),r:r:p*r);
z0{2} = zeros(size(w0));
pF{2} =  @(x) norm(x, 2) * lambda_g;
aff{2,1} = 1;
aff{2,2} = 0;

t_start = tic;

pnopt_options = pnopt_optimset( 'ftol', 1e-5, 'Lbfgs_mem', 50, 'max_iter',  50, 'method','qlbfgs', 'subprob_solver', 'lbfgs-tfocs');

[ w, f, output ] = pnopt( fun_obj, aff, dualproxF, w0, z0, pF, 0, pnopt_options);

toc(t_start)



tW = reshape(w, r, numel(w)/r);
tWo = tW';
[ssss, sssss] = max(X * tWo,[], 2);
sum(sssss ~= y) / size(y,1)

[ssss, sssss] = max(tX * tWo,[], 2);
sum(sssss ~= ty) / size(ty,1)

hFig = figure(1);

set(hFig, 'Position', [100 600 500 500*0.382])


imagesc(tW(:, 40:100))
colormap hot
colorbar
caxis([-0.3 0.2]);
ylabel('Tasks');
xlabel('Features');
