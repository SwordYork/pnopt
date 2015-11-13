function [New_Ninv_S, New_Ninv_Y, New_D, New_SNS, New_SNY] = quasi_bfgs_update(S, Y, YS, SS, rho, hDiag)
% Update cached item one by one
% using the routing of inv_lbfgs
% 

New_Ninv_Y = zeros(size(S,1),0);
New_Ninv_S = zeros(size(S,1),0);
New_SNY = zeros(0);
New_SNS = zeros(0);
New_D = zeros(0);
for i = 1:size(S,2)
    s = S(:,i);
    y = Y(:,i);
    New_Ninv_Y_tmp = quasi_bfgs(S, New_Ninv_S, New_Ninv_Y, New_D, YS, SS, New_SNY, New_SNS, y, rho, hDiag);
    New_Ninv_Y(:, i) = New_Ninv_Y_tmp;
        
    % no need to update if rho=0
    if rho ~= 0
        New_Ninv_S_tmp = quasi_bfgs(S, New_Ninv_S, New_Ninv_Y, New_D, YS, SS, New_SNY, New_SNS, s, rho, hDiag);
        New_Ninv_S(:, i) = New_Ninv_S_tmp;
        New_SNS(i) = s' * New_Ninv_S_tmp;
        New_SNY(i) = s' * New_Ninv_Y_tmp;
    end
   
    New_D(i) =  y'*s + y'*New_Ninv_Y_tmp;
end

end
