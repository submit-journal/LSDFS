function [W, opt_index, S, obj_value, obj2]= LocalFS(X, Y, h, m, anchor_num, lambda, r)
% Fast Adaptive Local Linear Discriminant Analysis
% min_{P1=1,||W||_{2,0}=h,Z} \sum\limits_{i = 1}^c 
% {\sum\limits_{j = 1}^{{n_i}} {\sum\limits_{t = 1}^{{m_i}}
% {{{\left( {P_{jt}^i} \right)}^r}\left\| {{W^T}x_j^i - Z_t^i} \right\|_2^2} } } - lambda*Tr(W^TStW)  


% input---
% X_train: data,     n*d
% Y_train: label,    n*1
% anchor_num: anchor number for each class
% r: option.r  fix 2 
% h: number of selected features
% dim: reduced dimension
% lambda: paramater to balance 

% output---
% W : transformation matrix
% opt_index: indexes of selected features
% obj_value : objective of main function
% obj2 : objective of sub function


maxIter = 3;

[~,dim] = size(X); 
c = unique(Y);
n = length(Y);


% St
H = eye(n) - 1/n * ones(n);
St = X' * H * X;

% initialize W
W = orth(rand(dim,m)); 
c_num = length(c);
anchor_all = c_num * anchor_num;

%% initialize anchor points and S/ update by fuzzy c_means
S = zeros(n,anchor_all);
ind2=[0];
for i = 1:c_num
    ind1 = find(Y ==c(i));
    ind2 = [ind2 length(ind1)];
    Xc{i} = X(ind1,:);                         
    options(1)= r;
    [center,U,~] = fcm1(Xc{i},anchor_num,options);       
    C{i} = center; 
    S(ind1,(i-1)*anchor_num+1:i*anchor_num) = U';   % similar matrix 
end


%% Iterative calculate the projection W S Z  
count = 0;
interval = 1;
Obj = 1e+8;
obj_value =[];

while (abs(interval)>=1e-5 && count<20)
    
    %% update anchor point Z
    S_in = S.^r;
    center = S_in'*X./(sum(S_in',2)*ones(1,size(X,2))); 
    for j = 1:size(center,1)/anchor_num
        C{j} = center((j-1)*anchor_num +1:j*anchor_num,:);
    end
    
    %% update transformation matrix W
    X_new = [X;center];
    S_new = zeros(n+anchor_all,n+anchor_all);
    S_new(1:n,n+1:end) = S_in;
    S_new(n+1:end,1:n) = S_in';
    D_s = diag(sum(S_new));
    L_s = D_s - S_new;
    Sw = X_new'*L_s*X_new;
    AA = Sw - lambda * St;
    eta = abs(eigs(AA,1));
    A = eta*eye(dim)-AA;
    A = max(A,A');
    for iter = 1:maxIter
        pinvAW = pinv(W'*A*W);
        P = A*W*pinvAW*W'*A;        
        PP{iter} = P;
        [~,ind] = sort(diag(P),'descend');
        opt_index = sort(ind(1:h));
        Aopt = A(opt_index, opt_index);
        [V, ~] = eigs(Aopt, m);
        W = zeros(dim,m);
        W(opt_index, :) = V;
        VV{iter} = V;
        obj2(iter) = trace(W'* P *W);
    end
    
    %% objective function     
    obj_sw = trace(W'*AA*W);
    interval = Obj-obj_sw;
    Obj = obj_sw;
    obj_value = [obj_value;obj_sw];
    count = count + 1;       

    %% update similarity matrix S
    S = zeros(n,anchor_all);   y = X*W;  
    for i = 1:c_num
        ind = find( Y == c(i));
        yc = y(ind,:);
        cc = C{i} * W;
        vc = slmetric_pw(yc',cc','sqdist') + 1e-10;
        mole = vc.^(1/(1-r));
        deno = repmat(sum(mole,2),1,size(mole,2));
        S(ind,1+(i-1)*anchor_num:i*anchor_num) = mole./deno;
    end

end
S = sparse(S);
end