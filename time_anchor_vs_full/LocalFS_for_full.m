
function [time2, W, opt_index, S, obj_value, obj2]= LocalFS_for_full(X, Y, opt)
% Fast Adaptive Local Linear Discriminant Analysis
% min_{P1=1,||W||_{2,0}=h} \sum\limits_{i = 1}^c 
% {\sum\limits_{j = 1}^{{n_i}} {\sum\limits_{t = 1}^{{n_i}}
% {{{\left( {P_{jt}^i} \right)}^r}\left\| {{W^T}x_j^i - {W^T}x_t^i} \right\|_2^2} } } - lambda*Tr(W^TStW)  


% input---
% X_train: data,     n*d
% Y_train: label,    n*1
% r: option.r  fix 2 
% h: number of selected features
% dim: reduced dimension
% lambda: paramater to balance 

% output---
% W : transformation matrix
% opt_index: indexes of selected features
% obj_value : objective of main function
% obj2 : objective of sub function

h = opt(1);
m = opt(2);
% anchor_num = opt(3);
lambda = opt(3);
r = opt(4);
T = opt(5);

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


%% initialize anchor points and S/ update by fuzzy c_means
S = zeros(n,n);
ind2=[0];
for i = 1:c_num
    ind1 = find(Y ==c(i));
    ind2 = [ind2 length(ind1)];
    Xc{i} = X(ind1,:);                         
    vc = slmetric_pw(Xc{i}',Xc{i}','sqdist') + 1e-10;
    mole = vc.^(1/(1-r));
    deno = repmat(sum(mole,2),1,size(mole,2));
    S(ind1,ind1) = mole./deno;
    
end


%% Iterative calculate the projection W S Z  
count = 0;
interval = 1;
Obj = 1e+8;
obj_value =[];
tic
while ( count<T)
    %% update anchor point
    S_in = S.^r;

    
    %% update transformation matrix
    Sw = zeros(dim,dim);
    for i = 1:length(Xc)
        index_1 = sum(ind2(1:i));
        index_2 = sum(ind2(1:i));
        for j = 1:size(Xc{i},1)
            for k = 1:size(Xc{i},1)
                Sw = (Xc{i}(j,:)-Xc{i}(k,:))'*(Xc{i}(j,:)-Xc{i}(k,:))*S_in(j+index_1,k+index_2)+Sw;
            end
        end
    end

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

    %% update S
    S = zeros(n,n); 
    y = X*W;  
    for i = 1:c_num
        ind = find( Y == c(i));
        yc = y(ind,:);
        vc = slmetric_pw(yc',yc','sqdist') + 1e-10;
        mole = vc.^(1/(1-r));
        deno = repmat(sum(mole,2),1,size(mole,2));
        S(ind,ind) = mole./deno;
    end
end
time2 = toc
S = sparse(S);
end