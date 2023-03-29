%% Weighbted local mean representation-based nearest neighbors classifier (WLMRKNN)
function [PredictTest_table, RR_L0WLMRKNN] = L0WLMRKNN_func(X_train, X_test, K, X_train_lable, X_test_lable)

[~, Ntest] = size(X_test);
Classes = unique(X_test_lable);
PredictTest_table = zeros(length(Classes),Ntest);
true_counter = 0;
false_counter = 0;
lambda = 0.5;

for c1 = 1:length(Classes)
    train_set_{c1} = X_train(:,find(X_train_lable==Classes(c1)));
    test_set_{c1} = X_test(:,find(X_test_lable==Classes(c1)));
    num_train_class(c1) = length(find(X_train_lable==Classes(c1)));
end

for c2 =1:Ntest
    test_d = X_test(:,c2);    
    for c3 = 1:length(Classes)
        X_mean_vecs = [];
%         temp1 = repmat(test_d,1,num_train_class(c3));
%         temp_train = train_set_{c3};
%         Euc_dist = (temp1 - temp_train).^2;
%         [~, ind_1] = sort(sum(Euc_dist));
     
%%%%%%%%  
        temp_X_train = train_set_{c3};
        rem_x = test_d;
        base_X = [];
        base_X_ind = [];
        resdu = [];
        cc1 =1;
        threshold = 0.035*norm(test_d);
        %     while((norm(rem_x)/D)>threshold)
        while(cc1<(min(K,num_train_class(c3))+1))
            % step 1
            Projection_Coeff = (rem_x' * temp_X_train) ./ (sqrt(sum(temp_X_train.^2)));
            [proj_coeff , proj_ind] = max((Projection_Coeff));
            [proj_coeffs , proj_inds] = sort(Projection_Coeff, 'descend');
            % step 2
            base_X = [base_X , temp_X_train(:,proj_ind)];
            temp_X_train(:,proj_ind) = [];
            % step 3
            reconstruct_coeff = (base_X' * base_X)^-1 * base_X' * test_d;
            %step 4
            rem_x = test_d - base_X * reconstruct_coeff;
%             break_condition_val = sqrt(sum(rem_x.^2));
            cc1=cc1+1;
        end
        
        temp_train = base_X; 
%%%%%%%%%

        X_mean_vecs(:,1) = temp_train(:,1);
        for c4=2:min(K,num_train_class(c3))
            X_mean_vecs(:,c4) = mean(temp_train(:,1:c4)')';
        end
        W = diag(sqrt(sum((repmat(test_d, 1,size(X_mean_vecs,2) )-X_mean_vecs).^2)));
        W = W./max(W(:));
%         Beta = ((X_mean_vecs' * X_mean_vecs + xigma*eye(K))^-1) * X_mean_vecs' * test_d;
        Beta = ((X_mean_vecs' * X_mean_vecs + lambda*W'*W)^-1) * X_mean_vecs' * test_d;
        r(c3) = sum((test_d - X_mean_vecs*Beta).^2);
    end

    [~, predict_lable]= min(r);
    PredictTest_table(predict_lable,c2) = 1;
    if  predict_lable == X_test_lable(c2)
        true_counter = true_counter +1;
    else
        false_counter = false_counter +1;
    end
end
RR_L0WLMRKNN = (true_counter/(true_counter+false_counter))*100