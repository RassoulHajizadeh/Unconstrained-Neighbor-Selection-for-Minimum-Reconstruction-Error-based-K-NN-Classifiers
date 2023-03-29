%% Weighbted local mean representation-based nearest neighbors classifier (WLMRKNN)
function [PredictTest_table, RR_WLMRKNN] = WLMRKNN_func(X_train, X_test, K, X_train_lable, X_test_lable)

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
        temp1 = repmat(test_d,1,num_train_class(c3));
        temp_train = train_set_{c3};
        Euc_dist = (temp1 - temp_train).^2;
        [~, ind_1] = sort(sum(Euc_dist));
        
        X_mean_vecs(:,1) = temp_train(:,ind_1(1));   
%         for c4=2:K
        for c4=2:min(K,num_train_class(c3))
            X_mean_vecs(:,c4) = mean(temp_train(:,ind_1(1:c4))')';            
        end
        W = diag(sqrt(sum((repmat(test_d, 1,size(X_mean_vecs,2) )-X_mean_vecs).^2)));
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
RR_WLMRKNN = (true_counter/(true_counter+false_counter))*100