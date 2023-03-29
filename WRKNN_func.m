%% Weighted representation-based K-nearest neighbors classifier (WRKNN)
function [PredictTest_table, RR_WRKNN,All_r] = WRKNN_func(X_train, X_test, K, X_train_lable, X_test_lable)

[~, Ntest] = size(X_test);
Classes = unique(X_test_lable);
PredictTest_table = zeros(length(Classes),Ntest);
true_counter = 0;
false_counter = 0;
Lambda = 0.5;

for c1 = 1:length(Classes)
    train_set_{c1} = X_train(:,find(X_train_lable==Classes(c1)));
    test_set_{c1} = X_test(:,find(X_test_lable==Classes(c1)));
    num_train_class(c1) = length(find(X_train_lable==Classes(c1)));
end

for c2 =1:Ntest
    test_d = X_test(:,c2);    
    for c3 = 1:length(Classes)
        temp1 = repmat(test_d,1,num_train_class(c3));
        temp_train = train_set_{c3};
        Euc_dist = (temp1 - temp_train).^2;
        [~, ind_1] = sort(sum(Euc_dist));
        
%         X_KN = temp_train(:,ind_1(1:K));
%         T = diag(sqrt(Euc_dist(ind_1(1:K))));           
        X_KN = temp_train(:,ind_1(1:min(K,num_train_class(c3))));
        T = diag(sqrt(sum(Euc_dist(:,ind_1(1:min(K,num_train_class(c3)))))));    
%         T = diag(sqrt((Euc_dist(ind_1(1:min(K,num_train_class(c3)))))));  
        Eta = ((X_KN' * X_KN + Lambda * T' * T)^-1) * X_KN' * test_d;
        r(c3) = sum((test_d - X_KN*Eta).^2);
        
    end
    [~, predict_lable]= min(r);
    All_r(c2,:) = r;
    PredictTest_table(predict_lable,c2) = 1;
    if  predict_lable == X_test_lable(c2)
        true_counter = true_counter +1;
    else
        false_counter = false_counter +1;
    end
end
RR_WRKNN = (true_counter/(true_counter+false_counter))*100