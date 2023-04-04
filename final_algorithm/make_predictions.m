function [predicted_dg] = make_predictions(test_data)
    % take in test_data and return predicted_dg

    % load test_data
    ecog1 = test_data{1};
    ecog2 = test_data{2};
    ecog3 = test_data{3};
    
    % remove channels
    ecog1(:, 55) = [];
    ecog2(:, [21,38]) = [];

    % get features from test_data
    allFeats1 = getAllWindowedFeats(ecog1, 1000);
    allFeats2 = getAllWindowedFeats(ecog2, 1000);
    allFeats3 = getAllWindowedFeats(ecog3, 1000);

    % create R matrices
    R1 = create_R_matrix(allFeats1, 4);
    R2 = create_R_matrix(allFeats2, 4);
    R3 = create_R_matrix(allFeats3, 4);

    % load model
    load('lasso_weights.mat');

    % make predictions
    preds1 = zeros(size(R1, 1), 5);
    preds2 = zeros(size(R2, 1), 5);
    preds3 = zeros(size(R3, 1), 5);
    
    for f = 1:5
        B = cell2mat(lasso_weights(1, f));   
        preds1(:, f) = R1*B;
        B = cell2mat(lasso_weights(2, f));   
        preds2(:, f) = R2*B;
        B = cell2mat(lasso_weights(3, f));   
        preds3(:, f) = R3*B;
    end

    % post-process predictions
    movpreds1 = zeros(size(R1, 1), 5);
    movpreds2 = zeros(size(R2, 1), 5);
    movpreds3 = zeros(size(R3, 1), 5);
    
    for f = 1:5
        movpreds1(:, f) = movmean(preds1(:, f), 42);
        movpreds2(:, f) = movmean(preds2(:, f), 42);
        movpreds3(:, f) = movmean(preds3(:, f), 42);
    end

    % spline predictions
    fullpreds1 = zeros(length(ecog1), 5);
    fullpreds2 = zeros(length(ecog2), 5);
    fullpreds3 = zeros(length(ecog3), 5);
    
    for f = 1:5
        x = linspace(1, length(movpreds1(:, f)),length(movpreds1(:, f)));
        xq = linspace(1,length(movpreds1(:, f)),length(ecog1));
        full_preds1(:,f) = spline(x,[0 movpreds1(:, f).' 0],xq);
        
        x = linspace(1, length(movpreds2(:,f)),length(movpreds2(:,f)));
        xq = linspace(1,length(movpreds2(:,f)),length(ecog2));
        full_preds2(:,f) = spline(x,[0 movpreds2(:,f).' 0],xq);
        
        x = linspace(1, length(movpreds3(:,f)),length(movpreds3(:,f)));
        xq = linspace(1,length(movpreds3(:,f)),length(ecog3));
        full_preds3(:,f) = spline(x,[0 movpreds3(:,f).' 0],xq);
    end
    
    % compile predictions
    predicted_dg = {full_preds1; full_preds2; fullpreds3};
end
