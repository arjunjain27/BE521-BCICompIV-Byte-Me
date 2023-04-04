%%
load("feat_mat_p1.mat")
load("Leaderboard_dat_p1.mat")
load("final_proj_part1_data.mat")

Y_hat_p1 = [];
for finger = 1:5
    dg = train_dg{1};
    dg = dg(26:end,:);
    dg = dg(1:end-25,:);
    dg_ds = downsample(dg,50);
    dg_ds = dg_ds(:,finger);

    mat = create_R_matrix(feat_mat_p1, 3);
    
    trainx = mat;
    trainy = dg_ds;

    testx = Leaderboard_dat_p1;
    testx(:, 301:306) = [];
    test_mat = create_R_matrix(testx, 3);

    [coeff, score, latent, tsquared, explained] = pca(trainx);
    trainx_pca = score(:,1:1000);
    testx_pca = test_mat*coeff;
    testx_pca = testx_pca(:,1:1000);

    [B,S] = lasso(trainx_pca,trainy,'Lambda',.02);
    y_pred = testx_pca*B;
    movmeanY = movmean(y_pred, 42);
    interpY2 = interp(movmeanY(:,1), 50);
    D = padarray(interpY2,25,0,'both');
    Y_hat_p1(:,finger) = D;
end

clear feat_mat_p1 Leaderboard_dat_p1, 


%%

load("feat_mat_p2.mat")
load("feat_mat_test_p2.mat")
load("final_proj_part1_data.mat")

Y_hat_p2 = [];
for finger = 1:5
    dg = train_dg{2};
    dg = dg(26:end,:);
    dg = dg(1:end-25,:);
    dg_ds = downsample(dg,50);
    dg_ds = dg_ds(:,finger);

    mat = create_R_matrix(feat_mat_p2, 3);
    
    trainx = mat;
    trainy = dg_ds;

    testx = feat_mat_test_p2;
    testx(:, [121:126, 223:228]) = [];
    test_mat = create_R_matrix(testx, 3);

    [coeff, score, latent, tsquared, explained] = pca(trainx);
    trainx_pca = score(:,1:800);
    testx_pca = test_mat*coeff;
    testx_pca = testx_pca(:,1:800);

    [B,S] = lasso(trainx_pca,trainy,'Lambda',.02);
    y_pred = testx_pca*B;
    movmeanY = movmean(y_pred, 42);
    interpY2 = interp(movmeanY(:,1), 50);
    D = padarray(interpY2,25,0,'both');
    Y_hat_p2(:,finger) = D;
end

clear feat_mat_p2 feats2, 

%%
load("feat_mat_p3.mat")
load("feat_mat_test_p3.mat")
load("final_proj_part1_data.mat")

Y_hat_p3 = [];
for finger = 1:5
    dg = train_dg{3};
    dg = dg(26:end,:);
    dg = dg(1:end-25,:);
    dg_ds = downsample(dg,50);
    dg_ds = dg_ds(:,finger);

    mat = create_R_matrix(feat_mat_p3, 3);
    
    trainx = mat;
    trainy = dg_ds;

    testx = feat_mat_test_p3;
    test_mat = create_R_matrix(testx, 3);

    [coeff, score, latent, tsquared, explained] = pca(trainx);
    trainx_pca = score(:,1:1000);
    testx_pca = test_mat*coeff;
    testx_pca = testx_pca(:,1:1000);

    [B,S] = lasso(trainx_pca,trainy,'Lambda',.02);
    y_pred = testx_pca*B;
    movmeanY = movmean(y_pred, 42);
    interpY2 = interp(movmeanY(:,1), 50);
    D = padarray(interpY2,25,0,'both');
    Y_hat_p3(:,finger) = D;
end

clear feats3 feat_mat_p2
%%

predicted_dg = {Y_hat_p1; Y_hat_p2; Y_hat_p3};


%%

% for i= 1:3
%     for j = 1:5
%         figure();
%         plot(predicted_dg{i}(:,j))
%     end
% end

%%
    

save("predicted_dg", "predicted_dg")


