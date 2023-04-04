function [R]=create_R_matrix_release(features, N_wind)
    num_samp = size(features,1);
    N_1_rows = features(1:N_wind-1,:);
    features = [N_1_rows; features];

    R = zeros(num_samp, N_wind*size(features,2));
    offset = 1;
    counter = 1;
    for i = 1:size(features,2)*N_wind
        ft_samp = features(:,counter);
        R(:,i) = ft_samp(offset:num_samp + offset - 1);
        offset = offset + 1;
        if offset == N_wind+1
            offset = 1;
            counter = counter + 1;
        end
    end
    
    R = [ones(num_samp,1), R];
end
