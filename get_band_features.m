function [features] = get_band_features(band_window_data,fs)
    [numSamples,numChannels] = size(band_window_data);
    
    features = [];
    for n = 1:numChannels
        channelData = band_window_data(:, n);
        AM = 0;
        for t = 1:length(channelData)
            AM = AM + channelData(t)^2;
        end
        features = [features, AM];
    end
end

