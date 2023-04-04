function [band_feats]=getBandWindowedFeats(band_data, fs, window_length, window_disp)
    [numSamples, numChannels] = size(band_data);
    
    NumWins = @(xLen, fs, winLen, winDisp) floor((xLen - winLen*fs)/(winDisp*fs))+1;
    numWins = NumWins(numSamples, fs, window_length, window_disp);
    
    pos = 0;
    numFeat = numChannels;
    band_feats = zeros(numWins,numFeat);

    for i = 1:numWins
        windowData = band_data(1+pos:pos+(window_length*fs),:);
        pos = pos + (window_disp)*fs;
        band_feats(i,:) = get_band_features(windowData,fs);
    end     
end