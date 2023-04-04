function [all_feats]=getAllWindowedFeats(raw_data, fs)
    [subA,subB] = butter(2,[1, 60]/(fs/2));
    [gamA,gamB] = butter(2,[60, 100]/(fs/2));
    [fastA,fastB] = butter(2,[100, 200]/(fs/2));

    [numSamples,numChannels] = size(raw_data);
    subEcog = zeros(numSamples,numChannels);
    gammaEcog = zeros(numSamples,numChannels);
    fastGammaEcog = zeros(numSamples,numChannels);
    for i = 1:numChannels
        subEcog(:,i) = filtfilt(subA,subB,raw_data(:,i));
        gammaEcog(:,i) = filtfilt(gamA,gamB,raw_data(:,i));
        fastGammaEcog(:,i) = filtfilt(fastA,fastB,raw_data(:,i));
    end

    subEcogFeats = getBandWindowedFeats(subEcog, fs, 0.1, 0.05);
    gammaFeats = getBandWindowedFeats(gammaEcog, fs, 0.1, 0.05);
    fastGammaFeats = getBandWindowedFeats(fastGammaEcog, fs, 0.1, 0.05);
    all_feats = horzcat(subEcogFeats, gammaFeats, fastGammaFeats);
end
