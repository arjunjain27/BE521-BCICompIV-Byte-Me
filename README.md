# BE521-BCICompIV-Byte-Me
BCI signal processing and classification model decoding finger movements in ECoG (https://www.bbci.de/competition/iv/index.html)

final_algorithm contains:
\make_predictions.m - script that takes in raw ECoG test data and returns finger flexion predictions in the format of the "predicted_dg" cell arrays
\lasso_weights.mat - lasso regression-based model we developed and trained to be loaded in script
\other .m files to support data processing

quick_test contains:
\prefeat_predictions.m - a test script that locally loads ECoG test data and pre-extracted feature matrices and returns finger flexion prediction arrays
\other .m files to support data processing
