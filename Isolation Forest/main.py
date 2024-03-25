from sklearn.ensemble import IsolationForest
import isolationforest
import data

if __name__ == "__main__":

    directory = "./"                                # Directory of csv files

    threshold = 0.01                                # Threshold for anomly score
    window_size = 10000                             # Size of each iteration window
    fit_train_split = 0.8                           # Fit & Train split ratio (common: 80/20)

    IF_contamination = 0.1                          # Range of contamination used for IsolationForest
    IF_n_estimators = 100                           # Number of Isolation Trees
    IF_max_samples = 10000                          # Max number of Random samples
    IF_max_features = 10                            # Max number of Features to calculate from

    verbose = True                                  # Should print useful information
    shouldRunSHAP = False                           # Should run SHAP
    shouldRunSwarm = False                          # Should run Particle Swarm Optimization - Global Threshold
    shouldRunAvgSwarm = False                       # Should run Particle Swarm Optimization - Upperbound Average Threshold
    shouldHalfSequence = False                      # Should only use first half of the sequence
    shouldUseSlidingWindow = False                  # Should use Sliding window iteration (0-100), (1-101), (2-102)

    if verbose:
            print("Loading & processing data ...")

    csv_files = data.getCSVsFromDirectory(directory)
    fit_files, train_files = data.splitFiles(file_paths = csv_files, split_ratio = fit_train_split)     
    
    normalized_fit = data.loadDataForIsolationForest(fit_files, half_data = shouldHalfSequence, verbose = verbose)
    normalized_train = data.loadDataForIsolationForest(train_files, half_data = shouldHalfSequence, verbose = verbose) 

    if verbose:
        print("Data processed and loaded ...")

    model_IF = IsolationForest(contamination = IF_contamination, n_estimators=IF_n_estimators, max_samples = IF_max_samples,  max_features = IF_max_features, verbose = verbose)   
    model_IF.fit(normalized_fit)

    model = isolationforest.IFPSOSHAP(model = model_IF, data = normalized_train, threshold = threshold, verbose = verbose)
    if verbose:
        print("IsolationForest initiated and built ...")

    model.run(window_size, shouldUseSlidingWindow, shouldRunSwarm, shouldRunSHAP)