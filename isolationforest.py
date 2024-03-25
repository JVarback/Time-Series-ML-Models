from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from datetime import datetime
from pyswarm import pso
import pandas as pd
import numpy as np
import shap
import os

class IFPSOSHAP:

    def __init__(self, model: IsolationForest, data, threshold:float, verbose:bool = False):
        self.threshold = threshold
        self.verbose = verbose
        self.model = model
        self.data = data

    # Anomaly scores
    def calculateAnomalyScores_if(self, X):
        """
        Calculate Anomaly scores

        Args:
        - X: Input data (Window).

        Returns:
        - A ndarray of anomaly scores
        """
        return self.model.decision_function(X)

    # Isolation Forest - Iterative
    def detectAnomaliesIterative(self, window_size: int):
        """
        Iterative window calculating IsolationForest anomly scoring 

        Args:
        - window_size: Iteration window size.
        - verbose: Should print information. Default is false.

        Returns:
        - A list of anomalies
        """
        anomalies = [] 

        for i in range(0, len(self.data) - window_size + 1, window_size):

            window_data = self.data[i:i + window_size]
            anomalies.append(self.calculateAnomalyScores_if(window_data))

            if self.verbose and i % (window_size * 10) == 0:
                print(f"{i} / {len(self.data) - window_size + 1}")

        return np.concatenate(anomalies)

    # IF - Sliding window
    def detectAnomaliesSlidingWindow(self, window_size: int, step_size: int = 1):
        """
        Sliding window calculating IsolationForest anomly scoring 

        Args:
        - input_data: Input data as whole.
        - window_size: Iteration window size.
        - step_size: Size of sliding each iteration. 
                    0-100, 1-101, 2-102 (step_size = 1, window_size = 100)
        - verbose: Should print information. Default is false.

        Returns:
        - A list of anomalies
        """
        anomalies = []

        for i in range(0, len(self.data) - window_size + 1, step_size):

            window_data = self.data[i:i + window_size]
            anomalies.append(self.calculateAnomalyScores_if(window_data))

            if self.verbose and i % (step_size * 10) == 0:
                print(f"{i} / {len(self.data) - window_size + 1}")

        return np.concatenate(anomalies)
    

    # Binary Particle Swarm optimization
    def psoOptimizeThreshold(self, window_size: int, shouldRunAvgSwarm):
        """
        Run Iterative PSO & IsolationForest

        Args:
        - window_size: Iteration window size.
        - verbose: Should print information. Default is false.

        Returns:
        - A tuple: the first is a list of anamolies, and the second is a float as threshold value for Isolation Forest
        """
        anomalies = []
        base_threshold = 0.05

        if shouldRunAvgSwarm:
            average_threshold = []
        else:
            best_optimal_threshold = 0

        for i in range(0, len(self.data) - window_size + 1, window_size):
            window_data = self.data[i:i + window_size]

            # Optimize threshold
            lb = np.array([0.05])  # Lower bound
            ub = np.array([0.1])   # Upper bound

            optimal_threshold, _ = pso(self.pso_objective, args=(
                window_data,), lb=lb, ub=ub, maxiter=3, swarmsize=15, debug = self.verbose)

            anomaly_scores = self.calculateAnomalyScores_if(window_data)
            anomalies.append(anomaly_scores)

            if shouldRunAvgSwarm:
                if optimal_threshold > base_threshold:
                    average_threshold.append(optimal_threshold)
            else:
                if optimal_threshold > best_optimal_threshold and optimal_threshold > base_threshold:
                    best_optimal_threshold = optimal_threshold

            if self.verbose and i % (window_size * 10) == 0:
                print(f"{i} / {len(self.data) - window_size + 1}")

        if shouldRunAvgSwarm:
            if self.verbose:
                if shouldRunAvgSwarm:
                    sortedList = sorted(average_threshold)
                    split = len(sortedList) // 2
                    upper_half = sortedList[split:]

                    average = sum(upper_half) / len(upper_half)

                if self.verbose:
                    print(f" Average Threshold: {average[0]}")
                return np.concatenate(anomalies), average[0]
        else:
            if self.verbose:
                print(f" Optimal Threshold: {best_optimal_threshold[0]}")

            return np.concatenate(anomalies), best_optimal_threshold[0]

    # PSO Objective
    def psoObjective(self, threshold, window_data):
        """
        Calculates the absolute difference between a PSO threshold and the percentile of anomaly scores

        Args:
        - threshold: PSO state threshold.
        - window_data: The input data on current window.

        Returns:
        - Return a float absolute float
        """
        anomaly_scores = self.calculateAnomalyScores_if(window_data)
        
        return np.abs(threshold - np.percentile(anomaly_scores, 100 * (1 - threshold)))

    # SHAP
    def calculateShapValues(self, X):
        """
        Create a SHAP explainer and return SHAP scores

        Args:
        - model: IsolationForest model.
        - X: The input data.

        Returns:
        - Return a matrix, where each vector corresponds to the SHAP values for each feature trough whole sequence
        """
        explainer = shap.Explainer(self.model)
        shap_values = explainer.shap_values(X)

        return shap_values
    
    # Plot
    def plot(self, anomaly_indices, color, threshold, shap_values, shouldRunSwarm):

        directory = './images'

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Plotting
        if self.verbose:
            print("Plotting starting ...")

        # Placeholder for using timesteps instead of datetime (due to dropping time tables)
        original_data = pd.DataFrame(
            {'original_data': self.data[:, 3]}, index=range(len(self.data)))

        # Fix back Feature columns
        num_features = self.data.shape[1]
        axes = ['X', 'Y', 'Z']
        time_series_columns = [f'Value{
            index // len(axes)} Axis {axes[index % len(axes)]}' for index in range(num_features)]

        # Prepare & Reverse engineering columns
        numFeatures = 4
        axis = 3
        axis_names = ['AxisX', 'AxisY', 'AxisZ']

        feature_names = []
        for i in range(numFeatures):
            feature_names.append(f'Value {i}') 

        # Plot - normalized_data
        fig, axs_normalized = plt.subplots(numFeatures, axis, figsize=(18, 18))
        fig.suptitle('Normalized Data')

        for i in range(numFeatures):
            for j in range(axis):
                    if axis == 1:
                            axs_normalized[i].plot(self.data[:, i])
                            axs_normalized[i].set_title(f'Feature {i}')
                    else:
                        for j in range(axis):
                            axs_normalized[i, j].plot(self.data[:, i * 3 + j])
                            axs_normalized[i, j].set_title(f'Feature {i} - {axis_names[j]}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{directory}/normalized_data' + str(datetime.now()) + '.png')
        plt.close()

        # IsolationForest Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(original_data.index,
                original_data['original_data'], label='Original (Normalized) Data')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Outer Ring - Axis X')
        ax.legend()

        if shap_values:
            if self.verbose:
                print("Finished SHAP ...")
                print("Saving SHAP figure ...")

            shap.summary_plot(shap_values, self.data,
                            feature_names=time_series_columns, show=False)
            plt.savefig(f'{directory}/IF_SHAP_'+ str(datetime.now()) +'.png')

            if self.verbose:
                print("Saved SHAP figure ...")

        ax.set_title('Anomaly Score Threshold ' + str(round(threshold, 4) * 1000) + '%')

        for anomaly_index in anomaly_indices:
            ax.axvline(x=anomaly_index, color=color, linestyle='-',
                    label='Anomaly Detected', zorder=10)

        if shouldRunSwarm:
            datafile = "IF_Swarm_"
        else:
            datafile = "IF_Iterative_"

        plt.savefig(f'{directory}/'+ datafile + str(datetime.now()) + '.png')

        if self.verbose:
            print("Saved figure " + datafile + '.png ' + "...")

    def run(self, window_size: int = 1000,  shouldUseSlidingWindow = True, shouldRunSwarm = True, shouldRunSHAP = True):

        optimal_threshold = 0
        shap_values = 0

        if self.verbose:
            print("Testing starting ...")

        df = pd.DataFrame()
        if shouldRunSwarm:
            df['anomaly_scores'], optimal_threshold = self.psoOptimizeThreshold(window_size=window_size)
        else:
            df = pd.DataFrame()
            if shouldUseSlidingWindow:
                df['anomaly_scores'] = self.detectAnomaliesSlidingWindow(window_size=window_size)
            else:
                df['anomaly_scores'] = self.detectAnomaliesIterative(window_size=window_size)

        if shouldRunSwarm:
            color = 'r'
            plotThreshold = optimal_threshold
            anomaly_indices = df[df['anomaly_scores'] > optimal_threshold].index
        else:
            color = 'g'
            plotThreshold = self.threshold
            anomaly_indices = df[df['anomaly_scores'] > self.threshold].index

        if shouldRunSHAP:
            if self.verbose:
                print("Running SHAP ...")
                print("May take time ...")

            shap_values = self.calculateShapValues(self.data)

        self.plot(anomaly_indices, color, plotThreshold, shap_values, shouldRunSwarm)

        