# Time-series-ML-models

We have developed an anomaly detection system for time series data from hardware encoders. Our project focuses on creating two models for this task; an Isolation Forest model and a self-attention-based temporal convolutional neural network. The target achieved for these models has been to isolate outlier data points and differentiate between inliers and outliers for the Isolation forest model, and a focus on the most relevant parts of the time series data along the temporal axis for the other model. 

## Technologies Used
This project is implemented in Python, including the following libraries but not limited to:
- NumPy
- Pandas
- PyTorch
- Scikit-learn

The code is designed to run in a Python environment incorporating the above libraries.

You may set up your local environment and clone this repository for installation. To run the self-attention-based TCN model you may start from inference by using the pth file located in the folder. The data used has been in CSV format where each row represents a value and the column represents time. For our use case, we have pivoted the CSV files in our data preprocessing steps in order to get relevant features in the column space. 

## License
The project is licensed under an MIT license. 

## Contributions
We encourage contributions from the community, whether it's in the form of feature requests, bug reports, or pull requests.
