# A-TCN & IsolationForest

We have developed an anomaly detection system for time series data from hardware encoders. Our project focuses on creating two models for this task; an Isolation Forest model and a self-attention-based temporal convolutional neural network (A-TCN). 
The target achieved for these models has been to isolate outlier data points and differentiate between inliers and outliers for the Isolation forest model, and A-TCN to focus on crucial segments along the temporal axis within time series.

## Technologies Used
This project is implemented in Python, including the following libraries but not limited to:
- NumPy
- Pandas
- PyTorch
- Scikit-learn
- Pyswarm
- Matplotlib
- SkLearn

The code is designed to run in a Python environment incorporating the above libraries.
If credited, you may set up your environment and freely clone this repository for installation. 

During data pre-processing, you may require to adjust column names in the CSV as needed. 
Whether your dataset requires fewer or additional features, you can adjust the input features for the A-TCN accordingly. 
The IsolationForest handles dynamic data effectively, while certain plotting sections may require minor adjustments.

To run the ATCN model you may start from inference by using the pth file located in the folder. The data used has been in CSV format where each row represents a value and the column represents time. 
For our use case, we have pivoted the CSV files in our data preprocessing steps in order to get relevant features in the column space.    

For IsolationForest we use similar data processing, you may further use the boolean values for choosing to run IsolationForest by itself, or include Swarm Optimization and SHAP for feature highlighting.
When enabling Swarm Optimization, there is one addiontal boolean that dictate two modes - global or average upper-bound.

## License
The project is licensed under an MIT license. 

## Contributions
We encourage contributions from the community, whether it's in the form of feature requests, bug reports, or pull requests.
