# Data Drift report

import pandas as pd
from sklearn import datasets

from evidently.dashboard import Dashboard
from evidently.tabs import DriftTab

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns = iris.feature_names)

iris_data_drift_report = Dashboard(iris_frame[:100], iris_frame[100:], tabs = [DriftTab])
iris_data_drift_report.save("reports/my_report.html")