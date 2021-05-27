import pandas as pd
import numpy as np

from sklearn import datasets

from evidently.dashboard import Dashboard
from evidently.tabs import DriftTab, CatTargetDriftTab

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns = iris.feature_names)

iris_frame['target'] = iris.target

iris_data_drift_report = Dashboard(iris_frame[100:], iris_frame[100:], tabs = [DriftTab, CatTargetDriftTab])
iris_data_drift_report.save("reports/my_report_with_2_tabs.html")