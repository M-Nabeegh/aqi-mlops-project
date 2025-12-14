import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference = pd.read_csv("features.csv").sample(100)
current = pd.read_csv("features.csv").sample(100)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

report.save_html("drift_report.html")
print("Drift report generated")