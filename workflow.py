from prefect import task, flow
import subprocess
import sys
import os

@task(name="Fetch Data", retries=2, retry_delay_seconds=5)
def run_fetch_data():
    """Runs the data ingestion script."""
    print(f"🚀 Starting Data Ingestion using {sys.executable}...")
    

    script_path = "data_ingestion/fetch_data.py"
    
    subprocess.run([sys.executable, script_path], check=True)
    print("✅ Data Ingestion Complete")

@task(name="Build Features")
def run_build_features():
    """Runs the feature engineering script."""

    script_path = "feature_pipeline/build_features.py" 
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at: {script_path}")

    print("🔧 Starting Feature Engineering...")
    subprocess.run([sys.executable, script_path], check=True)
    print("✅ Feature Engineering Complete")

@task(name="Train Model")
def run_train_model():
    """Runs the model training script."""

    script_path = "training/train_model.py"
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found at: {script_path}")

    print("🧠 Starting Model Training...")
    subprocess.run([sys.executable, script_path], check=True)
    print("✅ Model Training Complete")

@flow(name="Apple Stock Training Pipeline")
def ml_pipeline():

    run_fetch_data()
    

    run_build_features()
    
    run_train_model()

if __name__ == "__main__":
    ml_pipeline()