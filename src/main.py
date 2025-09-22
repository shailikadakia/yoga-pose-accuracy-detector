from load_yoga_training_data import load_training_date
from make_pose_dataset_csv import pose_dataset
from make_angles_dataset_csv  import angles_dataset
from train_model import train

RAW_DIR = "./training-data"
DATA_DIR = "./data"
POSE_CSV = "./src/pose_dataset.csv"
ANGLES_CSV = "./src/pose_angles_dataset.csv"  
BUNDLE = "./src/pose_knn_runtime.pkl"  

def prepare():
  # Step 1: Load training data 
  load_training_date(RAW_DIR, DATA_DIR)

  # Step 2: Create a CSV file with all training data (33 landmarks per pose). 
  # Output: pose_dataset.csv
  pose_dataset(DATA_DIR, POSE_CSV)

  # Step 3: Create a CSV file with each row in pose dataset used to determine angles 
  # Input: pose_dataset.csv
  # Output: pose_angles_dataset.csv  
  angles_dataset(POSE_CSV, ANGLES_CSV)
  print("✅ prepare complete")


def train():
  if not ANGLES_CSV.exists():
    print("Angles CSV missing; running prepare first…")
    prepare()
  train(ANGLES_CSV, BUNDLE)

def test()
  
def main():


if __name__ == '__main__':
  main()
