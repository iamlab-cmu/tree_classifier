import wandb
import pandas as pd

# Initialize the wandb API
api = wandb.Api()

# Specify the project path
project_path = "rspears-carnegie-mellon-university/contact-sound-classification-augmentations-v3"

# Get all runs from the project
runs = api.runs(project_path)

# Create a list to store run data
run_data = []

# Extract information from each run
for run in runs:
    # Get run name
    run_name = run.name
    
    # Get test accuracy from the run summary
    # Note: The exact key might be different depending on how data is logged
    # Common keys might be 'test_accuracy', 'accuracy/test', etc.
    test_accuracy = None
    
    # Try different possible keys for test accuracy
    for key in ['test_accuracy', 'accuracy/test', 'test/accuracy', 'accuracy_test', 'test_acc']:
        if key in run.summary:
            test_accuracy = run.summary[key]
            break
    
    # Add data to our list
    run_data.append({
        'Run Name': run_name,
        'Test Accuracy': test_accuracy
    })

# Create a DataFrame
df = pd.DataFrame(run_data)

# Sort by accuracy (if available)
if not df['Test Accuracy'].isna().all():
    df = df.sort_values(by='Test Accuracy', ascending=False)

# Print the table
print(df)

# Optionally, save to CSV
df.to_csv('wandb_test_accuracy.csv', index=False)

print("\nResults saved to wandb_test_accuracy.csv")
