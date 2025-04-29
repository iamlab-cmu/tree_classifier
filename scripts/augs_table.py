import wandb
import pandas as pd

api = wandb.Api()

project_path = (
    "rspears-carnegie-mellon-university/contact-sound-classification-augmentations-v3"
)

runs = api.runs(project_path)

run_data = []

for run in runs:
    run_name = run.name

    test_accuracy = None

    for key in [
        "test_accuracy",
        "accuracy/test",
        "test/accuracy",
        "accuracy_test",
        "test_acc",
    ]:
        if key in run.summary:
            test_accuracy = run.summary[key]
            break

    run_data.append({"Run Name": run_name, "Test Accuracy": test_accuracy})

df = pd.DataFrame(run_data)

if not df["Test Accuracy"].isna().all():
    df = df.sort_values(by="Test Accuracy", ascending=False)

print(df)

df.to_csv("wandb_test_accuracy.csv", index=False)

print("\nResults saved to wandb_test_accuracy.csv")
