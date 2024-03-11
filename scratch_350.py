import datapane as dp
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Generate Sample Data
fake = Faker()
data = {'Name': [fake.random_int() for _ in range(100)],
        'Age': [fake.random_int(18, 80) for _ in range(100)],
        'Label': [fake.random_element(elements=('Yes', 'No')) for _ in range(100)]}

df = pd.DataFrame(data)

# Step 2: Train Dataset Preprocessing Tab
preprocessing_report = dp.Page(
    title="### Train Dataset Preprocessing",
    blocks=[
        dp.Text("### Train Dataset Preprocessing"),
        dp.DataTable(df.head(), caption="Preview of the Train Dataset"),
        dp.Table(df.describe(), caption="Summary Statistics")
    ]
)

# Step 3: Model Evaluation Tab
# Split the data
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

evaluation_tab1 = dp.Blocks(
    dp.Text("#### Label Distribution"),
)

evaluation_tab2 = dp.Blocks(
    dp.Text("#### Model Metrics"),
    dp.Text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}"),
    dp.Text("#### Classification Report"),
    dp.Text(classification_report(y_test, y_pred)),
)

evaluation_report = dp.Page(title="Model Evaluation", blocks=[evaluation_tab1, evaluation_tab2])

# Step 4: Final Model Training Tab
final_training_tab = dp.Blocks(
    dp.Text("#### Final Model Training"),
    dp.Code(f"""
    # Code for final model training
    final_model = RandomForestClassifier()
    final_model.fit(X, y)
    """)
)

final_training_report = dp.Page(title="Final Model Training", blocks=[final_training_tab])

# Step 5: Warnings Tab
warnings_tab = dp.Blocks(
    dp.Text("#### Warnings"),
    dp.Text("No warnings to display."),
)

warnings_report = dp.Page(title="Warnings", blocks=[warnings_tab])

# Step 6: Combine Tabs into a Single Report
train_pipeline_report = dp.Report(
    dp.Blocks(
        preprocessing_report,
        evaluation_report,
        final_training_report,
        warnings_report,
    )
)

# Step 7: Save and View the Report
train_pipeline_report.save(path="train_pipeline_report.html", open=True)
