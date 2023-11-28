import pandas as pd
from sklearn.datasets import load_svmlight_file
  
def convert_svmlight_to_dataframe(filepath):
    X, y = load_svmlight_file(filepath)
    df = pd.DataFrame(X.todense())
    df['target'] = y
    return df
  

# Loop through days 0-100
for i in range(101):  # Change this range according to your needs
    # Load SVM light file and convert to DataFrame
    df = convert_svmlight_to_dataframe(f'Day{i}.svm')  # Replace with the correct filepath pattern

    # Save DataFrame to a CSV file
    df.to_csv(f'newfile{i}.csv', index=False)

    print(f"File has been saved as newfile{i}.csv")
