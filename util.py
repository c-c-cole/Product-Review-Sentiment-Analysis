import pandas as pd

from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import classification_report

def test (y_test, y_pred):
    print("y_test distribution:", y_test.value_counts())
    # This is the key metric! We see that ALL predictions are 5. Overfitting!
    print("y_pred distribution:", pd.Series(y_pred).value_counts())

    print("classification report" + str(classification_report(y_test, y_pred)))

    mcc = matthews_corrcoef(y_test, y_pred)
    print("Matthews Correlation Coefficient:", mcc)
    if mcc == 1:
        print("Perfectly Reliable")
    elif 0.70 <= mcc < 1.0:
        print("Excellent Reliability")
    elif 0.50 <= mcc < 0.7:
        print("Good Reliability")
    elif 0.30 <= mcc < 0.5:
        print("Moderate Reliability")
    elif 0.00 < mcc < 0.3:
        print("Weak Reliability")
    elif mcc == 0:
        print("Model performs no better than random predictions")
    else:
        print("Model predictions inversely related to actual outcomes")

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    if accuracy >= .80:
        print("Yippee")