import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def month_to_index(month):
    """
    Convert month name to index.
    """
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "June",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    return months.index(month)


def convert_visitor_type(visitor_type):
    """
    Convert visitor type to integer.
    """
    return 1 if visitor_type == "Returning_Visitor" else 0


def convert_weekend(weekend):
    """
    Convert weekend to integer.
    """
    return 1 if weekend == "TRUE" else 0


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    #  open the file 
    with open(filename) as file:
        reader = csv.reader(file)
        header = next(reader)

        # Initialize lists for evidence and labels
        evidence = []
        labels = []

        # Iterate through each row in the CSV file
        for row in reader:
            # Convert each value to the appropriate type
            evidence_row = [
                int(row[0]),
                float(row[1]),
                int(row[2]),
                float(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[9]),
                month_to_index(row[10]),  # Month
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                convert_visitor_type(row[15]),  # VisitorType
                convert_weekend(row[16]),  # Weekend
            ]
            label = int(bool(row[17]))  # Revenue

            # Append to the lists
            evidence.append(evidence_row)
            labels.append(label)
    
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positive = sum(1 for actual, pred in zip(labels, predictions) if actual == 1 and pred == 1)
    true_negative = sum(1 for actual, pred in zip(labels, predictions) if actual == 0 and pred == 0)
    false_positive = sum(1 for actual, pred in zip(labels, predictions) if actual == 0 and pred == 1)
    false_negative = sum(1 for actual, pred in zip(labels, predictions) if actual == 1 and pred == 0)

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

    return sensitivity, specificity


if __name__ == "__main__":
    main()
