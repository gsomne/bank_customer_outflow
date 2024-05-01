import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier



def main():
    """

    @rtype: object
    """


    forest()
    tree()
    pac_()
    mlpc_()


def read_data():
    data = pd.read_csv(r'Churn_Modelling.csv')
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    #print(data)
    #data.info()
    #data.describe()

    X = data.iloc[:, 0:10].values
    y = data.iloc[:, 10].values

    label_X_country_encoder = LabelEncoder()
    X[:, 1] = label_X_country_encoder.fit_transform(X[:, 1])

    label_X_gender_encoder = LabelEncoder()
    X[:, 2] = label_X_gender_encoder.fit_transform(X[:, 2])

    transform = ColumnTransformer([("countries", OneHotEncoder(), [1])],
                                  remainder="passthrough")  # 1 is the country column
    X = transform.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


def  forest():
    """

    @rtype: object
    """
    #read_data()
    X_train, X_test, y_train, y_test = read_data()

    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, class_weight=None, criterion='gini', max_depth=None,
                                max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=None, oob_score=False,
                                random_state=None, verbose=0, warm_start=False)
    rf.fit(X_train, y_train)
    # predictions_rf = rf.predict(X_test)
    #print(classification_report(y_test, predictions_rf))
    #print(confusion_matrix(y_test, predictions_rf))
    y_pred = rf.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    print(f'Accuracy Score of random decision forests: {round(accuracy_*100, 2)}%')


def pac_():
    """

    @rtype: object
    """
    #read_data()
    X_train, X_test, y_train, y_test = read_data()

    pac = PassiveAggressiveClassifier(random_state=35)
    pac.fit(X_train, y_train)
    y_pred = pac.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    print(f'Accuracy Score of Passive Aggresive Scassifier: {round(accuracy_ * 100, 2)}%')


def mlpc_():
    """

    @rtype: object
    """
    #read_data()
    X_train, X_test, y_train, y_test = read_data()

    mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=300, activation='relu', solver='adam')
    mlp_clf.fit(X_train, y_train)
    y_pred = mlp_clf.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    print(f'Accuracy Score of MLP Classifier: {round(accuracy_*100,2)}%')


def tree():
    """

    @rtype: object
    """

    #read_data()
    X_train, X_test, y_train, y_test = read_data()

    dtree = DecisionTreeClassifier(random_state=None, max_depth=None, criterion='gini', splitter='best',
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_features=None, max_leaf_nodes=10, min_impurity_decrease=0.0, class_weight=None)
    dtree.fit(X_train, y_train)

    # predictions_dtree = dtree.predict(X_test)
    y_pred = dtree.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    #print(classification_report(y_test, predictions_dtree))
    #print(confusion_matrix(y_test, predictions_dtree))

    print(f'Accuracy Score of Decision tree Classifier: {round(accuracy_*100, 2)}%')


if __name__ == "__main__":
     main()
     pass