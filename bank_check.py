import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from tkinter import *
from tkinter import ttk
import csv



def main():
    """

    @rtype: object
    """
    def show_message(text=[]):
        CreditScore = e_cs.get()
        Geography = e_geo.get()
        Gender = e_gen.get()
        Age = e_age.get()
        Tenure = e_ten.get()
        Balance = e_bal.get()
        NumOfProducts = e_nop.get()
        HasCrCard = e_cc.get()
        IsActiveMember = e_act.get()
        EstimatedSalary = e_sal.get()
        tur = (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        test = [('CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                'IsActiveMember', 'EstimatedSalary'),
                (1000, 'Spain', 'Female', 50, 6, 1323000.0, 3, 1, 1, 899321.75),
                (666, 'France', 'Male', 32, 4, 123000.0, 2, 1, 1, 89321.75),
                (666, 'Germany', 'Male', 32, 4, 123000.0, 2, 1, 1, 89321.75)]
        test.append(tur)
        with open('../../Desktop/bank_customer_outflow-main/output.csv', 'w', newline='') as f:
            csv.writer(f).writerows(test)
        dt = pd.read_csv('../../Desktop/bank_customer_outflow-main/output.csv')
        T = dt.iloc[:, 0:10].values
        label_T_country_encoder = LabelEncoder()
        T[:, 1] = label_T_country_encoder.fit_transform(T[:, 1])

        label_T_gender_encoder = LabelEncoder()
        T[:, 2] = label_T_gender_encoder.fit_transform(T[:, 2])

        transform = ColumnTransformer([("countries", OneHotEncoder(), [1])],
                                      remainder="passthrough")  # 1 is the country column
        T = transform.fit_transform(T)
        X_train, X_test, y_train, y_test = read_data()

        rf = RandomForestClassifier(n_estimators=100, bootstrap=True, class_weight=None, criterion='gini',
                                    max_depth=None,
                                    max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_samples_leaf=1,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=None, oob_score=False,
                                    random_state=None, verbose=0, warm_start=False)
        rf.fit(X_train, y_train)
        pac = PassiveAggressiveClassifier(random_state=35)
        pac.fit(X_train, y_train)
        mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=300, activation='relu', solver='adam')
        mlp_clf.fit(X_train, y_train)
        dtree = DecisionTreeClassifier(random_state=None, max_depth=None, criterion='gini', splitter='best',
                                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                       max_features=None, max_leaf_nodes=10, min_impurity_decrease=0.0,
                                       class_weight=None)
        dtree.fit(X_train, y_train)
        f_["text"] = f'forest predict {rf.predict(T)[3]}'
        t_["text"] = f'tree predict {dtree.predict(T)[3]}'
        p_["text"] = f'pac predict {pac.predict(T)[3]}'
        m_["text"] = f'mlpc predict {mlp_clf.predict(T)[3]}'
        print('Run successfully')



    root = Tk()


    # root['bg'] = '#fafafa'
    root.title('Bank customer outflow')
    root.geometry('600x550')

    canvas = Canvas(root, height=3000, width=2500)
    canvas.pack()

    # frame = Frame(root, bg='red')
    # frame.place(relx=0.15, rely=0.15, relwidth=0.7, relheight=0.7)

    res_forest = ttk.Label(canvas, text=forest(), font=40)
    res_forest.pack()
    res_tree = ttk.Label(canvas, text=tree(), font=40)
    res_tree.pack()
    res_pac = ttk.Label(canvas, text=pac_(), font=40)
    res_pac.pack()
    res_mlpc = ttk.Label(canvas, text=mlpc_(), font=40)
    res_mlpc.pack()

    e_cs = ttk.Entry()
    e_cs.pack()
    e_geo = ttk.Entry()
    e_geo.pack()
    e_gen = ttk.Entry()
    e_gen.pack()
    e_age = ttk.Entry()
    e_age.pack()
    e_ten = ttk.Entry()
    e_ten.pack()
    e_bal = ttk.Entry()
    e_bal.pack()
    e_nop = ttk.Entry()
    e_nop.pack()
    e_cc = ttk.Entry()
    e_cc.pack()
    e_act = ttk.Entry()
    e_act.pack()
    e_sal = ttk.Entry()
    e_sal.pack()

    btn = ttk.Button(canvas, text='Предсказать', command=show_message)
    btn.pack()

    f_ = ttk.Label()
    f_.pack()
    t_ = ttk.Label()
    t_.pack()
    p_ = ttk.Label()
    p_.pack()
    m_ = ttk.Label()
    m_.pack()
    root.mainloop()

def read_data():
    data = pd.read_csv(r'../../Desktop/bank_customer_outflow-main/Churn_Modelling.csv')
    data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

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

    X_train, X_test, y_train, y_test = read_data()

    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, class_weight=None, criterion='gini', max_depth=None,
                                max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=None, oob_score=False,
                                random_state=None, verbose=0, warm_start=False)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    return f'Accuracy Score of random decision forests: {round(accuracy_ * 100, 2)}%'


def pac_():
    """

    @rtype: object
    """

    X_train, X_test, y_train, y_test = read_data()

    pac = PassiveAggressiveClassifier(random_state=35)
    pac.fit(X_train, y_train)
    y_pred = pac.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    return f'Accuracy Score of Passive Aggresive Scassifier: {round(accuracy_ * 100, 2)}%'


def mlpc_():
    """

    @rtype: object
    """

    X_train, X_test, y_train, y_test = read_data()

    mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=300, activation='relu', solver='adam')
    mlp_clf.fit(X_train, y_train)
    y_pred = mlp_clf.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    return f'Accuracy Score of MLP Classifier: {round(accuracy_ * 100, 2)}%'


def tree():
    """

    @rtype: object
    """

    X_train, X_test, y_train, y_test = read_data()

    dtree = DecisionTreeClassifier(random_state=None, max_depth=None, criterion='gini', splitter='best',
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_features=None, max_leaf_nodes=10, min_impurity_decrease=0.0, class_weight=None)
    dtree.fit(X_train, y_train)

    y_pred = dtree.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)

    return f'Accuracy Score of Decision tree Classifier: {round(accuracy_ * 100, 2)}%'


if __name__ == "__main__":
     main()
     pass