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
        CreditScore = int(e_cs.get())
        Geography = e_geo.get()
        Gender = e_gen.get()
        Age = int(e_age.get())
        Tenure = int(e_ten.get())
        Balance = float(e_bal.get())
        NumOfProducts = int(e_nop.get())
        HasCrCard = CreditCardInfo.get()
        IsActiveMember = ClientStatus.get()
        EstimatedSalary = float(e_sal.get())
        tur = (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        test = [('CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                'IsActiveMember', 'EstimatedSalary'),
                (1000, 'Spain', 'Female', 50, 6, 1323000.0, 3, 1, 1, 899321.75),
                (666, 'France', 'Male', 32, 4, 123000.0, 2, 1, 1, 89321.75),
                (666, 'Germany', 'Male', 32, 4, 123000.0, 2, 1, 1, 89321.75)]
        test.append(tur)
        with open('output.csv', 'w', newline='') as f:
            csv.writer(f).writerows(test)
        dt = pd.read_csv('output.csv')
        T = dt.iloc[:, 0:10].values
        label_T_country_encoder = LabelEncoder()
        T[:, 1] = label_T_country_encoder.fit_transform(T[:, 1])

        label_T_gender_encoder = LabelEncoder()
        T[:, 2] = label_T_gender_encoder.fit_transform(T[:, 2])

        transform = ColumnTransformer([("countries", OneHotEncoder(), [1])],
                                      remainder="passthrough")  # 1 is the country column
        T = transform.fit_transform(T)
        pred_forest = rf.predict(T)[3]
        pred_tree = dtree.predict(T)[3]
        pred_pac = pac.predict(T)[3]
        pred_mlp = mlp_clf.predict(T)[3]
        f_["text"] = f'forest predict {pred_forest}'
        t_["text"] = f'tree predict {pred_tree}'
        p_["text"] = f'pac predict {pred_pac}'
        m_["text"] = f'mlpc predict {pred_mlp}'
        sr = pred_forest + pred_tree + pred_pac + pred_mlp
        if sr / 4 > 0.5:
            res_["text"] = 'Вероятно уйдет'
        elif sr / 4 < 0.5:
            res_["text"] = 'Вероятно не уйдет'
        else:
            res_["text"] = 'Нельзя сказать точно'
        print('Run successfully')


    data = pd.read_csv(r'Churn_Modelling.csv')
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

    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, class_weight=None, criterion='gini', max_depth=None,
                                max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0, n_jobs=None, oob_score=False,
                                random_state=None, verbose=0, warm_start=False)
    rf.fit(X_train, y_train)

    forest_y_pred = rf.predict(X_test)
    accuracy_forest = accuracy_score(y_test, forest_y_pred)
    res_forest = f'Accuracy Score of random decision forests: {round(accuracy_forest * 100, 2)}%'

    pac = PassiveAggressiveClassifier(random_state=35)
    pac.fit(X_train, y_train)
    pac_y_pred = pac.predict(X_test)
    accuracy_pac = accuracy_score(y_test, pac_y_pred)
    res_pac = f'Accuracy Score of Passive Aggresive Scassifier: {round(accuracy_pac * 100, 2)}%'

    mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=300, activation='relu', solver='adam')
    mlp_clf.fit(X_train, y_train)
    mlp_y_pred = mlp_clf.predict(X_test)
    mlp_accuracy_ = accuracy_score(y_test, mlp_y_pred)
    res_mlp = f'Accuracy Score of MLP Classifier: {round(mlp_accuracy_ * 100, 2)}%'

    dtree = DecisionTreeClassifier(random_state=None, max_depth=None, criterion='gini', splitter='best',
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_features=None, max_leaf_nodes=10, min_impurity_decrease=0.0, class_weight=None)
    dtree.fit(X_train, y_train)
    tree_y_pred = dtree.predict(X_test)
    tree_accuracy_ = accuracy_score(y_test, tree_y_pred)
    res_tree = f'Accuracy Score of Decision tree Classifier: {round(tree_accuracy_ * 100, 2)}%'


    root = Tk()
    root.title('Bank customer outflow')
    root.geometry('400x700')

    canvas = Canvas(root, height=3500, width=2500)
    canvas.pack()

    CreditCardInfo = IntVar()
    ClientStatus = IntVar()

    res_forest_ttk = ttk.Label(canvas, text=res_forest)
    res_forest_ttk.pack()
    res_tree_ttk = ttk.Label(canvas, text=res_tree)
    res_tree_ttk.pack()
    res_pac_ttk = ttk.Label(canvas, text=res_pac)
    res_pac_ttk.pack()
    res_mlpc_ttk = ttk.Label(canvas, text=res_mlp)
    res_mlpc_ttk.pack()

    ttk.Label(canvas, text='Кредитный рейтинг').pack()
    e_cs = ttk.Spinbox(canvas, from_=0.0, to=1000.0)
    e_cs.pack()
    ttk.Label(canvas, text='Страна').pack()
    e_geo = ttk.Combobox(canvas, values=['Spain', 'France', 'Germany'])
    e_geo.pack()
    ttk.Label(canvas, text='Пол').pack()
    e_gen = ttk.Combobox(canvas, values=['Female', 'Male'])
    e_gen.pack()
    ttk.Label(canvas, text='Возраст').pack()
    e_age = ttk.Spinbox(canvas, from_=18.0, to=100.0)
    e_age.pack()
    ttk.Label(canvas, text='Сколько лет клиент').pack()
    e_ten = ttk.Spinbox(canvas, from_=1.0, to=82.0)
    e_ten.pack()
    ttk.Label(canvas, text='Баланс').pack()
    e_bal = ttk.Spinbox(canvas, from_=0.0, to=1000000000.0)
    e_bal.pack()
    ttk.Label(canvas, text='Кол-во продуктов').pack()
    e_nop = ttk.Spinbox(canvas, from_=0.0, to=4.0)
    e_nop.pack()
    e_cc = ttk.Checkbutton(canvas, text="Есть кредитная карта", variable=CreditCardInfo)
    e_cc.pack()
    e_act = ttk.Checkbutton(canvas, text="Активный пользователь", variable=ClientStatus)
    e_act.pack()
    ttk.Label(canvas, text='Зарплата').pack()
    e_sal = ttk.Spinbox(canvas, from_=0.0, to=1000000000.0)
    e_sal.pack()

    btn = ttk.Button(canvas, text='Предсказать', command=show_message)
    btn.pack()

    f_ = ttk.Label(canvas)
    f_.pack()
    t_ = ttk.Label(canvas)
    t_.pack()
    p_ = ttk.Label(canvas)
    p_.pack()
    m_ = ttk.Label(canvas)
    m_.pack()
    res_ = ttk.Label(canvas)
    res_.pack()



    root.mainloop()


if __name__ == "__main__":
     main()