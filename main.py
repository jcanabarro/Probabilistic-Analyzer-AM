from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import kruskal, mannwhitneyu

from dataset import Dataset
from classifier import BestEstimator


def get_svm():
    return SVC(kernel='linear', probability=False, class_weight='balanced')


def get_knn():
    return KNeighborsClassifier(n_neighbors=3)


def get_nb():
    return GaussianNB()


def get_dt():
    return DecisionTreeClassifier()


def get_mlp():
    return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


def calculate_mannwhitne(classifiers_result):
    mann_result = []
    for i in range(len(classifiers_result)):
        for j in range(i + 1, len(classifiers_result)):
            result = mannwhitneyu(classifiers_result[i], classifiers_result[j], alternative="two-sided")
            mann_result.append((i, j, *result))
    return mann_result


def main():
    methods_scores = {'svm': [], 'knn': [], 'nb': [], 'dt': [], 'mlp': []}
    methods_names = ['svm', 'knn', 'nb', 'dt', 'mlp']
    methods_params = {
        'svm': [
            {'kernel': ['rbf', 'linear'],
             'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        ],
        'knn': [
            {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
             'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]}
        ],
        'dt': [
            {"max_leaf_nodes": [None, 5, 10, 20],
             "max_depth": [None, 5, 10, 20]}
        ],
        'mlp': [
            {'learning_rate': ["constant", "invscaling", "adaptive"],
             'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,),
                                    (60,), (70,), (80,), (90,), (100,)],
             'max_iter': [100, 200, 300, 400, 500]}
        ]
    }

    for i in range(10):
        dataset = Dataset('Base/Adult.csv')

        x_train, y_train = dataset.get_train_set()
        x_validation, y_validation = dataset.get_validation_set()
        x_test, y_test = dataset.get_test_set()

        optimizer = BestEstimator(x_train, x_validation, x_test, y_train, y_validation, y_test)
        methods = {'svm': get_svm(), 'knn': get_knn(), 'nb': get_nb(), 'dt': get_dt(), 'mlp': get_mlp()}

        for name in methods_names:
            optimizer.train_classifier(methods[name])
            if name != 'nb':
                optimizer.set_best_param(methods[name], methods_params[name])
            methods_scores[name].append(optimizer.get_classifier_score(methods[name]))

    methods_scores_fm = tuple(methods_scores[name] for name in methods_names)
    kruskal_h, kruskal_pvalue = kruskal(*methods_scores_fm)
    if kruskal_pvalue < 0.05:
        print("Significativo: %f" % kruskal_pvalue)
        for result in calculate_mannwhitne(methods_scores_fm):
            i, j, mann_z, mann_pvalue = result
            if mann_pvalue < 0.05:
                print("Significativo entre as instâncias %s e %s, valor de p %f" % (methods_names[i], methods_names[j], mann_pvalue))
            else:
                print("Não significativo entre as instâncias %s e %s, valor de p %f" % (methods_names[i], methods_names[j], mann_pvalue))
    else:
        print("Resultado não significativo %f" % kruskal_pvalue)
    print(methods_scores)


if __name__ == '__main__':
    main()
