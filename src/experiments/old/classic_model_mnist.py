"""
This script is just to test if the mnist dataset can be classically separated correctly with a simple model
"""
from icecream import ic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.utils import accuracy_score
import argparse
from sklearn.svm import SVC
from src.Sampler import MNISTSampler, Sampler
from warnings import filterwarnings

filterwarnings('ignore', category=FutureWarning)

SEED = 42

################ TUNING PARAMS #################################
parser = argparse.ArgumentParser()
parser.add_argument('--n_train', type=int, help='# points for train set', required=False, default=400)
parser.add_argument('--n_test', type=int, help='# points for test set', required=False, default=200)
parser.add_argument('--dataset', type=str, help='dataset to be used', required=False, default='MNIST')

args = parser.parse_args()
n_train = int(args.n_train)
n_test = int(args.n_test)
dataset = args.dataset

################ CREATE DATASET #################################
if dataset.lower() == 'mnist':
    train_set, train_labels, test_set, test_labels = MNISTSampler.fashion(n_train=n_train, points_dimension=3,
                                                                           n_test=n_test, seed=SEED)
elif dataset.lower() == 'circles':
    train_set, train_labels = Sampler.circle(n_points=n_train, seed=SEED)
    test_set, test_labels = Sampler.circle(n_points=n_test, seed=SEED)


################ TUNING EXPERIMENT FUNCTIONS #################################
def train_and_evaluate_classic_model():
    classic_model = RandomForestClassifier(random_state=SEED)
    classic_model.fit(train_set, train_labels)

    pred_train = classic_model.predict(train_set)
    ic(train_labels.shape, pred_train.shape)
    train_acc = accuracy_score(train_labels, pred_train)
    pred_test = classic_model.predict(test_set)
    test_acc = accuracy_score(test_labels, pred_test)

    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')


def model_pablo():
    clf = SVC(kernel='rbf', random_state=1234)
    clf.fit(train_set, train_labels)

    train_labels_pred = clf.predict(train_set)
    test_labels_pred = clf.predict(test_set)

    train_accuracy = accuracy_score(train_labels, train_labels_pred)
    test_accuracy = accuracy_score(test_labels, test_labels_pred)

    print(f'Training Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

################ MAIN #################################
if __name__ == '__main__':
    print('%%%%%%%%%%%%%%%%%')
    print('Random Forest from GPT')
    train_and_evaluate_classic_model()

    print('%%%%%%%%%%%%%%%%%')
    print('Modelo Pablo SVC')
    model_pablo()

"""
Copio aqui los resultados obtenidos:
%%%%%%%%%%%%%%%%%
Random Forest from GPT
ic| train_labels.shape: (400,), pred_train.shape: (400,)
Training Accuracy: 1.0000
Test Accuracy: 0.8500
%%%%%%%%%%%%%%%%%
Modelo Pablo SVC
Training Accuracy: 0.9050
Test Accuracy: 0.8850

"""

