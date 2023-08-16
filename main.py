from SVM import *
from utils import *

if __name__ == "__main__":

    print('Training Model...')
    model = DeliriumPredictor.from_dataset('./data.csv')
    (
        X_train,
        X_test,
        y_train,
        y_test,
        transformer
    ) = (
        model.X_train,
        model.X_test,
        model.y_train,
        model.y_test,
        model.transformer
    )
    print('\nStoring Model...')
    model.store_model('SVM.pkl')

    print('\nResults:')
    get_results(y_train, model.predict(X_train)[0], model.predict(X_train)[1], text='Training')
    get_results(y_test, model.predict(X_test)[0], model.predict(X_test)[1], text='Test')
    get_table_2(transformer, model)
    get_intercept(model)
    get_platt_scaling(model)

    plt.style.use('ggplot')
    plt.rcParams.update({'figure.autolayout': True})
    get_roc_curve(model)
    get_boxplot(model)
