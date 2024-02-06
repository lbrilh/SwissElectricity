import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, RationalQuadratic, ConstantKernel, WhiteKernel, ExpSineSquared, PairwiseKernel
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.impute import KNNImputer

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    """
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2)) # preview training data
    print('\n')

    """
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    """"
    print("Test data:")
    print(test_df.shape)    
    print(test_df.head(2))
    """

    X_train = np.zeros((900,9))

    # extract seasonal data
    spring = train_df[(train_df.values == "spring").any(axis=1)].drop(["season"],axis=1).to_numpy()
    summer = train_df[(train_df.values == "summer").any(axis=1)].drop(["season"],axis=1).to_numpy()
    autumn = train_df[(train_df.values == "autumn").any(axis=1)].drop(["season"],axis=1).to_numpy()
    winter = train_df[(train_df.values == "winter").any(axis=1)].drop(["season"],axis=1).to_numpy()
    
    spring_test = test_df[(test_df.values == "spring").any(axis=1)].drop(["season"],axis=1).to_numpy()
    summer_test = test_df[(test_df.values == "summer").any(axis=1)].drop(["season"],axis=1).to_numpy()
    autumn_test = test_df[(test_df.values == "autumn").any(axis=1)].drop(["season"],axis=1).to_numpy()
    winter_test = test_df[(test_df.values == "winter").any(axis=1)].drop(["season"],axis=1).to_numpy()

    # k-nearest neighbors imputation for seasonal data
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    spring_imp = imputer.fit_transform(spring)
    summer_imp = imputer.fit_transform(summer)
    autumn_imp = imputer.fit_transform(autumn)
    winter_imp = imputer.fit_transform(winter)
    spring_test_imp = imputer.fit_transform(spring_test)
    summer_test_imp = imputer.fit_transform(summer_test)
    autumn_test_imp = imputer.fit_transform(autumn_test)
    winter_test_imp = imputer.fit_transform(winter_test)
    
    # obtain imputed data
    train_data = np.zeros((900,10))
    test_data = np.zeros((100,9))
    for j in range(0,len(spring_imp)):
        train_data[j*4,:] = spring_imp[j,:]
        train_data[j*4+1,:] = summer_imp[j,:]
        train_data[j*4+2,:] = autumn_imp[j,:]
        train_data[j*4+3,:] = winter_imp[j,:]
    for j in range(0, len(spring_test_imp)):
        test_data[j*4,:] = spring_test_imp[j,:]
        test_data[j*4+1,:] = summer_test_imp[j,:]
        test_data[j*4+2,:] = autumn_test_imp[j,:]
        test_data[j*4+3,:] = winter_test_imp[j,:]

    # print(test_data)

    
    # extract X_train, Y_train, X_test
    X_test = test_data
    X_train[:,0] = train_data[:,0]
    X_train[:,1:8] = train_data[:,2:9]
    y_train = train_data[:,1] # price_CHF is second column

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    """"
    score = []

    score.append(cross_val_score(GaussianProcessRegressor(kernel = DotProduct(), alpha=0.38), X_train, y_train, cv=5, scoring = "r2"))
    score.append(cross_val_score(GaussianProcessRegressor(kernel = RBF(), alpha=0.15), X_train, y_train, cv=3, scoring = "r2"))
    score.append(cross_val_score(GaussianProcessRegressor(kernel = RationalQuadratic(), alpha=0.38), X_train, y_train, cv=5, scoring = "r2"))
    score.append(cross_val_score(GaussianProcessRegressor(alpha=0.38), X_train, y_train, cv=5, scoring = "r2"))
    score.append(cross_val_score(KernelRidge(kernel = "linear", alpha=0.22), X_train, y_train, cv=5, scoring = "r2"))
    alpha = 0.01 best cv = 3, k = 2 == 0.9689
    score.append(cross_val_score(KernelRidge(kernel = "rbf", alpha=0.01), X_train, y_train, cv=3, scoring = "r2"))
    score.append(cross_val_score(KernelRidge(kernel='polynomial', degree=4, alpha=40, coef0=1.5), X_train, y_train, cv=4, scoring = "r2"))
    score.append(cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring = "r2"))

    score = np.array(score)
    print(score.mean(axis=1))
    """

    clf = KernelRidge(kernel='polynomial', degree=4, alpha=30, coef0=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # print(r2_score(y_train, y_pred))

    #print(y_pred)
    
    """
    plt.bar(['G Dot Product', 'G RBF', "G Rational Quadratic", 'GPR', "linear", "RBF",'polynomial', "LR"], score.mean(axis=1))
    plt.title('5-fold CV R^2 score for different kernels')
    plt.xlabel('Kernels')
    plt.ylabel('R^2 score')
    plt.show()
    """

    # r2_score(y_true, y_pred, squared=False)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()

    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

