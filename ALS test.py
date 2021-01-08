import pandas as pd, numpy as np, matplotlib.pyplot as plt
from utils import get_data
from config import TRAIN_PATH, VALIDATION_PATH
from tqdm import tqdm



def getInitialMatrix():
    """
    Gets data from files and creates user-item matrices
    :return: A, R user-item matrices
    """
    train = pd.read_csv(TRAIN_PATH)
    train = train.iloc[0:100]
    print("Getting 'A' matrix with rows: user and columns: movies...")
    A = train.pivot_table(columns=['Movie_ID_Alias'], index=['User_ID_Alias'], values='Ratings_Rating').fillna(0).values

    print(" 'A' matrix shape is", A.shape)

    print("Getting 'R' Binary Matrix of rating or no rating...")
    R = A > 0.5
    R[R == True] = 1
    R[R == False] = 0
    R = R.astype(np.float64, copy=False)

    return A, R


def runALS(A, R, n_factors, n_iterations, lambda_):
    '''
    Runs Alternating Least Squares algorithm in order to calculate matrix.
    :param A: User-Item Matrix with ratings
    :param R: User-Item Matrix with 1 if there is a rating or 0 if not
    :param n_factors: How many factors each of user and item matrix will consider
    :param n_iterations: How many times to run algorithm
    :param lambda_: Regularization parameter
    :return:
    '''
    print("Initiating ")
    lambda_ = 0.1
    n_factors = 3
    n, m = A.shape
    n_iterations = 20
    Users = 5 * np.random.rand(n, n_factors)
    Items = 5 * np.random.rand(n_factors, m)

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MSE_List = []

    print("Starting Iterations")
    for iter in tqdm(range(n_iterations)):
        for i, Ri in enumerate(R):
            Users[i] = np.linalg.solve(np.dot(Items, np.dot(np.diag(Ri), Items.T)) + lambda_ * np.eye(n_factors),
                                       np.dot(Items, np.dot(np.diag(Ri), A[i].T))).T
        print("Error after solving for User Matrix:", get_error(A, Users, Items, R))

        for j, Rj in enumerate(R.T):
            Items[:, j] = np.linalg.solve(np.dot(Users.T, np.dot(np.diag(Rj), Users)) + lambda_ * np.eye(n_factors),
                                          np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])))
        print("Error after solving for Item Matrix:", get_error(A, Users, Items, R))

        MSE_List.append(get_error(A, Users, Items, R))
        print('%sth iteration is complete...' % iter)

    print(MSE_List)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(1, len(MSE_List) + 1), MSE_List)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title('Python Implementation MSE by Iteration \n with %d users and %d movies' % A.shape)
    plt.savefig('Python MSE Graph.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    A, R = getInitialMatrix()
    runALS(A, R, n_factors=3, n_iterations=20, lambda_=.1)
