from featureSelector import Selector


def executeProgram():
    #selector = Selector('../risk_factors_cervical_cancer.csv')
    selector = Selector('../risk_factors_cervical_cancer.csv')
    featureSelectedDS = selector.apply_feature_selection(10)
    print(featureSelectedDS)

if __name__ == '__main__':
    executeProgram()