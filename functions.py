import statsmodels.api as sm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# properties of plot
color = "#115396"
font = "Cambria"
alpha = 0.5

def residual_plots(predictions_train, residuals_train, predictions_test, residuals_test, model_name: str):
    # creates two target vs residuals plots for inspection of model
    # reikia paderinti asiu range
    
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (16, 8))

    f.suptitle(f"Distributions of residuals for {model_name} model", fontsize=24, weight='bold', fontname=font)

    # ax1 subplot
    ax1.scatter(x=predictions_train, y=residuals_train, c=color, alpha=alpha)
    ax1.set_title("Target values against residuals (train data)", fontsize = 16, fontname = font)
    ax1.set_ylabel("Residual values", fontsize = 16, fontname = font)
    ax1.set_xlabel("Target values (train data)", fontname = font, fontsize = 16)

    # ax2 subplot
    ax2.scatter(x=predictions_test, y=residuals_test, c=color, alpha=alpha)
    ax2.set_title("Target values against residuals (test data)", fontsize = 16, fontname = font)
    ax2.set_xlabel("Target values (test data)", fontname = font, fontsize = 16);

def residual_hist_qq_plots(residuals_train, residuals_test):
    # creates histogram and qq plots for residuals of test and train data values

    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16, 10))

    f.suptitle("Distributions of residual values", fontsize=24, weight='bold', fontname=font)

    # ax1 subplot - histogram of train data residuals
    ax1.hist(residuals_train)
    ax1.set_title("Histogram of training data residuals", fontsize = 16, fontname = font)


    # ax2 subplot - qqplot of train values residuals
    sm.qqplot(residuals_train, line = 's', ax=ax2, )
    ax2.set_title("QQ plot of training data residuals", fontsize = 16, fontname = font)
    

    # ax3 subplot - histogram of test values residuals
    ax3.hist(residuals_test)
    ax3.set_title("Histogram of testing data residuals", fontsize = 16, fontname = font)


    # ax4 subplot  - qqplot of test values residuals             
    sm.qqplot(residuals_test, line = 's', ax=ax4)
    ax4.set_title("QQ plot of testing data residuals", fontsize = 16, fontname = font);
    
def real_vs_predicted_plots(y_train, predictions_train, y_test, predictions_test):
    ### function displays 2 subplots visualising true and predicted values of target values
    ### it is advised to give sorted values of target variables in order to get clearer view
    
    
    f, ((ax1), (ax2)) = plt.subplots(2, 1, sharey = True, figsize = (16, 8))

    f.suptitle("Real values vs Predictions", fontsize=24, weight='bold', fontname=font)

    # plot on train data
    ax1.scatter(np.arange(0, len(y_train)), y_train, color = color, label = 'real values', alpha = 0.3)
    ax1.scatter(np.arange(0, len(predictions_train)), predictions_train, color = 'r', label = 'predicted values', alpha = 0.3)
    ax1.set_xticklabels([])
    plt.setp(ax1.get_yticklabels(), fontweight="bold")
    ax1.set_title("Train data", fontsize = 16, fontname = font)

    ax1.legend(fontsize = 12)



    # plot on test data
    ax2.scatter(np.arange(0, len(y_test)), y_test, color = color, label = 'real values', alpha = 0.3)
    ax2.scatter(np.arange(0, len(predictions_test)), predictions_test, color = 'r', label = 'predicted values', alpha = 0.3)
    ax2.set_xticklabels([])
    plt.setp(ax2.get_yticklabels(), fontweight="bold")
    ax2.set_title("Test data", fontsize = 16, fontname = font)

    ax2.legend(fontsize = 12);