import pip
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def import_or_install(package, packageDownload):
    packageDownload = packageDownload if packageDownload != '' else package
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', packageDownload])
        __import__(package)



def metrics(model_name, actual, predictions):
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    rmsle = np.log1p(rmse)  # safer than log() in case rmse is very small
    r2 = r2_score(actual, predictions)
    
    # Create a DataFrame with one row
    df = pd.DataFrame([{
        'model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'RMSLE': rmsle,
        'R2': r2
    }])
    
    df.set_index('model', inplace=True)
    return df