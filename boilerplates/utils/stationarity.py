import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller


def find_min_d_FFD(df, corr_thres=.9, p_thres=.05, col="CLOSE", verbose=False):
    """
    Finds the minimum `d` value that is close to the ADF critical value limit. 
    Selects for the value against correlation and p value thresholds
    
    Params
    --------
    df          : pd.DataFrame
        Series to find `d` for
    corr_thres  : float
        The threshold for correlation between diffed series and original
    p_thres     : float
        P value for rejecting ADFuller stationarity test
    col         : str
        The column in the dataframe to diff
    verbose     : bool
        Whether to log all values to stdout
    
    Returns
    --------
    tuple
        A tuple containing (a) the d value, and (b) the ADF statistic
    """
    d_space = np.linspace(0, 1, 100)
    
    if verbose:
        print("Finding best d value for column {}".format(col))
        print("====================\n")
        d_space = tqdm(d_space)
    
    for d in d_space:
        df1 = df[[col]]
        df2 = frac_diff_FFD(df[[col]], d, .01)
        
        # get correlation
        corr = np.corrcoef(df1.loc[df2.index, col], df2[col])[0,1]
        adf_results = adfuller(df2[col], maxlag=1, autolag=None)
        p_value = adf_results[1]
        
        if p_value <= p_thres:
            if corr < corr_thres:
                print("No d value for both p value and correlation thresholds")
                print("CORRELATION", corr)
                return None, None
        
            if verbose:
                print("\n\n====================")
                print("Best d value found:", d)
                print("P value for best d:", p_value)
                print("Correlation to original:", corr)
                print("====================\n")
            
            return (round(d, 2), adf_results[4]["5%"])
    
    return None, None

def get_weights_FFD(d, threshold):
    """
    Generates weights for a fixed width fractional differentiation
    
    Params
    ---------
    d           : float
        Chosen `d` value, to be computed
    threshold   : float
        Threshold for dropping weights by modulus

    Returns
    ---------
    np.array
        Array of generated weights
    """
    # threshold>0 drops insignificant weights
    w, k = [1.], 1
    
    while True:
        w_ = -w[-1] / k*(d-k+1)
        
        if abs(w_) < threshold:
            break
        
        w.append(w_)
        k+=1
    
    w = np.array(w[::-1]).reshape(-1,1)
    
    return w

def frac_diff_FFD(series, d, threshold=1e-5):
    """
    Fractional differentiation of a series using a fixed window. This attempts 
    to create stationarity without losing data memory
    
    Note 1: threshold determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    
    Params
    ---------
    series      : pd.DataFrame
        Series to fractionally differentiate
    d           : float
        Chosen `d` value, to be computed
    threshold   : float
        Threshold for dropping weights by modulus

    Returns
    ---------
    pd.DataFrame
        The series differentiated
    """
    #1) Compute weights for the longest series
    w = get_weights_FFD(d, threshold)
    width = len(w) - 1
    
    #2) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        
        for iloc1 in range(width,seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            
            # exclude NAs
            if not np.isfinite(series.loc[loc1,name]):
                continue
            
            df_.at[loc1] = np.dot(w.T,seriesF.loc[loc0:loc1].values)[0,0]
        
        df[name] = df_.copy(deep=True)
    
    df = pd.concat(df,axis=1)
    return df
