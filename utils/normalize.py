import pandas as pd

def center_series(series):
    """
    Centers a series. Centering involves subtracting the series 
    mean from each entry in series, resulting in a new total mean 
    of 0.
    
    Parameters
    ------------
    series  : pd.Series
    
    Returns
    ----------
    tuple
        A tuple containing (1) the mean value used, and (2) the 
        resultant transformed series
    """
    mu = series.mean()
    series = series.apply(lambda x: x - mu)
    
    return (mu, series)

def scale_series(series):
    """
    Scales a series. Scaling involves dividing the series standard 
    deviation from each entry in series, resulting in a new 
    standard deviation of 1
    
    Parameters
    ------------
    series  : pd.Series
    
    Returns
    ----------
    tuple
        A tuple containing (1) the std deviation used, and (2) the 
        resultant transformed series
    """
    sigma = series.std()
    
    if sigma > 0:
        series = series.apply(lambda x: x / sigma)
    
    return (sigma, series)
