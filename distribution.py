# app.py
import math
from math import log, sqrt
from math import erf, lgamma
import numpy as np

Z09 = 1.2815515655446004
SOURCES_WITH_OTHERS = {"AppUsage_min","GalleryPOI_cnt","YouTube_min","WebTitle_cnt"}
SOURCES_NO_OTHERS   = {"CEExercise_cnt","NowBarSports_cnt"}

def safe_float(x):
    try: return float(x)
    except: return np.nan

def ln_mu_sigma_from_median_p90(median, p90):
    m = max(1e-12, float(median))
    p = max(m*1.0000001, float(p90))
    mu = log(m); sigma = (log(p) - log(m)) / Z09
    sigma = max(sigma, 1e-6)
    return mu, sigma

def cdf_lognorm(x, median, p90):
    x = float(x)
    if x <= 0: return 0.0
    mu, sigma = ln_mu_sigma_from_median_p90(median, p90)
    z = (log(x) - mu) / (sigma*sqrt(2.0))
    return 0.5*(1.0 + erf(z))

def pdf_lognorm(x, median, p90):
    x = float(x)
    if x <= 0: return 0.0
    mu, sigma = ln_mu_sigma_from_median_p90(median, p90)
    return (1.0/(x*sigma*sqrt(2.0*math.pi))) * math.exp(-0.5*((log(x)-mu)/sigma)**2)

def poisson_cdf(k, lam):
    if k < 0: return 0.0
    k = int(math.floor(k))
    term = math.exp(-lam)
    s = term
    for i in range(1, k+1):
        term *= lam / i
        s += term
    return s

def nb_params_from_mean_var(mean, var):
    mean = max(0.0, float(mean))
    var  = max(mean + 1e-9, float(var))
    if var <= mean:  # 근사 안정화
        var = mean + 1e-6
    r = (mean**2) / (var - mean)
    p = r / (r + mean)  # "성공확률"
    r = max(r, 1e-8); p = min(max(p, 1e-8), 1-1e-8)
    return r, p

def nbinom_cdf(k, r, p):
    # pmf(0) = p^r, pmf(n+1) = pmf(n) * (1-p)*(r+n)/(n+1)
    if k < 0: return 0.0
    k = int(math.floor(k))
    pmf = p**r
    s = pmf
    for n in range(0, k):
        pmf *= (1-p) * (r+n) / (n+1)
        s += pmf
    return s

def zip_cdf(x, mean, p0):
    p0 = min(max(float(p0), 0.0), 1.0)
    mean = max(0.0, float(mean))
    if p0 >= 1.0: return 1.0 if x >= 0 else 0.0
    lam = mean / max(1e-12, (1.0 - p0))
    return p0 + (1.0 - p0) * poisson_cdf(x, lam)

def zinb_cdf(x, mean, var, p0):
    p0 = min(max(float(p0), 0.0), 1.0)
    if p0 >= 1.0: return 1.0 if x >= 0 else 0.0
    r, p = nb_params_from_mean_var(mean, var)  # NB 구성요소 기준
    return p0 + (1.0 - p0) * nbinom_cdf(x, r, p)

def zip_pmf_grid(xs, mean, p0):
    p0 = min(max(float(p0), 0.0), 1.0)
    if p0 >= 1.0:
        return np.array([1.0 if x==0 else 0.0 for x in xs])
    lam = mean / max(1e-12, (1.0 - p0))
    pmf = []
    for k in xs:
        if k==0:
            pm = p0 + (1.0 - p0) * math.exp(-lam)
        else:
            pm = (1.0 - p0) * (math.exp(-lam) * lam**k / math.factorial(k))
        pmf.append(pm)
    return np.array(pmf)

def zinb_pmf_grid(xs, mean, var, p0):
    r, p = nb_params_from_mean_var(mean, var)
    pmf = []
    for k in xs:
        if k==0:
            # p0 + (1-p0)*NB(0)
            pm = p0 + (1.0 - p0) * (p**r)
        else:
            # NB pmf: C(k+r-1,k) * (1-p)^k * p^r
            logC = lgamma(k+r) - lgamma(k+1) - lgamma(r)
            nb   = math.exp(logC) * ((1-p)**k) * (p**r)
            pm   = (1.0 - p0) * nb
        pmf.append(pm)
    return np.array(pmf)