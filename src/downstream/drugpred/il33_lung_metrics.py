"""Population-level response metrics; no correspondence between control and target cells."""
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu
from sklearn.metrics import r2_score


def correlation(x,y):
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if np.std(x)<1e-12 or np.std(y)<1e-12:
        return None
    return float(np.corrcoef(x,y)[0,1])


def r2(truth,prediction):
    return float(r2_score(truth,prediction)) if np.std(truth)>1e-12 else None


def effect_order(values):
    """Rank nonzero responses by absolute mean-log-expression change, without p-values."""
    values = np.asarray(values)
    active = np.flatnonzero(np.abs(values) >= 1e-8)
    return active[np.argsort(-np.abs(values[active]), kind='stable')]


def effect_rank_metrics(observed, predicted, top_n=(20, 50, 100, 200)):
    true_order, predicted_order = effect_order(observed), effect_order(predicted)
    result = {}
    for n in top_n:
        idx = true_order[:n]
        result[f'effect_rank_overlap_{n}'] = len(set(idx) & set(predicted_order[:n])) / len(idx) if len(idx) else None
        result[f'effect_rank_delta_RMSE_{n}'] = float(np.sqrt(np.mean((predicted[idx] - observed[idx]) ** 2))) if len(idx) else None
    return result


def differential(target,control,eligible):
    """Nominal cell-level rank test, descriptive rather than animal-level inference."""
    cols = np.flatnonzero(eligible)
    p = np.ones(len(eligible))
    for left in range(0,len(cols),256):
        idx = cols[left:left+256]
        p[idx] = mannwhitneyu(target[:,idx],control[:,idx],axis=0,
                             alternative='two-sided',method='asymptotic').pvalue
    q = np.ones_like(p)
    q[cols] = false_discovery_control(p[cols])
    delta = target.mean(0,dtype=np.float64)-control.mean(0,dtype=np.float64)
    # Numerical zero must not create a ranked response or a significant candidate.
    q[np.abs(delta)<1e-8] = 1
    selected = np.flatnonzero(eligible & (q<.05))
    order = selected[np.lexsort((selected,-np.abs(delta[selected]),p[selected]))]
    return pd.DataFrame(dict(delta=delta,pvalue=p,qvalue=q,eligible=eligible)),order


def evaluate(prediction,truth,control,true_order,predicted_order,top_n=(20,50,100,200)):
    pm = prediction.mean(0,dtype=np.float64)
    tm = truth.mean(0,dtype=np.float64)
    cm = control.mean(0,dtype=np.float64)
    pv = prediction.var(0,dtype=np.float64)
    tv = truth.var(0,dtype=np.float64)
    dp,dt = pm-cm,tm-cm
    response = np.sign(np.where(np.abs(dp)<1e-8,0,dp))
    result = dict(Pearson=correlation(pm,tm),Pearson_delta=correlation(dp,dt),
                  R2_mean_all=r2(tm,pm),R2_var_all=r2(tv,pv),R2_delta_all=r2(dt,dp),
                  delta_RMSE=float(np.sqrt(np.mean((dp-dt)**2))),
                  delta_MAE=float(np.mean(np.abs(dp-dt))),
                  variance_ratio=float(pv.sum()/tv.sum()) if tv.sum()>0 else None,
                  negative_fraction=float(np.mean(prediction<0)),predicted_DEGs=len(predicted_order))
    for n in top_n:
        idx = true_order[:n]
        hits = len(set(idx)&set(predicted_order[:n]))
        result.update({f'GT_DEG_n{n}':len(idx),f'DEG_overlap_count_{n}':hits,
                       f'DEG_overlap_{n}':hits/len(idx) if len(idx) else None,
                       f'DEG_direction_{n}':float(np.mean(response[idx]==np.sign(dt[idx]))) if len(idx) else None,
                       f'R2_mean_DEG{n}':r2(tm[idx],pm[idx]) if len(idx)>1 else None,
                       f'Pearson_mean_DEG{n}':correlation(pm[idx],tm[idx]) if len(idx)>1 else None,
                       f'R2_delta_DEG{n}':r2(dt[idx],dp[idx]) if len(idx)>1 else None,
                       f'R2_var_DEG{n}':r2(tv[idx],pv[idx]) if len(idx)>1 else None,
                       f'Pearson_delta_DEG{n}':correlation(dp[idx],dt[idx]) if len(idx)>1 else None})
    return result,dict(predicted_mean=pm,observed_mean=tm,control_mean=cm,
                       predicted_var=pv,observed_var=tv,predicted_delta=dp,observed_delta=dt)
