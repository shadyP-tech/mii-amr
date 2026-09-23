"""Batch the five current-pixel rail profiles without changing merge gates."""
import numpy as np


def _sample(frame, positions, *, allow_partial=False):
    low = np.floor(positions).astype(int)
    x, y = low[..., 0], low[..., 1]
    valid = np.all((x >= 0) & (y >= 0) & (x+1 < frame.shape[1])
                   & (y+1 < frame.shape[0]), axis=1)
    if not allow_partial and not valid.all():
        return None
    if allow_partial:
        x = np.clip(x, 0, frame.shape[1]-2)
        y = np.clip(y, 0, frame.shape[0]-2)
    fraction = positions-low
    fx, fy = fraction[..., 0, None], fraction[..., 1, None]
    values = ((1-fx)*(1-fy)*frame[y,x]+fx*(1-fy)*frame[y,x+1]
              +(1-fx)*fy*frame[y+1,x]+fx*fy*frame[y+1,x+1])
    return (values, valid) if allow_partial else values


def single_thin_stripe(frame, a, b, c, d):
    """Keep opposite-gradient/single-ridge proof, including multi-ridge veto."""
    first, last, other, end = (np.array((p.u_px, p.v_px), dtype=float) for p in (a,b,c,d))
    tangent = last-first
    tangent /= max(float(np.linalg.norm(tangent)), 1.)
    normal = np.array((-tangent[1],tangent[0]))
    if max(abs(float((other-first)@normal)),abs(float((end-last)@normal))) > 6.:
        return False
    fraction = np.array((.2,.35,.5,.65,.8))[:,None]
    p, q = first+fraction*(last-first), other+fraction*(end-other)
    values, valid = _sample(frame,np.stack((p-.75*normal,p+.75*normal,q-.75*normal,q+.75*normal),axis=1),allow_partial=True)
    g1,g2=values[:,1]-values[:,0],values[:,3]-values[:,2]
    magnitude=np.linalg.norm(g1,axis=1)*np.linalg.norm(g2,axis=1)
    opposite=valid & (magnitude>=400.) & (np.sum(g1*g2,axis=1)<=-.8*magnitude)
    radius=np.abs((q-p)@normal)/2+2.5
    positions=(p+q)[:,None,:]/2 + radius[:,None,None]*np.linspace(-1,1,25)[None,:,None]*normal
    values=_sample(frame,positions)
    if values is None:
        return False
    channels=np.argmax(np.ptp(values,axis=1),axis=1)
    profiles=values[np.arange(5),:,channels]
    baseline=(profiles[:,0]+profiles[:,-1])/2
    eligible=(np.abs(profiles[:,0]-profiles[:,-1])<=15.) & (np.ptp(profiles,axis=1)>=25.)
    positive=profiles.max(axis=1)-baseline>=baseline-profiles.min(axis=1)
    excursion=(profiles-baseline[:,None])*np.where(positive,1.,-1.)[:,None]
    eligible &= excursion.min(axis=1)>=-15.
    active=excursion>=np.maximum(12.,.25*excursion.max(axis=1))[:,None]
    starts=np.sum(active & ~np.roll(active,1,axis=1),axis=1)
    multiple=np.count_nonzero(eligible & (starts>1))
    supported=np.count_nonzero(eligible & (starts==1) & ~active[:,0] & ~active[:,-1])
    return bool(multiple<2 and (supported>=4 or np.count_nonzero(opposite)>=4))
