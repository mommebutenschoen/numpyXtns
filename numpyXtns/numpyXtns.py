from __future__ import print_function
try:
    from itertools import izip as zip
except ImportError:
    pass
from numpy import array, round, where, ones, convolve, \
    hanning, hamming, bartlett, blackman, r_, median, sqrt, floor, resize, \
    empty, sort, apply_along_axis, int_, mean, unravel_index, log10, float_, NaN
from gzip import open as opengz
from numpy.ma import MaskedArray, getmaskarray, nomask,masked_where
from datetime import datetime, timedelta
from numpy.ma import array as marray
try:
    xrange(1)  # python2
except NameError:
    xrange = range  # python3
import logging

def quantiles(data, prob=[.25, .5, .75], alphap=.4, betap=.4, axis=None):
    """Computes empirical quantiles for a *1xN* data array.
    Samples quantile are defined by:
    *Q(p) = (1-g).x[i] +g.x[i+1]*
    where *x[j]* is the jth order statistic,
    with *i = (floor(n*p+m))*, *m=alpha+p*(1-alpha-beta)* and *g = n*p + m - i)*.

    Typical values of (alpha,beta) are:

    - (0,1)    : *p(k) = k/n* : linear interpolation of cdf (R, type 4)
    - (.5,.5)  : *p(k) = (k+1/2.)/n* : piecewise linear function (R, type 5)
    - (0,0)    : *p(k) = k/(n+1)* : (R type 6)
    - (1,1)    : *p(k) = (k-1)/(n-1)*. In this case, p(k) = mode[F(x[k])].
      That's R default (R type 7)
    - (1/3,1/3): *p(k) = (k-1/3)/(n+1/3)*. Then p(k) ~ median[F(x[k])].
      The resulting quantile estimates are approximately median-unbiased
      regardless of the distribution of x. (R type 8)
    - (3/8,3/8): *p(k) = (k-3/8)/(n+1/4)*. Blom.
      The resulting quantile estimates are approximately unbiased
      if x is normally distributed (R type 9)
    - (.4,.4)  : approximately quantile unbiased (Cunnane)
    - (.35,.35): APL, used with PWM

    Args:
        x (sequence): Input data, as sequence or array of maximum 2 dimensions.
        prob (sequence): List of quantiles to compute.
        alpha (float): Plotting positions parameter.
        beta (float): Plotting positions parameter.
        axis (integer): axis along which to compute quantiles.
            If `None`, uses the whole (flattened/compressed) dataset.

    Returns:
        Array with quantiles at the probabilities requested, returns NaN for points masked in all samples.
    """
    def _quantiles1D(data, m, p):
        x = sort(data.compressed())
        n = len(x)
        if n == 0:
            return marray(empty(len(p), dtype=float_), mask=True)
        elif n == 1:
            return marray(resize(x, p.shape), mask=nomask)
        aleph = (n*p + m)
        k = floor(aleph.clip(1, n-1)).astype(int_)
        gamma = (aleph-k).clip(0, 1)
        return (1.-gamma)*x[(k-1).tolist()] + gamma*x[k.tolist()]

    # Initialization & checks ---------
    data = marray(data, copy=False)
    p = array(prob, copy=False, ndmin=1)
    m = alphap + p*(1.-alphap-betap)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        return _quantiles1D(data, m, p)
    else:
        assert data.ndim <= 2, "Array should be 2D at most !"
        return apply_along_axis(_quantiles1D, axis, data, m, p)

try:
    from scipy.stats.mstats import mquantiles
except:
    mquantiles = quantiles

def wquantiles(data,weights,prob=[0.5,],axis=None):
    """
    Computes weighted quantiles as the intersect of the probability level with
    the corresponding cumulative distribution function, where the bin height
    of each sample on the probability axis is given by its relative weight
    (i.e. the weights are normalised to add up to 1).

    Args:
        data (array of sortable type): input dataset
        weights (array of floats): weight for each sample, same dimension
            as input dataset
        prob (sequence of floats): quantiles to be computed (defaults to median)
    Returns:
        Array with quantile data for each probability requested in the first
        dimension.
    This function does not deal with mask, if you have masked data,
    retrieve the pure data and set the weight factors to 0 for the masked
    elements.
    """
    if axis == None:
        # Process all data as flat array computing the oeverall quantiles
        a = array(data).ravel()
        w = array(weights).ravel()
    else:
        # Reorder, so that the dimension to be processed is last
        transorder=list(range(array(data).ndim))
        transorder.append(transorder.pop(axis))
        a = array(data).transpose(transorder)
        w = array(weights).transpose(transorder)
    # Preallocate return list:
    nprobs=len(prob)
    wquants=empty((nprobs,)+a.shape[:-1])
    #Process by column
    for n,(a_col,w_col) in enumerate(
        zip(a.reshape([-1,a.shape[-1]]),w.reshape([-1,w.shape[-1]])) ):
        if any(w_col==0.):
            a_col=masked_where(w_col==0.,a_col).compressed()
            w_col=masked_where(w_col==0.,w_col).compressed()
        if len(a_col)>1:
            idx=a_col.argsort()
            a_col.sort()
            w_cum=w_col[idx].cumsum()/w_col.sum()
            for np,p in enumerate(prob):
                k_prob=where(w_cum<p,1,0).sum()
                wquants.reshape([nprobs,-1])[np,n]=a_col[k_prob]
        elif len(a_col)==1:
            wquants.reshape([nprobs,-1])[:,n]=a_col[0]
        else:
            wquants.reshape([nprobs,-1])[:,n]=NaN
    return wquants

def point_inside_polygon(x, y, poly):
    """Check if point (x,y) is in polygon poly.
    Polygon poly is of shape N,2.

    Args:
        x (float): x-coordinate of point to check.
        y (float): y-coordinate of point to check.
        poly (seqeunce of float tuples): polygon.

    Returns:
        `True` if inside, `False` if not.
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def polygonArea(corners):
    """
    Compute polygon area.

    Args:
        corners (seqeunce of float tuples): polygon corner points.

    Returns:
        Polygon area (float).
    """
    segments = zip(corners, list(corners)[1:] + [list(corners[0])])
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments))
# def EarthPolygonArea(corners):
#     acorners=array(corners)
#  #latmin,latmax,latmean,lonmean=acorners[:,1].min(),acorners[:,1].max(),acorners[:,1].mean(),acorners[:,0].mean()
#     latmean=acorners[:,1].mean()
#     p=Proj("+proj=cea +lat_ts=latmean")
#     xy=[p(*c) for c in corners]
#     return polygonArea(xy)


def spread(a):
    """Print  basic statistical information on data.

    Args:
        a (float or int array): data array.
    """
    print("\tMaximum: ", a.max(), " Minimum ", a.min())
    print("\tMean: ", a.mean(), " Standard Deviation: ", a.std())
    try:
        from scipy.stats.mstats import mquantiles
    except:
        mquantiles = quantiles
    q = mquantiles(a.ravel(), prob=[.01, .25, .5, .75, .99])
    print("\tQuantiles: %f (1%%) %f (25%%) %f (50%%) %f (75%%) %f (99%%)"
          % tuple(q))


def iqr(a):
    """Compute inter-quartile range of data.

    Args:
        a (float or int array): data array.

    Returns:
       inter-quartile range (float).
    """
    q, r = mquantiles(a, prob=[.25, .75])
    return r-q


def arrayMin(a):
    """
    Finds position of data array minimum.

    Args:
        a (float): data array.

    Returns:
        Integer tuple with minimum position.
    """
    return unravel_index(a.argmin(), a.shape)


def arrayMax(a):
    """
    Finds position of data array minimum.

    Args:
        a (float): data array.

    Returns:
        Integer tuple with maximum position.
    """
    return unravel_index(a.argmax(), a.shape)


def loadASCIIlist(fname, separator=None, fillValue=None, dtype=None):
    """
    Reads an ascii data file into a nested list with outer elements being
    lines, inner elements separated by an arbitrary separating character.
    The default separator `None` corresponds to any whitespace character.
    `fillValue` (if defined) replaces empty elements ('').

    Args:
       fname (str): input file path.
       separator (str): str used as element separator in input file.
       fillValue (): object that will be used as fill value for empty elements.
       dtype (function): conversion function to be applied on input element
                         strings.

    Returns:
        Nested list (1 level) with data from file.

    """
    if fname[-3:] == '.gz':
        fid = opengz(fname, 'r')
    else:
        fid = open(fname, 'r')
    dlist = fid.readlines()
    fid.close()
    data = []
    for el in dlist:
        data.append(el.rstrip('\r\n').split(separator))
    if fillValue:
        logging.info("filling empty entries...")
        data = [[(el.strip() == '' and [fillValue] or [el])[0]
                 for el in elem] for elem in data]
    if dtype:
        data = [[dtype(el) for el in row] for row in data]
    return data


class deflatableArray(MaskedArray):
    """
    Class of deflatable arrays, i.e. masked arrays that can be deflated
    (removing shape and masked elements), manipulated and reinflated to
    original shape.

    Attributes:
        idx (integer list tuple): tuple with integer lists of non-masked
                                  elements.
        fullShape (integer tuple): shape of inflated array.
        fillval (float, integer): fill value of masked array.
        compressedShape (integer): size of deflated array.
        deflated: array data in compressed shape.
    """
    def __init__(self, data, *args, **opts):
        """
        Initialises the deflatable array as masked array.

        Inherits:
            MaskedArray

        Args:
            data (array): data array to be deflated.
            *args, **opts: positional and optional arguments passed to
                `MaskedArray` initialisation function.
        """
        MaskedArray.__init__(data, *args, **opts)
        self.idx = (-getmaskarray(self)).nonzero()
        self.fullShape = self.shape
        self.fillval = self.fill_value
        self.compressedShape = len(self.idx[0])

    def deflate(self):
        """
        Deflates array removing shape and masked elements.
        """
        if self.fullShape != self.compressedShape:
            self.deflated = self.compressed()
        else:
            self.deflated = self.flatten()

    def inflate(self):
        """
        Inflates array by reinserting masked elements and
        reshaping into original structure.
        """
        if self.fullShape != self.compressedShape:
            self[self.idx] = self.deflated
        else:
            self[:] = self.deflated.reshape(self.fullShape)


def daysPerMonth(y, m):
    """
    Computes days of a month  and its center as datetime object.

    Args:
        y (int): Year.
        m (int): Month.

    Returns:
        Tuple of days per month (int) and centre of the month (datetime).
    """
    yp1 = m == 12 and y+1 or y
    mp1 = m == 12 and 1 or m+1
    dpm = (datetime(yp1, mp1, 1)-datetime(y, m, 1))
    md = datetime(y, m, 1)+timedelta(dpm.days/2., dpm.seconds/2.)
    return dpm.days, md


def secsSince(date, refdate):
    """
    Computes the seconds passed since a reference date.

    Args:
        date (datetime): Date.
        refdate (datetime) Reference date.
    Returns:
        Seconds passed (integer).
    """
    dt = date-refdate
    return dt.days*86400+dt.seconds


def midDate(do1, do2):
    """
    Compute the centre of two date time object.

    Args:
        do1 (datetime): First date.
        do2 (datetime): Second date.

    Returns:
        `datetime` object in the middle of the two input dates.
    """
    dt = do2-do1
    dt = timedelta(.5*dt.days, .5*dt.seconds)
    return do1+dt


def midMonth(y, m):
    """
    Computes centre of a month as datetime object.

    Args:
        y (int): Year.
        m (int): Month.
    Returns:
        `datetime` object in the middle of the month.
    """
    mp = m+1
    yp = mp > 12 and y+1 or y
    mp = mp > 12 and 1 or mp
    return midDate(datetime(y, m, 1), datetime(yp, mp, 1))


def midYear(y):
    """
    Computes centre of a year as datetime object.

    Args:
        y (int): Year.
    Returns:
        `datetime` object in the middle of the year.
    """
    return midDate(datetime(y, 1, 1), datetime(y+1, 1, 1))


def smooth(x, window_len=11, window='hanning', extrapolate='rotation'):
    """Smooth input data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Args:
        x (array): the input signal
        window_len (int): the dimension of the smoothing window;
                          should be an odd integer
        window (str): the type of window from 'flat', 'hanning', 'hamming',
                      'bartlett', 'blackman';
                      flat window will produce a moving average smoothing.

    Returns:
        The smoothed signal.

    Example::

      t=linspace(-2,2,0.1)
      x=sin(t)+randn(len(t))*0.1
      y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead
          of a string
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman',
                      'robust']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming','bartlett', 'blackman','robust'")

    # extrapolation by reflection of dat at end point
    wlp1 = window_len+1
    # get stopping index for backward iteration in extrapolation
    if wlp1 > x.size:
        mwlp1 = None
    else:
        mwlp1 = -wlp1
    if extrapolate == 'axially':
        s = r_[x[window_len-1:0:-1], x, x[-2:mwlp1:-1]]
    # extrapolation by circular wrapping
    elif extrapolate == 'periodically':
        s = r_[x[-window_len+1:], x, x[:window_len-1]]
    # extrapolation by rotation of data at end point
    elif extrapolate == 'rotation':
        s = r_[2*x[0]-x[window_len-1:0:-1], x, 2*x[-1]-x[-2:mwlp1:-1]]
    if window == 'flat':  # moving average
        w = ones(window_len, 'd')
    elif window == 'robust':
        pass
    else:
        w = eval(window+'(window_len)')
    if window == 'robust':
        y = s.copy()
        for n in xrange(window_len-1, len(y)-window_len):
                y[n] = median(s[n-window_len/2:n+window_len/2+1])
    else:
        y = convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def RMSE(d1, d2, ax=None):
    """
    Compute root-mean-square difference.

    Args:
        d1 (array): first data set.
        d2 (array): second data set (should be of same length and shape).

    Returns:
        Root-mean-square difference of the two input arrays (float).
    """
    if ax is None:
        return sqrt(((d1-d2)**2).sum()/d1.ravel().shape[0])
    else:
        return sqrt(((d1-d2)**2).sum(ax)/d1.shape[ax])
