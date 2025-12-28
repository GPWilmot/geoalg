#!/usr/bin/env python
################################################################################
## File: calcLib.py is the basis for calcR.py and is part of GeoAlg.
## Copyright (c) 2021, 2023 G.P.Wilmot
##
## GeoAlg is free software: you can redistribute it and/or modify it under the
## terms of the GNU General Public License as published by the Free Software 
## Foundation, either version 3 of the License, or any later version.
##
## GeoAlg is distributed in the hope that it will be useful, but WITHOUT ANY
## WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
## FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along with
## GeoAlg. If not, see <https://www.gnu.org/licenses/>.
## -----------------------------------------------------------------------------
## CalcR is a commnd line calculator that converts basis numbers into classes.
## It only supports floats but can be extended to Quaternions, etc.
##
## Default case is just the real numbers but calcQ.py defines Quaternions.
## For example 1+i+j+k -> Quat(1,1,1,1). It then runs exec() or eval().
## Assumes calc?.py is in the same directory and this has more documentation.
## Start with either calcR.py, python calcR.py or see ./calcR.py -h.
## This module contains classes (run help for more info):
##   * Lib     - miscellaneous functions for all calculators
##   * Tensor - simple 1&2-D matricies used for testing and algebra elements
##   * Matrix - interface to numpy if it exists otherwise Tensor is used
##   * Euler  - extended Euler angles (calcQ uses 3 angles, calcCA uses more)
################################################################################
__version__ = "0.8"
import math, sys, os
import time, datetime
try:
  np = True
  for arg in sys.argv:
    if arg == "--skipNumpy":
      np = None
    elif arg[:1] == "-" and arg[1:2] != "-" and "s" in arg[1:]:
      np = None
  if np:
    import numpy as np
except:
  np = None

################################################################################
class Lib():
  """Class to provide common resources for basis numbers. Tests are a list of
     strings that can be run in the current calculator with results logged."""
  _EARTH_MAJOR_M = "WGS-84 earth ellipse equatorial radius(m) variable"
  EARTH_MAJOR_M  = 6378137
  _EARTH_MINOR_M = "WGS-84 earth ellipse polar radius(m) variable"
  EARTH_MINOR_M  = 6356752
  _EARTH_ECCENT  = "WGS-84 earth ellipse eccentricity variable"
  EARTH_ECCENT   = 0.08181979                     # sqrt(1-minor2/major2)
  _EARTH_ECCENT2  = EARTH_ECCENT *EARTH_ECCENT    # eccentricity squared
  _EARTH_ECCENT1  = math.sqrt(1 -_EARTH_ECCENT2)  # minor/major ratio
  _PI            = "Pi variable from math"
  PI             = math.pi
  _D2R           = "Multiplier for degrees to radians"
  D2R            = math.pi /180.0
  _E             = "Euler's number from math"
  E              = math.e
  __precision   = 1E-15         # Precision used for equality
  __resolution  = 9             # Digits for float display
  __resol_form  = r"%0.9G"      # Format string for float display
  __resol_float = r"%0.9f"      # Format string for float display
  __verbose     = False         # Traceback logging
  __info        = False         # User info logging
  __plotFigure  = 0             # Matplotlib unique figures count
  __lastTime    = 0             # Store epoch for time
  __lastProcTime= 0             # Store program time float seconds
  __checkMemSrt = True          # checkMem to run procTime at start
  __storeName   = []            # Store and check loaded filenames
  __storeCalc   = []            # Store and check loaded calculators
  _memLimitMB   = 500           # Abort if less mem than this
  if sys.version_info.major == 2:
    _basestr = basestring       # Handle unicode
  else:
    _basestr = str

  # Overload globals and maths functions (abs & pow handled within classes)
  @staticmethod
  def exp(arg):
    if isinstance(arg, (float,int)):
      return math.exp(arg)
    return arg.exp()

  @staticmethod
  def log(arg):
    if isinstance(arg, (float,int)):
      return math.log(arg)
    return arg.log()

  @staticmethod
  def version():
    """version()
       Return the module version string."""
    return __version__

  @staticmethod
  def sincos(a):
    """sincos(rad)
       Return sin(rad) & cos(rad)."""
    Lib._checkType(a, (int, float), "sincos")
    return Lib._sincos(a)

  @staticmethod
  def readText(filename):
    """readTest(filename)
       Local read_text(filename) to return contents of a text file."""
    try:
      with open(filename) as fp:
        return fp.read()
    except Exception:
      typ,var,tb = sys.exc_info()
      raise typ(str(var).replace("\\\\\\\\", "\\"))

  @staticmethod
  def nextFigure():
    """nextFigure()
       Return next Matplotlib unique figure count."""
    Lib.__PlotFigures += 1
    return Lib.__PlotFigure 

  @staticmethod
  def getMainDoco(filt="test"): #TBD
    """getMainDoco([filt])
       For tests defined as functions return the doco strings."""
    lines = ""
    for name in dir(sys.modules["__main__"]):
      if name[:len(filt)] == filt and len(filt) > len(filt):
        docs = getattr(sys.modules["__main__"], name).__doc__
        for doc in docs.splitlines():
          lines += doc.strip() +'\n'
    return lines

  @staticmethod
  def info(info=None):
    """info([info])
       Toggle or set info for user reporting."""
    if info is None:
      info = not Lib.__info
    Lib._checkType(info, bool, "info")
    Lib.__info = info
    return verbosity
  @staticmethod
  def verbose(verbosity=None):
    """verbose([verbosity])
       Toggle or set verbosity for logging and traceback reporting."""
    if verbosity is None:
      verbosity = not Lib.__verbose
    Lib._checkType(verbosity, bool, "verbose")
    Lib.__verbose = verbosity
    return ("on" if verbosity else "off")

  @staticmethod
  def resolution(digits=None, get=False):
    """resolution([digits,get])
       Set or reset to 18 &/or return format digits. Min. is 4 & default 9."""
    if get:
      if digits is not None:
        raise Exception("Can't set digits when get set")
    else:
      if digits is not None:
        Lib._checkType(digits, int, "resolution", (0,0))
      if digits is None:
        digits = 18
      Lib._checkType(digits, int, "resolution")
      if digits < 4:
        digits = 4
      Lib.__resolution = digits
      Lib.__resol_form = "%%0.%dG" %digits
      Lib.__resol_float = "%%0.%df" %digits
    return Lib.__resolution

  @staticmethod
  def precision(precise=None, get=False):
    """precision([precise,get])
       Set or reset to default e-15 &/or return equality precision."""
    if get:
      if precise is not None:
        raise Exception("Can't set precision when get set")
    else:
      if precise is None:
        precise = 1E-15
      Lib._checkType(precise, float, "precision")
      if precise <= 0.0 or precise > 0.9:
        raise Exception("Precision must be positive and smaller that 0.9")
      Lib.__precision = precise
    return Lib.__precision

  @staticmethod
  def isInfo():
    """isInfo()
       Return true if verbosity is set."""
    return Lib.__info
  @staticmethod
  def isVerbose():
    """isVerbose()
       Return true if verbosity is set."""
    return Lib.__verbose
  _isVerbose = isVerbose
  @staticmethod
  def getResolNum(val):
    """getResolNum(val)
       Return int or the float rounded to resolution or in exponent format."""
    num = str(val)
    pos = num.find('.')
    if pos < 0 or len(num) -pos -1 < Lib.__resolution:
      return num
    else:
      return Lib.__resol_float %val

  @staticmethod
  def _checkType(arg, typ, method, size=[]):
    """Raise exception if arg not the correct type in method for size."""
    if typ == int:
      if isinstance(arg, bool):   # Filter out int for bool or bool in list
        tmp = str(arg)
        if len(tmp) > 9:
          tmp = str(type(arg))
        raise Exception("Invalid parameter type (%s) for %s" %(tmp, method))
    if not isinstance(arg, typ):
      tmp = str(arg)
      if len(tmp) > 9:
        tmp = str(type(arg))
      raise Exception("Invalid parameter type (%s) for %s" %(tmp, method))
    if size:
      Lib._checkSize(size, arg, method, "value")
  @staticmethod
  def _checkList(arg, typ, method, size=[]):
    """Raise exception if not a list/tuple of typ types and optional size as 
       int or range with (x,0) meaning length at least x."""
    Lib._checkType(arg, (list, tuple, Matrix), method)
    Lib._checkType(size, (int, list, tuple, Matrix), method)
    if size:
      if isinstance(arg, (Tensor, Matrix)):
        Lib._checkSize(size, arg.shape[0], method, "row length")
      else:
        Lib._checkSize(size, len(arg), method, "list length")
    for elem in arg:
      if typ is not None:
        Lib._checkType(elem, typ, method)
  @staticmethod
  def _checkSize(size, val, method, src):
    """Raise exception if val != int size or in (x,y), y=0=infinity."""
    if isinstance(size, int):
      if size != val:
        raise Exception("Invalid %s !=%d for %s" %(src, size, method))
    elif isinstance(size, (list, tuple)):
      if len(size) == 2 and (val < size[0] or \
            (size[1] > 0 and val > size[1])):
        raise Exception("Invalid %s !in [%d,%s] for %s" %(src, size[0],
                         size[1] if size[1] else "..", method))
    elif size:
      raise Exception("Invalid check %s parameters in %s" %(src, method))

  @staticmethod
  def _getResolutions():
    """Internal method to return the digits and print format."""
    return Lib.__resolution, Lib.__resol_form, Lib.__resol_float
  @staticmethod
  def _getPrecision():
    """Internal method to return the precision."""
    return Lib.__precision
  @staticmethod
  def _resolutionDump(sign, val, basis):
    """Internal method to return a formated number and basis or blank."""
    out = ""
    num = "%s" %val
    pos = num.find('.')
    if num[0] == '-' and sign:
      sign = " "
    if pos < 0 or len(num) -pos -1 < Lib.__resolution:
      if abs(val) == 1 and basis:
        num = "" if val > 0 else "-"
      if val != 0:
        out = "%s%s%s" %(sign, num, basis)
    elif val != 0.0:
      flt = Lib.__resol_form %val
      if flt.find(".") < 0:
        flt = Lib.__resol_float %val
      resolForm = r"%%s%s%%s" %Lib.__resol_form
      out = sign +flt +basis
    return out
  @staticmethod
  def _storeName(name):
    """Internal method to store a named file."""
    if name not in Lib.__storeName:
      Lib.__storeName.append(name)
  @staticmethod
  def __checkNames(names):
    """Internal method to raise an exception if any of the named files
       are not loaded."""
    out = []
    for name in names.split(','):
      name = name.strip()
      if name not in Lib.__storeName:
        out.append(name)
    return out
  @staticmethod
  def _checkNames(names=None):
    Lib._checkType(names, Lib._basestr, "load")
    out = Lib.__checkNames(names)
    if out:
      raise Exception("Need to load %s" %",".join(out))
  @staticmethod
  def _storeCalc(name):
    """Internal method to store a loaded calculator for isCalc()."""
    if name not in Lib.__storeCalc:
      Lib.__storeCalc.append(name)
  @staticmethod
  def __isCalc(names):
    """Internal method to return true is the named calculators are loaded."""
    for name in names.split(','):
      name = name.strip()
      if name not in Lib.__storeCalc:
        return False
    return True
  @staticmethod
  def _getCalcList():
    """Internal reuse."""
    return Lib.__storeCalc
  @staticmethod
  def _float(val):
    """Internal late binding for calcR.parsing /int for Python V2."""
    if np and isinstance(val, np.integer):
      return float(val)
    if isinstance(val, int):
      return float(val)
    return val

  @staticmethod
  def _piRange(a):
    """Return an euler angle to range -pi..+pi."""
    a = a % (math.pi *2)
    if a > math.pi:
      a -= math.pi *2
    return a
  @staticmethod
  def _sincos(a):
    """Internal conversion utility. Doesn't check type."""
    sina = math.sin(a)
    cosa = abs((1.0 -sina **2) **0.5)  #TBD
    signa = ((a -math.pi *-1.5) %(2.0 *math.pi)) -math.pi
    return sina, cosa if signa > -1.0 else -cosa
  @staticmethod
  def _mergeBasis(arr1, arr2):
    """Internal utility to merge 2 str basis lists."""
    if arr1 and arr2:
      out = []
      for ii in arr1:
        for jj in arr2:
          if jj[0] == "-":
            if ii[0] == "-":
              out.append(ii[1:] +jj[1:])
            else:
              out.append("-" +ii +jj[1:])
          else:
            out.append(ii +jj)
    elif arr1:
      out = arr1[:]
    else:
      out = arr2[:]
    return out
  @staticmethod
  def _unzipBasis(*pairs):
    """First pair is source name and basis size. Return others as an array
       of unzipped pairs with second of pair being zero if pair is single."""
    args = []
    replace = 0 if pairs[0] else pairs[0]
    for val in pairs[1:]:
      if isinstance(val, (tuple, list)) and len(val) == 2:
        args.extend(val)
      else:
        args.extend((val, replace))
    if pairs[0]:
      Lib._checkList(pairs[0], None, "_unzipBasis")
      Lib._checkType(pairs[0][0], Lib._basestr, "_unzipBasis")
      Lib._checkType(pairs[0][1], int, pairs[0][0])
      for val in args:
        Lib._checkType(val, int, pairs[0][0])
        if val > pairs[0][1] or val < 0:
          raise Exception("Invalid %s parameter size" %pairs[0][0])
    return args
  @staticmethod
  def _unzipPairs(dim, pairs):
    """Return pairs or sets of pairs as a permutation list."""
    Lib._checkList(pairs, None, "_unzipPairs")
    flat = []
    for pair in pairs:
      if isinstance(pair, (list, tuple, Matrix)):
        if len(flat) %2:
          raise Exception("Pairs index must be even")
        flat.extend(pair)
      else:
        flat.append(pair)
    if len(flat) %2:
      raise Exception("Pairs length must be even")
    if flat and dim < max(flat):
      raise Exception("Pairs dim must be greater then indices")
    perm = list(x +1 for x in range(dim))
    for idx in range(0, len(flat), 2):
      idx0, idx1 = flat[idx] -1, flat[idx +1] -1
      tmp = perm[idx0]; perm[idx0] = perm[idx1]; perm[idx1] = tmp
    return perm

  @staticmethod
  def _morph(basisNames, value, pairs):
    """Internal utility to perform a single basis swap for each pair in a list
       of basis pair names."""
    Lib._checkList(pairs, None, "morph")
    if isinstance(basisNames, (list, tuple, Matrix)):
      Lib._checkList(basisNames, Lib._basestr, "morph")
      basisNames = "".join(basisNames)
    Lib._checkType(basisNames, Lib._basestr, "morph")
    if len(pairs) %2:
      raise Exception("Pairs in morph needs to be of even length")
    morphed = False
    out = {}
    for idx in range(len(pairs) //2):
      x,y = pairs[idx *2:idx *2 +2]
      Lib._checkType(x, Lib._basestr, "morph")
      Lib._checkType(y, Lib._basestr, "morph")
      if x in pairs[:idx] or y in pairs[:idx +1]:
        raise Exception("Pairs in morph need to be unique")
      if basisNames == x:
        out[y] = value
        morphed = True
    if not morphed:
      out[basisNames] = value
    return out

  @staticmethod
  def isLoaded(names=None):
    """isLoaded([names])
       Return true if comma separated file names are loaded."""
    if names is None:
      return Lib.__storeName
    Lib._checkType(names, Lib._basestr, "isLoaded")
    return Lib.__checkNames(names) == []
  @staticmethod
  def isCalc(names):
    """isCalc(names)
       Return true if comma separated calculator names are loaded."""
    Lib._checkType(names, Lib._basestr, "isCalc")
    return Lib.__isCalc(names)
  @staticmethod
  def freeMemMB():
    """free[MemMB]()
       Return the amount of free memory left."""
    if sys.platform == "win32":
      process = os.popen('systeminfo 2>nul |find "Available Phys"')
      result = process.read()
      process.close()
      return int(result.split()[3].replace(",",""))
    return os.sysconf('SC_AVPHYS_PAGES')//256
  free=freeMemMB
  @staticmethod
  def checkMem(inc=0, mod=1, extra=0, finish=False):
    """checkMem([inc,mod,extra,finish])
       Dump progess if !inc%mod and return freeMem<_memLimitMB."""
    Lib._checkType(inc, (int, float), "checkMem")
    Lib._checkType(mod, (int, float), "checkMem")
    Lib._checkType(finish, bool, "checkMem")
    if Lib.__checkMemSrt:
      Lib.__checkMemSrt = False
      Lib.procTime(True)
      if mod < 1:
        raise Exception("Invalid mod valid for checkMem")
    if finish:
      Lib._checkMemSrt = True
      mod = 1
    return Lib._checkMem(inc, mod, extra)
  @staticmethod
  def _checkMem(inc, mod, extra=0):
    """Internal version for checkMem. Assumes procTime and finish called."""
    if inc %mod == 0:
      sys.stdout.write("%s (%0.1fs) inc=%s extra=%s %dMB\n" %(Lib.date(True),
                     int(Lib._procTime()), inc, extra, Lib.freeMemMB()))
      if Lib.freeMemMB() < Lib._memLimitMB and mod > 0:
        sys.stdout.write("ABORT: Memory limit reached\n")
        return True
    return False

  @staticmethod
  def chain(*iterables):
    """chain(list)
       Same as itertools.chain(). Concatenate list of lists or generators."""
    for it in iterables:
      for element in it:
        yield element

  @staticmethod
  def date(noMs=False):
    """date([noMs=False])
       Return the datetime object for now with str() formated as date_time."""
    now = str(datetime.datetime.today())
    if noMs:
      idx = now.find(".")
      now = now[:idx]
    return now

  @staticmethod
  def time(epoch=False):
    """time([epoch])
       Return seconds since epoch or difference to previous call as a float."""
    lastTime = 0 if not epoch else Lib.__lastTime
    Lib.__lastTime = time.time()
    return Lib.__lastTime -lastTime

  @staticmethod
  def procTime(start=False):
    """procTime([start])
       Return program user+sys seconds since start or diff. to previous call."""
    if Lib.__lastProcTime == 0 or start:
      if sys.version_info.major == 2:
        Lib.__lastProcTime = time.time()
      else:
        Lib.__lastProcTime = time.process_time()
    return Lib._procTime()
  @staticmethod
  def _procTime():
    lastTime = Lib.__lastProcTime
    if sys.version_info.major == 2:
      Lib.__lastProcTime = time.time()
    else:
      Lib.__lastProcTime = time.process_time()
    return Lib.__lastProcTime -lastTime

  @staticmethod
  def pascalsTriangle(n, dump=False):
    """pascal[sTriangle](n, [dump])
       Return a list of the n-th row of Pascal's Triangle starting at 0."""
    Lib._checkType(n, int, "pascalTriangle")
    Lib._checkType(dump, bool, "pascalTriangle")
    if n < 0:
      raise Exception("Invalid parameter for comb(%s,%s)" %(n, r))
    out = []
    for r in range(n +1):
      out.append(Lib.comb(n, r))
    return out
  pascal = pascalsTriangle

  @staticmethod
  def combinations(n, r, basis=False, dump=False):
    """comb[inations]/choose/binom(n,r,[basis,dump])
       Return number of combinations of r in n or list generator if basis as
       boolean or as a list of length n of items to form the combinations."""
    Lib._checkType(n, (int,float), "comb")
    Lib._checkType(r, (int,float), "comb")
    Lib._checkType(basis, (bool, list, tuple, Matrix), "comb")
    Lib._checkType(dump, bool, "comb")
    if n < r or r < 0:
      raise Exception("Invalid parameter for comb(%s,%s)" %(n, r))
    if basis:
      if dump:
        dump = [r,0]
      if basis == True:
        return Lib.__perm(n, [1] *r, 1, dump)
      if len(basis) < n:
        raise Exception("Invalid basis length in comb")
      return Lib.__comb(n, r, basis, dump)
    return math.factorial(n) /math.factorial(n-r) /math.factorial(r) 
  comb = combinations
  choose = combinations
  binom = combinations

  @staticmethod
  def __comb(n, r, basis, dump):
    """Need a yield to substitute basis for the return in comb. Also for speed
       expands r > n/2 as inverse of first half which needs ordering fixed."""
    if r > n //2:
      rng = range(1, len(basis) +1)
      for elem in reversed(list(Lib.__perm(n, [1] *(n -r), 1, dump))):
        yield tuple(basis[idx -1] for idx in rng if idx not in elem) 
    else:
      for elem in Lib.__perm(n, [1] *r, 1, dump):
        yield tuple(basis[idx -1]  for idx in elem)
  _comb = __comb
  
  @staticmethod
  def permute(n, r=None, dump=False):
    """perm[ute](n, [r,dump])
       Return number of permutations of n terms or generator of r in n perms."""
    Lib._checkType(n, (int, float), "perm")
    Lib._checkType(dump, bool, "perm")
    if isinstance(r, int) and not isinstance(r, bool):
      if dump:
        return Lib.__perm(n, [1] *r, 0, [r,0])
      return Lib.__perm(n, [1] *r, 0, None)
    elif r is None:
      return math.factorial(n) 
    raise Exception("Invalid r parameter for perm")
  perm=permute

  @staticmethod
  def __perm(n, arr, offset, dump):
    """Return permutation or combination arrays. The arr needs to be set to a
       list of one. Set offset to 0 to get permutations or 1 for combinations
       instead. This is used for BasisArgs() with offset=1."""
    if offset < 0 or n < 0 or len(arr) > n:
      raise Exception("Invalid parameter for perm or comb: %s" %n)
    if len(arr) == 0:
      yield []
    else:
      for recuse in range(offset if offset else 1, n +1):
        if len(arr) > 1:
          for more in Lib.__perm(n, arr[1:], recuse if offset else 0, dump):
            arr = [recuse] +more
            dup = False
            for idx,elem in enumerate(arr):
              if elem in arr[idx+1:]:
                dup = True
                break
            if dump and len(arr) == dump[0] and arr[0] != dump[1]:
              dump[1] = arr[0]
              sys.stdout.write("%s %s %s\n" %(Lib.date(), dup, arr))
            if not dup:
              yield arr
        else:
          yield [recuse]

  @staticmethod
  def additionTree(dim, split, maxs=()):
    """additionTree(dim,split, [maxs])
       Split dim into split parts with options maxs list."""
    Lib._checkType(dim, int, "additionTree")
    Lib._checkType(split, int, "additionTree")
    Lib._checkList(maxs, None, "additionTree")
    if dim < 1 or split < 2:
      raise Exception("additionTree has invalid dim or split size")
    if maxs:
      if len(maxs) != split:
        raise Exception("additionTree has invalid maxs lens")
    else:
      maxs = [dim] *split
    for val in maxs:
      Lib._checkType(val, int, "additionTree")
      if val < 0:
        raise Exception("additionTree has invalid maxs value")
    out = []
    if split == 2:
      for i in range(dim +1):
        if not maxs or (maxs[0] >= dim-i and maxs[1] >= i):
          out.append([dim-i, i])
    else:
      splits = [0] *split
      splits[0] = dim
      Lib.__additionTree(splits, out, maxs, 0)
    return out

  @staticmethod
  def __additionTree(splits, out, maxs, idx):
    """Splits number in splits[0] into all additions of length of splits.
       Uses out for output list, idx for current tree and optional maxs list."""
    store = True
    for pos,val in enumerate(maxs):
      if splits[pos] > val:
        store = False
    if store:
      out.append(splits[:])
    old = splits[idx]
    for cnt in range(1, old +1):
      splits[idx] = old -cnt
      if idx < len(splits) -1:
        splits[idx +1] += cnt
        Lib.__additionTree(splits, out, maxs, idx +1)
        splits[idx +1] -= cnt
    splits[idx] = old

  @staticmethod
  def allIndices(dim, limit):
    """allIndices(dim, limit)
       Yield dim len list of all numbers 0 to limit."""
    yield [0]*dim
    for siz in range(1, dim *limit +1):
      for out in (x for x in Lib.additionTree(siz, dim) if max(x) < limit +1):
        yield out

  @staticmethod
  def triads(dim):
    """triads(dim)
       Generate a list of all independent triads for a basis of dim > 2."""
    Lib._checkType(dim, int, "triads")
    if dim < 3:
      raise Exception("Invalid triad dimension")
    pAll = list(Lib.comb(dim, 3, True))
    faces = []
    facesLen = 0       # First count the expected independent faces
    for triad in pAll:
      cnt = 0
      for face in faces:
        cnt = 0
        if triad[0] in face:
          cnt += 1
        if triad[1] in face:
          cnt += 1
        if triad[2] in face:
          cnt += 1
        if cnt > 1:
          break
      if cnt < 2:
        facesLen += 1
        faces.append(triad)
    if dim == 3:
      yield [1, 2, 3]
    else:
      # Use all combinations of triads and check for independence (cnt <= 1)
      morphs = []
      for triads in Lib.comb(len(pAll), facesLen, True):
        faces = []
        cnt = 0
        for idx in triads:
          triad = pAll[idx -1]
          for face in faces:
            cnt = 0
            if triad[0] in face:
              cnt += 1
            if triad[1] in face:
              cnt += 1
            if triad[2] in face:
              cnt += 1
            if cnt > 1:
              break
          if cnt < 2:
            faces.append(triad)
        if len(faces) == facesLen:
          yield faces

  @staticmethod
  def fixFilename(filename, dirname=None, extname=None):
    """fixFilename(filename, [dirname, extname])
       Fix Windows path separators and optionally add dir and ext."""
    if dirname and not os.path.dirname(filename):   # Use default path
      filename = os.path.join(dirname, filename)
    if extname and not os.path.splitext(filename)[1]:
      filename += extname
    if sys.platform == "win32":
      filename = filename.replace("\\", "\\\\")
    return filename

  @staticmethod
  def __save(value):
    """Internal function to iterate and expand strings."""
    if isinstance(value, (list, tuple, Matrix, set)):
      out = []
      for val in value:
        out.append(Lib.__save(val))
    elif isinstance(value, Lib._basestr):
      out = str(value)
    else:
      out = value
    return out

  @staticmethod
  def _save(filename, name, value, path="", ext="", mode="w"):
    """save(filename, name, value,[path,ext,mode="w"])
       Print value into a file which needs an extension.
       Used by Calculator. No nested dictionaries."""
    Lib._checkType(filename, Lib._basestr, "save")
    Lib._checkType(name, Lib._basestr, "save")
    Lib._checkType(path, Lib._basestr, "save")
    Lib._checkType(ext, Lib._basestr, "save")
    with open(Lib.fixFilename(filename, path, ext), mode) as fp:
      if isinstance(value, dict):
        fp.write("%s = { \\\n" %name)
        for key,var in value.items():
          if isinstance(key, Lib._basestr):
            key = '"%s"' %key
          fp.write(" %s: %s,\n" %(key, Lib.__save(var)))
        fp.write("}\n")
      elif isinstance(value, (list, tuple, Matrix, set)):
        typ = "Matrix" if isinstance(value, (Tensor, Matrix)) else ""
        fp.write("%s = %s(\\\n" %(name, typ))
        for val in value:
          fp.write("  %s,\n" %Lib.__save(val))
        fp.write(")\n")
      elif isinstance(value, (Tensor, Matrix)):
        fp.write("%s = Matrix(%s)\n" %(name, Lib.__save(value)))
      else:
        fp.write("%s = %s\n" %(name, Lib.__save(value)))
  save=_save

  @staticmethod
  def _basisStrs(basis):
    """Return basis and mBasis as strings."""
    pBasis = []
    mBasis = []
    for val in basis:
      txt = isinstance(val, Lib._basestr)
      if txt and ("-" in val or "+" in val):
        raise Exception("Multiple text terms not supported: %s" %val)
      if str(val)[:1] == "-":
        mBasis.append(str(val))
        pBasis.append("-" +val if txt else str(-val))
      else:
        pBasis.append(str(val))
        mBasis.append("-" +val if txt else str(-val))
    return pBasis,mBasis
  
  @staticmethod
  def cycles(x,y=None, xy=None):
    """cycles(x, [y,xy]):
       Return list ((b,c),(b,bc),(c,bc)) where x or x,y=(b,c,...) & bc=xy or
       abs(b*c). These are cycles of non-associative triads."""
    if not isinstance(x, (list, tuple, Matrix)):
      x = [x]
    if y is not None:
      if not isinstance(y, (list, tuple, Matrix)):
        y = [y]
      x = list(x) +list(y)
    if len(x) < 2:
      raise Exception("Invalid Lib.cycles array length not >= 2")
    if xy is None:
      xy = abs(x[0] *x[1])
    d = list(x[2:])
    b,c,bc = sorted((x[0], x[1], xy))
    out = [tuple([b, c] +d)]
    out.append(tuple([b, bc] +d))
    out.append(tuple([c, bc] +d))
    return out

  @staticmethod
  def triadPairs(pairFn, basis, name, dump=False, param=None, cntOnly=False):
    """triadPairs(pairFn,basis,[dump,param,cntOnly])
       Return list for pairFn(list,basis,idx,a,b,(aa,bb,param)) being called for
       all pairs in triadDump() order if set. Dump logs progress and checks
       memory & aborts if too small. cntOnly uses paiFn(None) & returns cnt."""
    Lib._checkList(basis, None, name)
    Lib._checkType(dump, bool, name)
    Lib._checkType(cntOnly, bool, name)
    if not hasattr(pairFn, "__call__"):
      raise Exception("%s triadPairs pairFn needs to be a function" %name)
    if cntOnly and pairFn == Lib.inverseTriads:
      raise Exception("%s inverse triads can't have cntOnly" %name)
    lr = len(basis)
    out = None if cntOnly else [[]] *(lr *(lr -1)) # Overwrite [] if non-empty
    cnt = 0
    if dump:
      Lib.procTime(True)
      step = max(100, lr //10)
    for a in range(lr):
      aa = basis[a]
      if dump and Lib._checkMem(a, step, cnt):
        break
      for b in range(a +1, lr):
        params = (aa, basis[b], param)
        tmp = pairFn(out, basis, lr, a, b, params)
        if isinstance(tmp, int):
          cnt += tmp
        elif cnt == 0:
          cnt = list(tmp)
        else:
          for idx,val in enumerate(tmp):
            cnt[idx] += val
    if dump:
      Lib.checkMem(a, extra=cnt, finish=True)
    return cnt if cntOnly else out

  @staticmethod
  def inverseTriads(out, basis, lr, b, c, params):
    """inverseTriads(out,basis,lr,b,c,params)
       Lib.triadPairs pairFn to invert triad result."""
    cnt = 0
    bufOut = []
    bb,cc,buf = params
    got = buf[b *lr +c]
    for d in range(c +1, lr):
      if d not in got:
        bufOut.append(d)
        cnt += 1
    if out:
      out[b *lr +c] = bufOut
    return cnt

  @staticmethod
  def allTriads(basis, dump=False, cntOnly=False):
    """allTriads(basis,[dump,cntOnly])
       Return a list of all simple faces (or cnt). See triadDump()."""
    Lib._checkList(basis, None, "allTriads")
    return Lib.triadPairs(Lib._allTriads, basis, "allTriads", dump,
                             None, cntOnly)
  @staticmethod
  def _allTriads(out, basis, lr, b, c, params):
    cnt = 0
    bufOut = []
    bb,cc,tmp = params
    for d in range(c +1, lr):
      if tmp:
        if d not in tmp[b *lr +c]:
          bufOut.append(d)
      else:
        bufOut.append(d)
      cnt += 1
    if out:
      out[b *lr +c] = bufOut
    return cnt

  @staticmethod
  def allCycles(basis, table=None, dump=False):
    """allCycles(basis,[table,dump])
       Return two lists of all simplex single & 3-cycle faces. Single contains
       non cycles including assocCycles() or quaternion-like triads. All other
       triads are non-associative and 3-cycle faces only list the first cycle of
       (b,c,d), (b,bc,d), (c,bc,d). bc is multiplied from basis or if table is a
       multiplication table then bc is looked up. Simple triples with repeated
       elements or scalars are not triads. See triadDump to display results."""
    buf = Lib.triadPairs(Lib.__allCycles, basis, "allCycles", dump, table)
    lr = len(basis)
    single, cycles = [[]] *(lr *(lr -1)), [[]] *(lr *(lr -1)) 
    for b in range(lr):
      for c in range(b +1, lr):
        offs = b *lr +c
        tmp = sorted(buf[offs])
        bufSng, bufCyc = [], []
        if tmp:
          oldIdx = tmp[0]
          cntIdx = 0
          for idx in tmp +[-2]:
            if idx == oldIdx:
              cntIdx += 1
            elif cntIdx == 1:
              bufSng.append(oldIdx)
              cntIdx = 1
            elif cntIdx == 3:
              bufCyc.append(oldIdx)
              cntIdx = 1
            else:
              raise Exception("Invalid Lib.allCycle count %d for %s" %(cntIdx,
                               (basis[b], basis[c], basis[oldIdx])))
            oldIdx = idx
        single[offs], cycles[offs] = bufSng, bufCyc
    return single, cycles
  @staticmethod
  def __allCycles(out, basis, lr, b, c, params):
    bb,cc,table = params
    if table:
      bc = abs(table.get(b,c))
    else:
      bc = basis.index(abs(bb *cc))
    b1,c1,bc1 = sorted((b, c, bc))
    bufOut = out[b1 *lr +c1]
    if not bufOut: bufOut = []
    bufOut.extend(range(c +1, lr))
    out[b1 *lr +c1] = bufOut
    return 0

  @staticmethod
  def associativeCycles(basis, table=None, dump=False):
    """assoc[iative]Cycles(basis,[table,dump]) See Tensor.assocTriads()
       Return a list of independent simplex 3-cycle faces. See allCycles()."""
    return Lib.triadPairs(Lib.__assocCycles, basis, "assocCycles", dump, table)
  @staticmethod
  def __assocCycles(out, basis, lr, b, c, params):
    cnt = 0
    bb,cc,table = params
    if table:
      bc = abs(table.get(b,c))
    else:
      bc = basis.index(abs(bb *cc))
    b1,c1,bc1 = sorted((b, c, bc))
    if out[b1 *lr +c1] != [bc1]:
      out[b1 *lr +c1] = [bc1]
      cnt = 1
    return cnt
  assocCycles = associativeCycles

  @staticmethod
  def expandPairList(pairList):
    """expandPairList(pairList)
       Input a list of (b,c),(ds...) pairs and yield the triads (b,c,d)."""
    Lib._checkList(pairList, (tuple, list), "expandPairList")
    for pair in pairList:
      Lib._checkList(pair, (tuple, list), "expandPairList")
      if len(pair) != 2 or len(pair[0]) != 2:
        raise Exception("Invalid PairList for expandPairList")
      b,c = pair[0]
      dList = pair[1]
      for d in dList:
        yield (b,c,d)

  @staticmethod
  def triadDump(pairBuf, basis, paired=False, dump=False):
    """triadDump(pairBuf,basis, [paired,dump])
       Yield triad list from Lib.triadPairs() with (b,c) unique pairs of basis
       elements for non-empty d results. If expand yield pairs ((b,c), (ds...)).
       See Lib.expandPairList() to generate triads."""
    Lib._checkList(pairBuf, None, "triadDump")
    Lib._checkList(basis, None, "triadDump")
    Lib._checkType(paired, bool, "triadDump")
    Lib._checkType(dump, bool, "triadDump")
    lr = len(basis)
    if len(pairBuf) != lr *(lr -1):
      raise Exception("triadDump pairBuf should be result of triadPairs")
    if dump:
      Lib.procTime(True)
      step = max(100, lr //10)
    for a in range(lr):
      if dump and Lib._checkMem(a, step):
        break
      idx = a *lr
      for b in range(a +1, lr):
        buf = pairBuf[idx +b]
        if buf:
          aa,bb = basis[a], basis[b]
          if paired:
            yield (aa,bb), list(basis[c] for c in buf)
          else:
            for c in buf:
              yield aa, bb, basis[c]
    if dump:
      Lib.checkMem(a, finish=True)
Common = Lib

################################################################################
class Tensor(list):
  """Class Tensor for development and test instead of using numpy. May contain
     basis numbers like i, j, k. These are tensors since they may contain basis
     vectors. If g_lo = Tensor(e0,e1,e2,e3) and g_hi = Tensor(-e0,e1,e2,e3) then
     metric tensor is (g_lo*g_hi.transpose()).scalar() = Tensor.Diag([1]*4).
     Class Matrix maps to this class if numpy is not available."""
  shape = (0,0)

  def __init__(self, *args):
    """Tensor(list)
       Define a nx1 or nxm matrix as a list or list of lists."""
    super(Tensor, self).__init__()
    Lib._checkList(args, None, "Tensor")
    if not args:
      self.shape = (0, 0)
    else:
      if not args or len(args) == 1 and not args[0]:
        args = (0,)
      self.shape = (len(args), 1)
      if isinstance(args[0], (list, tuple, Matrix)):
        if len(args) == 1:
          args = args[0]
          self.shape = (len(args), 1)
        if isinstance(args[0], (list, tuple, Matrix)):
          self.shape = (len(args), len(args[0]))
    for row in args:
      if isinstance(row, (list, tuple, Matrix)):
        if len(row) == 1 and self.shape[1] == 1:
          self.append(row[0])
        else:
          self.append(list(row))
          if len(self[-1]) != self.shape[1]:
            raise Exception("Inconsistant Tensor row lengths")
      else:
        self.append(row)

  def get(self, x, y):
    """get(x,y)
       Return the value of a matrix element. Used for FrameMatrix."""
    return self[x][y]

  def len(self):
    """len()
       Return the vector length or total number of tensor cells."""
    return self.shape[0] *self.shape[1]

  def __str__(self):
    """Overload standard string output."""
    out = "(" if self else "( \n"
    sep = ""
    for row in self:
      if isinstance(row, list):
        sep = "("
        for col in row:
          if isinstance(col, float):
            out += sep +Lib.getResolNum(col)
          else:
            out += sep +str(col)
          sep = ", "
        out += '),\n'
      else:
        if isinstance(row, float):
          out += sep +Lib.getResolNum(row)
        else:
          out += sep +str(row)
        sep = ", "
    return (out[:-2] +')') if out[-1]=='\n' else out +')'

  def __eq__(self, mat):
    """Return True if 2 matrices are equal taking resolution into account."""
    precision = Lib._getPrecision()
    if not isinstance(mat, (Tensor, Matrix)):
      if isinstance(mat, (list, tuple)):
        mat = Tensor(mat)
      else:
        return self.__reduce(lambda x,y: x and (y -mat) < precision, True)
    if self.shape != (mat.shape if len(mat.shape)==2 else (mat.shape[0], 1)):
      return False
    for idx1,val1 in enumerate(self):
      if isinstance(val1, (list, tuple, Matrix)):
        for idx2,val2 in enumerate(val1):
          val3 = mat[idx1][idx2]
          if isinstance(val2, (int, float)) and isinstance(val3, (int, float)):
            if abs(val2 -val3) >= precision:
              return False
          else:
            if val2 != val3:
              return False
      else:
        val3 = mat[idx1]
        if isinstance(val1, (int, float)) and isinstance(val3, (int, float)):
          if abs(val1 - val3) >= precision:
            return False
        else:
          if val1 != val3:
            return False
    return True
 
  def __mul__(self, mat):
    """Numpy multpily is element by element - use Tensor.dot()."""
    raise Exception("Tensor does not need numpy multipy - use Matrix.dot().")

  def dot(self, mat):
    """Matrix multiplication for ranks 3x3 * 3x3 and 3x3 * 3x1."""
    if not isinstance(mat, Matrix):
      if isinstance(mat, (list, tuple)) and not isinstance(mat, Tensor):
        mat = Tensor(*mat)
    a = []
    if isinstance(mat, (Tensor, Matrix)):
      if self.shape[1] != mat.shape[0]:
        raise Exception("Invalid Matrix sizes for multiplying: %sx%s" \
                     %(self.shape, mat.shape))
      if self and isinstance(self[0], (list, tuple, Matrix)): # self.shape>(1,1)
        if mat.shape[1] > 1:                         # matrix * matrix
          for row,val1 in enumerate(self):
            a.append([None] *len(mat))
            for k,val2 in enumerate(val1):
              for col,val3 in enumerate(mat[k]):
                if a[row][col]:
                  a[row][col] += val2 *val3
                else:
                  a[row][col] = val2 *val3
        else:                                         # matrix * vector
          for row,val1 in enumerate(self):
            a.append(val1[0] *mat[0])
            for col,val2 in enumerate(val1[1:]):
              a[row] += val2 *mat[col +1]
      elif isinstance(mat[0], (list, tuple, Matrix)): # mat.shape > (1,1)
        for row,val1 in enumerate(mat):               # vector * matrix
          a.append(self[0] *val1[0])
          for col,val2 in enumerate(val1[1:]):
            a[row] += self[col +1] *val2
      elif self.shape[1] != mat.shape[0]:
        raise Exception("Invalid Matrix sizes for product: %sx%s" \
                     %(self.shape, mat.shape))
      elif self.shape[0] == 1:
        a.append(self[0] *mat[0])                    # inner product
        for row,val1 in enumerate(self[1:]):
          a[0] += val1 *mat[row +1]
      else:                             
        for row,val1 in enumerate(self):             # outer product
          a.append([None] *len(mat))
          for col,val2 in enumerate(mat):
            a[row][col] = val1 *val2
      return Tensor(*a)
    for row,val1 in enumerate(self):
      if isinstance(val1, (list, tuple, Matrix)):
        a.append([None] *len(val1))
        for col,val2 in enumerate(val1):
          if isinstance(mat, Lib._basestr):
            a[row][col] = self.__mulStr(str(val2), mat)
          elif isinstance(val2, Lib._basestr):
            a[row][col] = self.__mulStr(val2, str(mat))
          else:
            a[row][col] = val2 *mat
      elif isinstance(mat, Lib._basestr):
        a.append(self.__mulStr(str(val1), mat))
      elif isinstance(val1, Lib._basestr):
        a.append(self.__mulStr(val1, str(mat)))
      else:
        a.append(val1 *mat)
    return self.copy(a)

  def __rmul__(self, mat):
    """Matricies elements may be non-commutative."""
    if mat and isinstance(mat, (list, tuple)):
      return Tensor(mat) *self
    a = []
    for row,val1 in enumerate(self):
      if isinstance(val1, (list, tuple)):
        a.append([0] *len(val1))
        for col,val2 in enumerate(val1):
          if isinstance(mat, Lib._basestr):
            a[row][col] = self.__mulStr(str(val2), mat)
          elif isinstance(val2, Lib._basestr):
            a[row][col] = self.__mulStr(val2, str(mat))
          else:
            a[row][col] = mat * val2
      elif isinstance(mat, Lib._basestr):
        a.append(self.__mulStr(mat, str(val1)))
      elif isinstance(val1, Lib._basestr):
        a.append(self.__mulStr(str(mat), val1))
      else:
        a.append(mat * val1)
    return self.copy(a)

  def __mulStr(self, valStr1, valStr2):
    if valStr1 == "0" or valStr2 == "0":
      return "0"
    if valStr1 == "1":
      return valStr2
    if valStr2 == "1":
      return valStr1
    if valStr1.find("+") >= 0 or valStr1[1:].find("-") >= 0:
      valStr1 = "(%s)" %valStr1
    if valStr2.find("+") >= 0 or valStr2[1:].find("-") >= 0:
      valStr2 = "(%s)" %valStr2
    if valStr1 == "-1":
      if valStr2[:1] == "-":
        return valStr2[1:]
      return "-" +valStr2
    if valStr2 == "-1":
      if valStr1[:1] == "-":
        return valStr1[1:]
      return "-" +valStr1
    return "%s*%s" %(valStr1, valStr2)

  def __div__(self, den, isFloor=False):
    """Matrix division by scalar only."""
    if isinstance(den, (list, tuple, Matrix)):
      raise Exception("Invalid Matrix division")
    a = []
    for row,val1 in enumerate(self):
      if isinstance(val1, (list, tuple, Matrix)):
        a.append([0] *len(val1))
        for col,val2 in enumerate(val1):
          a[row][col] = int(val2 /den) if isFloor else val2 /den
      else:
        a.append(int(val1 /den) if isFloor else val1 /den)
    return self.copy(a)
  __truediv__ = __div__
  def __floordiv__(self, den):
    """Matrix div (//) by scalar only."""
    return self.__div__(den, True)

  def __add__(self, mat):
    """Return addition matrix."""
    out = []
    if not isinstance(mat, Matrix):
      if isinstance(mat, (list, tuple)) and not isinstance(mat, Tensor):
        mat = Tensor(*mat)
    Lib._checkType(mat, (Tensor, Matrix), "add")
    if self.shape != (mat.shape if len(mat.shape)==2 else (mat.shape[0],1)):
      raise Exception("Invalid Matrix size for add/sub")
    if self and isinstance(self[0], (list, tuple, Matrix)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          tmp = mat[idx1][idx2]
          if isinstance(val2, Lib._basestr) or \
             isinstance(tmp, Lib._basestr):
            if str(val2) == "0":
              out[idx1].append(tmp)
            elif str(tmp) == "0":
              out[idx1].append(val2)
            elif str(tmp)[:1] == "-":
              out[idx1].append("%s %s" %(val2,tmp))
            else:
              out[idx1].append("%s +%s" %(val2,tmp))
          else:
            out[idx1].append(val2 +tmp)
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(val1 +mat[idx1])
    return self.copy(out)

  def __neg__(self):
    """Use matrix multiplication for negation."""
    return self.dot(-1)
  def __sub__(self, mat):
    """Subtract 2 matricies."""
    if not isinstance(mat, (Tensor, Matrix)):
      if isinstance(mat, (list, tuple)) and not isinstance(mat, (Tensor, Matrix)):
        mat = Tensor(*mat)
    Lib._checkType(mat, (Tensor, Matrix), "sub")
    return self.__add__(-mat)
  def __rsub__(self, mat):
    """Subtract matricies"""
    return self.__neg__().__add__(mat)
  def __pos__(self):
    """Unitary + operator for matrix."""
    return self
  def __abs__(self):
    """Unitary abs operator for matrix."""
    out = []
    if self and isinstance(self[0], (list, tuple, Matrix)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          out[idx2].append(abs(val2))
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(abs(val1))
    return self.copy(out)
  abs = __abs__

  def copy(self, arr=None):
    """copy(arr)
       Return deep copy at both levels and shape set with optional overwrite."""
    if arr is None:
      arr = self
    Lib._checkList(arr, None, "copy")
    if arr and isinstance(arr[0], (list, tuple)):  # arr.shape > (1,1)
      return Tensor([row[:] for row in arr])
    out = Tensor(*arr)
    out.shape = self.shape  # Keep transpose
    return out

  def sym(self, mat):
    """sym(mat)
       Return self*mat +mat*self."""
    return self.dot(mat) +mat.dot(self)

  def asym(self, mat):
    """asym(mat)
       Return self*mat -mat*self."""
    return self.dot(mat) -mat.dot(self)

  def diag(self, vector=None):
    """diag([vector])
       Return diagonal of square matrix or diagonal of self * vector.transpose
       as vector. Hence trace=sum(matrix.diag()) and dot product is
       sum(v.diag(vector)). This allows Dickson algebra (the product of Real,
       Complex, Quaternion and Octernion numbers) as Tensor(R,Q(1),Q(2), O(3)).
       Addition is via + and multiplication via diag(v)."""
    out = []
    shape = self.shape
    if vector is None:
      if shape[0] != shape[1]:
        raise Exception("Matrix for diag must be square")
      for idx in range(shape[0]):
        out.append(self[idx][idx])
    else:
      Lib._checkList(vector, None, "diag")
      if len(self) != len(vector):
        raise Exception("Vectors for diag must be the same length")
      for idx,val in enumerate(self):
        out.append(val *vector[idx])
    return Tensor(*out)

  def multiply(self, mat):
    """multiply(mat)
       Return Hadamard product matrix."""
    out = []
    if not isinstance(mat, Matrix):
      if isinstance(mat, (list, tuple)) and not isinstance(mat, Tensor):
        mat = Tensor(*mat)
    Lib._checkType(mat, (Tensor, Matrix), "multiply")
    if self.shape != mat.shape:
      raise Exception("Invalid Matrix sizes for multiply %sx%s" \
                     %(self.shape, mat.shape))
    if self and isinstance(self[0], (list, tuple, Matrix)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          out[idx1].append(val2 *mat[idx1][idx2])
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(val1 *mat[idx1])
    return self.copy(out)

  def cayleyDicksonMult(self, vector, baezRule=False):
    """cayleyDicksonMult(vector, [baezRule])
       Multiply Tensor pairs using Cayley-Dickson rule, Wikipedia or J.C.Baez:
       q1=(p,q); q2=(r,s); q1*q2 = (pr -s*q, sp +qr*) [wikiRule]
       q1=(p,q); q2=(r,s); q1*q2 = (pr -sq*, p*s +rq) [baezRule]."""
    out = []
    Lib._checkList(vector, None, "cayleyDicksonMult")
    Lib._checkType(baezRule, bool , "cayleyDicksonMult")
    if len(self) != len(vector) or len(self) != 2:
      raise Exception("Vectors for cayleyDicksonMult must have length two")
    if self and not isinstance(self[0], (list, tuple, Matrix)):
      if baezRule:
        out.append(self[0] *vector[0] -vector[1] *self[1].conjugate())
        out.append(self[0].conjugate() *vector[1] +vector[0] *self[1])
      else:
        out.append(self[0] *vector[0] -vector[1].conjugate() *self[1])
        out.append(vector[1] *self[0] +self[1] *vector[0].conjugate())
      return Tensor(*out)
    raise Exception("Self for cayleyDicksonMult needs to be a vector")

  def trim(self):
    """trim()
       Return copy with cells smaller than precision set to zero."""
    out = []
    if self and isinstance(self[0], (list, tuple, Matrix)):  # self.shape > (1,1)
      for idx1 in range(len(self[0])):
        out.append([])
      for idx1,val1 in enumerate(self):
        for val2 in val1:
          out[idx1].append(0.0 if abs(val2) < Lib._getPrecision() else val2)
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(0.0 if abs(val1) < Lib._getPrecision() else val1)
    return self.copy(out)

  def slice(self, offset, shape=None):
    """slice(offset,[shape])
       Return copy of 2-D selection or square if offset, shape are integers."""
    if not isinstance(offset, (list, tuple, Matrix)):
      offset = (offset, offset)
    Lib._checkList(offset, int, "slice", 2)
    if shape is None:
        shape = (self.shape[0], self.shape[1])
    else:
      if not isinstance(shape, (list, tuple, Matrix)):
        shape = (shape, shape)
    Lib._checkList(shape, int, "slice", 2)
    out = []
    if self and isinstance(self[0], (list, tuple, Matrix)):  # self.shape>(1,1)
      if shape is None:
        shape = (self.shape[0] -offset[0], self.shape[1] -offset[1])
      for idx1 in range(max(1,shape[0])):
        out.append([])
      for idx1 in range(max(1,shape[0])):
        if offset[0] +idx1 < len(self[0]):
          val1 = self[offset[0] +idx1]
        for idx2 in range(shape[1]):
          if offset[1] +idx2 < len(val1):
            out[idx1].append(val1[offset[1] +idx2])
      return Tensor(*out)
    if shape is None:
      shape = (self.shape[0] -offset[0], )
    out = self[offset[0]:offset[0] +shape[0]]
    out = Tensor(*out)
    if self.shape[1] > 1:
      out.shape = (1, out.size[0])
    return out

  def reshape(self, shape):
    """reshape(shape)
       Return copy with new 2-D shape or square if shape is integer."""
    twoD = self and isinstance(self[0], (list, tuple, Matrix))
    if not isinstance(shape, (list, tuple, Matrix)):
      shape = (shape, shape if twoD else 1)
    Lib._checkList(shape, int, "reshape", 2)
    if shape[0] < 1 or shape[1] < 1:
      raise Exception("Invalid reshape size")
    out = []
    if twoD: 
      Lib._checkType(shape[1], int, "reshape")
      for idx1 in range(max(1,shape[0])):
        out.append([])
      for idx1 in range(max(1,shape[0])):
        val1 = self[idx1] if idx1 < len(self[0]) else []
        for idx2 in range(shape[1]):
          out[idx1].append(val1[idx2] if idx2 < len(val1) else 0)
      return Tensor(*out)
    if shape[0] == 1:  # Transpose case
      out = Tensor(self[:shape[1]])
      out.shape = (1, shape[1])
      return out
    elif shape[1] == 1:
      return Tensor(self[:shape[0]])
    raise Exception("Invalid reshape size for matrix")

  def scalar(self):
    """scalar()
       Return copy with non-scalar cells set to the scalar part."""
    out = []
    if self and isinstance(self[0], (list, tuple, Matrix)):  # self.shape>(1,1)
      for idx1 in range(len(self[0])):
        out.append([])
      for idx1,val1 in enumerate(self):
        for idx2,val2 in enumerate(val1):
          val3 = val2.scalar() if hasattr(val2, "scalar") else val2
          out[idx2].append(val2.scalar() if hasattr(val2, "scalar") else val2)
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out[idx2].append(val1.scalar() if hasattr(val1, "scalar") else val1)
    return self.copy(out)

  def transpose(self):
    """transpose()
       Return transpose matrix."""
    if self and isinstance(self[0], (list, tuple, Matrix)):  # self.shape>(1,1)
      out = []
      for idx1 in range(len(self[0])):
        out.append([])
      for idx1,val1 in enumerate(self):
        for idx2,val2 in enumerate(val1):
          out[idx2].append(val2)
      return Tensor(*out)
    out = Tensor(self)
    out.shape = (self.shape[1], self.shape[0])
    return out

  def reduce(self, fn, init=0):
    """reduce(fn, [init=0])
       Return matrix with x=fn(x[=init],y) applied to each element of self."""
    if not hasattr(fn, "__call__"):
      raise Exception("reduce(fn) needs to be a function")
    return self.__reduce(fn, init)

  def __reduce(self, fn, init):
    """Internal reduce(fn) method."""
    out = init
    #if self and isinstance(self[0], (list, tuple, Matrix)):
    if self.shape[1] > 1:
      for val1 in self:
        for val2 in val1:
          out = fn(out, val2)
      return out
    for val1 in self:
      out = fn(out, val1)
    return out

  def all(self):
    """all()
       Return True if all matrix values are True else False."""
    return self.__reduce(lambda x,y: x and y, True)

  def any(self):
    """any()
       Return True if any matrix values are True else False."""
    return self.__reduce(lambda x,y: x or y, False)

  def product(self):
    """prod[uct]()
       Return the product of the elements."""
    return self.__reduce(lambda x,y: x *y, 1)
  prod = product

  def function(self, fn):
    """function(fn)
       Return matrix with fn() applied to each element of self."""
    if not hasattr(fn, "__call__"):
      raise Exception("function(fn) needs to be a function")
    return self.__function(fn)

  def __function(self, fn):
    """Internal function(fn) method."""
    out = []
    if self and isinstance(self[0], (list, tuple, Matrix)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for val2 in val1:
          out[idx1].append(fn(val2))
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(fn(val1))
    return self.copy(out)

  def abs(self):
    """abs()
       Return matrix with absolute value applied to each element of self."""
    return self.__function(abs)
  __abs__ = abs

  def pow(self, exp):
    """pow(exp)
       Return matrix with power applied to each element of self."""
    return self.__function(lambda x: pow(x, exp) if isinstance(x, \
                    (int, float)) else x.__class__.pow(x, exp))
  __pow__ = pow

  def exp(self):
    """exp()
       Return matrix with exponentiation applied to each element of self."""
    return self.__function(lambda x: exp(x) if isinstance(x, \
                    (int, float)) else x.__class__.exp(x))

  def log(self):
    """log()
       Return matrix with logarithm applied to each element of self."""
    return self.__function(lambda x: log(x) if isinstance(x, \
                    (int, float)) else x.__class__.log(x))

  def morph(self, basis, labels=None):
    """morph(basis, [labels])
       Return self morphed using a list of pairs or basis->labels. Pairs are
       string names mapped as first->second. Basis & ones are replaced by
       labels and +-1 of labels type so may be of basis or string type."""
    Lib._checkList(basis, None, "morph")
    if labels is None:
      Lib._CheckList(basis, (list, tuple, Matrix), "morph")
      labels = list(x[1] for x in basis if len(x)==2)
      basis = list(x[0] for x in basis if len(x)==2)
    else:
      Lib._checkList(labels, None, "morph", len(basis))
    return self.__morphIn(basis, True).__morphOut(labels, True)

  def __morphIn(self, basis, unknown=False):
    """Internal method to return a Matrix as indices into basis elements."""
    out = []
    basis = list(basis)[:] +([1, 0,-1] if unknown else [1])
    pBasis,mBasis = Lib._basisStrs(basis)
    for val1 in self:
      row = []
      if isinstance(val1, (list, tuple, Matrix)):
        for val2 in val1:
          val2 = str(val2)
          if val2 in pBasis:
            row.append(pBasis.index(val2) +1)
          elif val2 in mBasis:
            row.append(-mBasis.index(val2) -1)
          elif unknown:
            row.append(basis.index(-1) +1)
          else:
            raise Exception("Element not found in basis for morph")
      else:
        val1 = str(val1)
        if val2 in pBasis:
          out.append(pBasis.index(val1) +1)
        elif var2 in mBasis:
          out.append(-mBasis.index(val1) -1)
        elif unknown:
          out.append(basis.index(-1) +1)
        else:
          raise Exception("Element not found in basis for morph")
      if row:
        out.append(row)
    return self.copy(out)

  def __morphOut(self, basis, unknown=False):
    """Internal method to return a Matrix of indices into basis elements."""
    try:
      out = []
      basis = list(basis)[:] +([1,0,-1] if unknown else [1])
      if any(map(lambda x: isinstance(x, Lib._basestr), basis)) or unknown:
        if unknown:
          basis[-1] = "XXX"
        basis,mBasis = Lib._basisStrs(basis)
      else:
        mBasis = list(-x for x in basis)
      for val1 in self:
        row = []
        if isinstance(val1, (list, tuple, Matrix)):
          for val2 in val1:
            row.append(mBasis[-val2 -1]  if val2 < 0 else basis[val2 -1])
        else:
          out.append(mBasis[-val1 -1] if val1 < 0 else basis[val1 -1])
        if row:
          out.append(row)
    except IndexError:
      raise Exception("Element not found in output for morph")
    return self.copy(out)

  def __morphPerm(self, perm):
    """Internal routine to return self with values, rows and cols permuted."""
    rows = []
    for idx in perm: # Swap rows
      if idx > 0:
        rows.append(self[idx -1])
      elif self.shape[1] == 1:
        rows.append(-self[-idx -1])
      else:
        rows.append(list((-v for v in self[-idx -1])))
    swap = []  # Swap columns
    if self.shape[1] == 1:
      if len(self) != len(perm):
        raise Exception("Invalid permute row length")
      swap = rows
    else:
      for idx in range(len(rows)):
        swap.append([]) 
        if len(self[idx]) != len(perm):
          raise Exception("Invalid permute row length")
      for idx1,row in enumerate(swap):
        for idx2 in perm:
          if idx2 > 0:
            row.append(rows[idx1][idx2 -1])
          else:
            row.append(-rows[idx1][-idx2 -1])
    out = []
    #compPerm = list(Tensor(perm).permInvert()) +[len(perm) +1]
    compPerm = perm[:] +[len(perm) +1]
    for val1 in swap:
      row = []
      if isinstance(val1, (list, tuple, Matrix)):
        for val2 in val1:
          val = compPerm[abs(val2) -1]
          row.append(-val if val2 < 0 else val)
      elif val1 in basis:
        val = compPerm[abs(val1) -1]
        row.append(-val if val1 < 0 else val)
      if row:
        out.append(row)
    return self.copy(out)

  def __morph(self, basis, labels):
    """Internal routine to return self with basis & ones replaced by labels
       and +-1 of labels type."""
    isStrBasis = isinstance(basis[0], Lib._basestr)
    if isStrBasis:
      pBasis = (str(x) for x in basis)
      mBasis = list((x[1:] if x[0]=="-" else "-"+x for x in pBasis))
    else:
      mBasis = list((-x for x in basis))
    isStrLabel = isinstance(labels[0], Lib._basestr)
    if isStrLabel:
      pLabels = (str(x) for x in labels)
      mLabels = list((x[1:] if x[0]=="-" else "-"+x for x in pLabels))
    else:
      mLabels = list((-x for x in labels))
    out = []
    for val1 in self:
      row = []
      if isinstance(val1, (list, tuple, Matrix)):
        for val2 in val1:
          if val2 in basis:
            val2 = labels[basis.index(val2)]
          elif val2 in mBasis:
            val2 = mLabels[mBasis.index(val2)]
          else:
            try:
              if isStrBasis:
                val2 = float(val2)
                if val2 == int(val2):
                  val2 = int(val2)
              elif isinstance(val2, (int, float)):
                pass
              elif val2.isScalar():
                val2 = val2.scalar()
                if val2 == int(val2):
                  val2 = int(val2)
              else:
                raise ValueError
              if isStrLabel:
                val2 = str(val2)
            except (ValueError, AttributeError):
              val2 = str(val2) # Swap element not found in basis
          row.append(val2)
      elif val1 in basis:
        row = labels[basis.index(val1)]
      elif val1 in mBasis:
        row = mLabels[mBasis.index(val1)]
      else:
        try:
          if isStrBasis:
            val1 = float(val1)
            if val1 == int(val1):
              val1 = int(val1)
          elif isinstance(val1, (int, float)):
            pass
          elif val2.isScalar():
            val1 = val1.scalar()
            if val1 == int(val1):
              val1 = int(val1)
          else:
            raise ValueError
          if isStrLabel:
            val2 = str(val1)
        except (ValueError, AttributeError):
          raise Exception("Swap element not found in basis: %s" %val1)
      out.append(row)
    return self.copy(out)

  def differences(self, mat, ignore=None):
    """diff[erences](mat,[ignore])
       Return list of indicies for differences of 2 matricies & ignore value."""
    Lib._checkList(mat, None, "diff")
    out = []
    isStr = (len(self) > 0 and isinstance(self[0], Lib._basestr))
    for idx1,val1 in enumerate(self):
      if isinstance(val1, (list, tuple, Matrix)):
        isStr = (len(self[0]) > 0 and isinstance(self[0][0], Lib._basestr))
        for idx2,val2 in enumerate(val1):
          if val2 != ignore:
            diff = (val2 != mat[idx1][idx2])
            if isStr and not diff:
              try:
                diff = (float(val2) != float(mat[idx1][idx2]))
              except:
                pass
            if diff:
              out.append([idx1,idx2])
      elif val1 != ignore:
        diff = (val1 != mat[idx1])
        if isStr and not diff:
          try:
            diff = (float(val2) != float(mat[idx1][idx2]))
          except:
            pass
        if diff:
          out.append([idx1])
    return out
  diff=differences

  def dump(self, xLabels=[], yLabels=[], name=None):
    """dump([xLabels,yLabels,name])
       Pretty print of Matrix with optional labels."""
    s = t = 0
    xName = "" if name is None else name
    if xLabels:
      Lib._checkList(xLabels, None, "dump", self.shape[1])
      if not yLabels and len(xLabels) == self.shape[0]:
        yLabels = xLabels
      s = max(map(lambda x :len(str(x)), xLabels))
    if yLabels:
      Lib._checkList(yLabels, None, "dump", self.shape[0])
      t = max(map(lambda x :len(str(x)), yLabels))
    if self.shape[1] > 1:
      trans = self.transpose()
      sizes = list(max(map(lambda x: len(str(x)), row)) for row in trans)
    else:
      sizes = [max(map(lambda x :len(str(x)), self))]
    formXs = list(" %%%ds" %(max(s, x)) for x in sizes)
    nameX = "%%-%ds" %(max(t,len(str(xName))) +3)
    formY = " %%%ds" %max(t,len(str(xName)))
    if xLabels or name is not None:
      sys.stdout.write(nameX %str(xName) +"%s\n" %"".join(formXs[x[0]] \
                    %x[1] for x in enumerate(xLabels)))
      if xLabels:
        sys.stdout.write(formY %"" +'-' *(sum(max(s,x)+1 for x in sizes) +2)+'\n')
    for ii,vals in enumerate(self):
      if yLabels:
        sys.stdout.write(formY %yLabels[ii] +"| ")
      if self.shape[1] == 1:
        sys.stdout.write(formXs[0] %vals)
      elif self.shape[0] == 1:
        for jj,val in enumerate(self):
          sys.stdout.write(formXs[jj] %val)
        sys.stdout.write("\n")
        break
      else:
        for jj,val in enumerate(vals):
          sys.stdout.write(formXs[jj] %val)
      sys.stdout.write("\n")

  def __checkSquare(self, basis, name, basisName):
    """Internal method to raise exception for invalid Tensor."""
    Lib._checkList(basis, None, name)
    if len(self) != len(basis) or len(self) == 0:
      raise Exception("Parameter %s length invalid for %s" \
                       %(basisName, name))
    if self.shape[0] != self.shape[1]:
      raise Exception("Tensor is not square for %s" %name)
    #if hasattr(self[0][0], "grades") != hasattr(basis[0], "grades"):
    #  raise Exception("Tensor is not the same type as basis for %s" %name)

  def search(self, basis, cf, cfBasis=None, num=-2, diffs=-1, cycles=False,
             permCycle=False, initPerm=[], squares=False, noAntiIso=False, 
             dump=False):
    """search(basis, cf, [cfBasis,num,diffs,cycles,permCycle,initPerm,squares,
              dump])
       Find self in signed permutation of cf for basis optionallly replacing
       cfBasis in cf with basis. All n! permutations and 2**n combinations of
       signs are searched and the histogram of difference counts is returned
       (default num=-2). If num=-1 then return the first match while num>-1
       returns this signed permutation. Otherwise diffs>=0 can be used to
       report each permutation for this many differences or less. Parameter cf
       is swapped as cfBasis -> basis and text searching is quicker. Parameter
       cycle is also quicker (default) because it converts to Tensor.cycles
       and uses morphCycles instead of isomorph. Parameter initPerm is used
       to fix the first part of all permutations such as octonians when
       looking for sedenions. permCycle outputs cycles instead of maps. If
       squares match diagagonal signature first."""
    self.__checkSquare(basis, "search", "basis")
    Lib._checkList(cf, None, "search")
    cf = Tensor(*cf)
    if cfBasis is None:
      cfBasis = basis
    cf.__checkSquare(cfBasis, "search", "cf")
    Lib._checkType(num, int, "search")
    Lib._checkType(diffs, int, "search")
    Lib._checkType(cycles, bool, "search")
    Lib._checkType(permCycle, bool, "search")
    Lib._checkList(initPerm, None, "search")
    Lib._checkType(squares, bool, "search")
    Lib._checkType(dump, bool, "search")
    cycPerm = Tensor(list(x+1 for x in range(len(basis))))
    chkPerm = list(sorted(list(abs(x) for x in initPerm)))
    for idx in range(1, len(initPerm)):
      if chkPerm[idx -1] == chkPerm[idx]:
        raise Exception("Search initPerm needs unique numbers 1-%s" \
                %range(len(basis)))
    extraPerm = list(x for x in range(1, len(basis) +1) if x not in chkPerm)
    if squares:
      basisDiag = Tensor(list(int(x) for x in self.diag()))
      cfDiag = list(int(x) for x in cf.diag())
    dim = len(basis)
    if dim != len(cfBasis):
      raise Exception("Search cfBasis length is not valid")    
    val1 = self[0]
    if isinstance(val1, (list, tuple, Matrix)) and len(val1) > 0:
      for val2 in val1[:]:
        if type(val1) == type(basis[0]):
          break
        val1 = val2
    if type(val1) != type(basis[0]):
      raise Exception("Invalid search basis type: %s !~ %s" %(type(val1),
                       type(basis[0])))
    dim0 = dim -len(initPerm)
    perms = Lib.perm(dim0, dim0, dump)
    difHisto = {}
    difRange = [99999999, 0]
    isStr = isinstance(basis[0], Lib._basestr)
    if isStr:
      mBasis = list((x[1:] if x[:1] == "-" else ("-" +x) for x in basis))
    else:
      mBasis = list((-x for x in basis))
    if cycles:
      cf = cf.cycles(cfBasis)
      cycleIso = self.cycles(basis)
      if len(cf) < len(self) or len(cycleIso) < len(self):
        raise Exception("Invalid basis for table.cycles()")
    else:
      newSelf = self.__morphIn(basis)
      cf = cf.__morphIn(cfBasis)
    cnt = 0
    antiIso = 0 if noAntiIso else (dim)
    for p in perms:                # For all permutations
      p0 = list(extraPerm[x -1] for x in p)
      p0 = initPerm +p0
      for n in range(antiIso +1):      # For all negative sign combinations
        for sgns in Lib.comb(dim, n, True):
          p1 = p0[:]
          for sgn in sgns:
            p1[sgn -1] *= -1

          # iso = self.isomorph(basis, p1)   # Signed swap rows, columns & cells
          if cycles:
            iso = cycleIso.morphCycles(p1, basis if isinstance(cycleIso[0][0],
                           Lib._basestr) else None)
          else:
            if squares and list(basisDiag.permute(p1)) != cfDiag:
              break
            iso = newSelf.morphPerm(p1)

          # Now report found, perm, diffs & accumulate histogram
          if diffs < 0:
            dif = [0]
            if iso == cf:
              dif = []
          else:
            dif = iso.differences(cf)
          if len(dif) not in difHisto:
            difHisto[len(dif)] = 0
          difRange[0] = len(dif) if len(dif) <= difRange[0] else difRange[0]
          difRange[1] = len(dif) if len(dif) >= difRange[1] else difRange[1]
          difHisto[len(dif)] += 1
          if permCycle:
            p1 = cycPerm.permute(p1, True).permCycles()
          if len(dif) == 0: # iso == cf
            sys.stdout.write("FOUND at %d %s\n" %(cnt, p1))
          elif diffs >= 0 and len(dif) <= diffs:
            sys.stdout.write("DIFFS at %d %s has %d: %s\n" %(cnt, p1,
                              len(dif), dif))
          if cnt == num or (num == -1 and len(dif) == 0):
            if diffs < 0 and len(dif) > 0:
              sys.stdout.write("GOT at %d %s\n" %(cnt, p1))
            return (p1, iso if cycles else iso.__morphOut(basis))
          cnt += 1
    i = difRange[0]
    stats = [0] *(i if cnt else 0)
    while i < difRange[1] +1:
      j = difHisto[i] if i in difHisto else 0
      stats.append(j)
      i += 1
    if num == -1 and dump:
      sys.stdout.write("NOT FOUND for %d: %s\n" %(cnt, stats))
    return (cnt, stats)
  
  def cycles(self, basis, results=None, degree=0, indices=False):
    """cycles(basis, [results, degree,indices])
       Return a list of multiplication triads for degree using basis
       for table inputs and results for outputs."""
    size = self.shape
    self.__checkSquare(basis, "cycles", "basis")
    if results is None:
      results = basis
    Lib._checkList(results, None, "cycles")
    Lib._checkType(degree, int, "cycles")
    Lib._checkType(indices, bool, "cycles")
    if degree and (isinstance(val, Lib._basestr) or not hasattr(val, "grades")):
      raise Exception("Parameter grade for cycles needs graded basis")
    pBasis,mBasis = Lib._basisStrs(basis)
    pResults,mResults = Lib._basisStrs(results)
    prod = {}
    out = []
    for el in basis:
      if el not in results:
        raise Exception("Basis of cycles missing from results: %s" %el)
    for i1 in range(size[0]):
      for i2 in range(i1 +1, size[1]):
        p1 = pBasis[i1]
        p2 = pBasis[i2]
        p3 = self[i1][i2]  # p1 * p2
        p3 = str(p3)
        if p3 != "0":
          sgn3 = "-" if p3[:1] == "-" else ""
          pp3 = p3[1:] if sgn3 else p3
          if p1 not in prod:
            prod[p1] = []
          if p2 not in prod:
            prod[p2] = []
          if pp3 not in prod:
            prod[pp3] = []
          if p1 not in prod[pp3] and p2 not in prod[pp3]:
            prod[p1].extend((p2, pp3))
            prod[p2].extend((p1, pp3))
            prod[pp3].extend((p1, p2))
            if not degree or basis[i3].grades(degree)[degree]:
              if indices:
                if pp3 not in pResults:
                  raise Exception("Basis of cycles not in results: %s" %pp3)
                i3 = pResults.index(pp3)
                out.append((pResults.index(p1) +1, pResults.index(p2) +1,
                        (-i3 -1) if sgn3 else i3 +1))
              else:
                out.append((p1, p2, pp3))
    return Tensor(*out)

  def __assocTriads1(self, x1, x2, basis, mBasis):
    """Internal function to return +/- index of x1 * x2 from square table."""
    scalar = len(basis) +1  # Represent +-1 as +-(len(basis) +1) & 0 as +2
    if abs(x1) == scalar: return x2 *(-1 if x1 < 0 else 1)
    if abs(x2) == scalar: return x1 *(-1 if x2 < 0 else 1)
    if x1 > scalar or x2 > scalar: return scalar +1
    mul = self[abs(x1) -1][abs(x2) -1]
    smul = str(mul)
    if smul == "0": return scalar +1
    if abs(x1) == abs(x2):
      neg = (smul[0] == "-")
      idx = scalar * (-1 if neg else 1)
    else:
      idx = (basis.index(mul) +1) if mul in basis else -mBasis.index(str(mul))-1
    if (x1 < 0) != (x2 < 0):
      idx = -idx
    return idx

  def __assocTriads2(self, x, left, basis, mBasis):
    """Internal fn to return +/- index of [x[0],x[1],x[2]] from square table."""
    if left:
      lhs = self.__assocTriads1(x[0], x[1], basis, mBasis)
      idx = self.__assocTriads1(lhs, x[2], basis, mBasis)
    else:
      rhs = self.__assocTriads1(x[1], x[2], basis, mBasis)
      idx = self.__assocTriads1(x[0], rhs, basis, mBasis)
    return idx

  def assocTriads(self, basis, nonAssoc=False, alternate=False, dump=False):
    """assocTriads(basis,[nonAssoc,alternate,dump])
       Return assoc traids [a,b,c]=0 or !=0 if nonAssoc. Alternate associativity
       is [a,b,c] = 0 if any two of a,b,c are equal. See Lib.triadDump()."""
    self.__checkSquare(basis, "assocTriads", "basis")
    Lib._checkType(nonAssoc, bool, "assocTriads")
    Lib._checkType(alternate, bool, "assocTriads")
    Lib._checkType(dump, bool, "assocTriads")
    pBasis,mBasis = Lib._basisStrs(basis)
    if isinstance(self[0][0], Lib._basestr):
      basis = pBasis
    tmp = Lib.triadPairs(self.__assocTriads, basis, "assocTriads", dump,
                         (alternate,mBasis))
    if not nonAssoc:  return tmp
    return Lib.triadPairs(Lib._allTriads, basis, "assocTriads", dump, tmp)
  def __assocTriads(self, out, basis, lr, a, b, params):
    cnt = 0
    buf = []
    aa,bb,param = params
    alternate,mBasis = param
    for c in range(b +1, lr):
      x = (a+1, b+1, c+1)
      ass = self.__assocTriads2(x, True, basis, mBasis) \
           -self.__assocTriads2(x, False, basis, mBasis)
      if ass:
        if alternate:
          none = True
          for y in ((x[1], x[0], x[2]), (x[1], x[2], x[0]), (x[0], x[2], x[1])):
            if ass != self.__assocTriads2(y, False, basis, mBasis) \
                     -self.__assocTriads2(y, True, basis, mBasis):
              none = False
              break
          if none:         # Alternate Associativity
            buf.append(y)
      else:
        buf.append(c)      # Associative
        cnt += 1
    out[a *lr +b] = buf
    return cnt

  def moufangTriads(self, basis, moufang=0, dump=False):
    """moufangTriads(basis,[moufang,dump])
       Return moufang traids depending on moufang=0-5 where 0 is none and 5 is
       the sum of all four. See Lib.triadDump() for return and other values.
         1: c*(a*(c*b)) -((c*a)*c)*b, 2: a*(c*(b*c)) -((a*c)*b)*c,
         3: (c*a)*(b*c) -(c*(a*b))*c, 4: (c*a)*(b*c) -c*((a*b)*c)."""
    self.__checkSquare(basis, "moufangTriads", "basis")
    Lib._checkType(moufang, int, "moufangTriads", (0,5))
    Lib._checkType(dump, bool, "moufangTriads")
    pBasis,mBasis = Lib._basisStrs(basis)
    if isinstance(self[0][0], Lib._basestr):
      basis = pBasis
    return Lib.triadPairs(self.__moufangTriads, basis, "moufangTriads", dump,
                          (moufang, mBasis))
  def __moufangTriads(self, out, basis, lr, a, b, params):
    cnt = 0
    buf = []
    aa,bb,param = params
    code,mBasis = param
    for c in range(b +1, lr):
      found = False
      doAll = code in (0,5)
      x = (a+1, b+1, c+1)
      if code == 1 or doAll:
        lhs = self.__assocTriads2((x[0], x[2], x[1]), False, basis, mBasis)
        rhs = self.__assocTriads2((x[2], x[0], x[2]), True, basis, mBasis)
        if self.__assocTriads1(x[0], lhs, basis, mBasis) \
          -self.__assocTriads1(rhs, x[2], basis, mBasis):
          found = True
          doAll = False
      if code == 2 or doAll:
        lhs = self.__assocTriads2((x[2], x[1], x[2]), False, basis, mBasis)
        rhs = self.__assocTriads2((x[0], x[2], x[1]), True, basis, mBasis)
        if self.__assocTriads1(x[0], lhs, basis, mBasis) \
          -self.__assocTriads1(rhs, x[2], basis, mBasis):
          found = True
          doAll = False
      if code == 3 or doAll:
        lhs = self.__assocTriads1((x[2], x[0]), basis, mBasis)
        rhs = self.__assocTriads2((x[2], x[0], x[1]), False, basis, mBasis)
        if self.__assocTriads2((lhs, x[1], x[2]), False, basis, mBasis) \
          -self.__assocTriads1(rhs, x[2], basis, mBasis):
          found = True
          doAll = False
      if code == 4 or doAll:
        lhs = self.__assocTriads1(x[2], x[0], basis, mBasis)
        rhs = self.__assocTriads2((x[0], x[1], x[2]), True, basis, mBasis)
        if self.__assocTriads2((lhs, x[1], x[2]), False, basis, mBasis) \
          -self.__assocTriads1(x[2], rhs, basis, mBasis):
          found = True
          doAll = False
      if code == 0:
        if not found:
          buf.append(c)
          cnt += 1
      elif found:
        buf.append(c)
        cnt += 1
    out[a *lr +b] = buf
    return cnt

  def abcTriads(self, basis, abc=0, nonAssoc=False, dump=False):
    """abcTriads(basis,[abc,nonAssocdump])
       Return traids for abc=0-3 where 0 is triple associator and 1-3: [a,c,b],
       [a,b,c], [b,a,c]. See Lib.triadDump() for return and other params."""
    self.__checkSquare(basis, "abcTriads", "basis")
    Lib._checkType(abc, int, "abcTriads", (0,3))
    Lib._checkType(nonAssoc, bool, "assocTriads")
    Lib._checkType(dump, bool, "abcTriads")
    pBasis = list((str(x) for x in basis))
    mBasis = list((x[1:] if x[:1] == "-" else "-" +x for x in pBasis))
    if isinstance(self[0][0], Lib._basestr):
      basis = pBasis
    tmp = Lib.triadPairs(self.__abcTriads, basis, "abcTriads", dump,
                          (abc, mBasis))
    if not nonAssoc:  return tmp
    return Lib.triadPairs(Lib._allTriads, basis, "abcTriads", dump, tmp)
  def __abcTriads(self, out, basis, lr, a, b, params):
    cnt = 0
    buf = []
    aa,bb,param = params
    code,mBasis = param
    for c in range(b +1, lr):
      found = False
      x = (a+1, b+1, c+1)
      if code == 0:
        lhs1 = self.__assocTriads2((x[0], x[2], x[1]), True, basis, mBasis)
        rhs1 = self.__assocTriads2((x[0], x[2], x[1]), False, basis, mBasis)
        lhs2 = self.__assocTriads2((x[0], x[1], x[2]), True, basis, mBasis)
        rhs2 = self.__assocTriads2((x[0], x[1], x[2]), False, basis, mBasis)
        lhs3 = self.__assocTriads2((x[1], x[0], x[2]), True, basis, mBasis)
        rhs3 = self.__assocTriads2((x[1], x[0], x[2]), False, basis, mBasis)
        if lhs1 -rhs1 +lhs2 -rhs2 +lhs3 -rhs3 == 0:
          buf.append(c)
          cnt += 1
      elif code == 1:
        if self.__assocTriads2((x[0], x[2], x[1]), True, basis, mBasis) \
          -self.__assocTriads2((x[0], x[2], x[1]), False, basis, mBasis) == 0:
          buf.append(c)
          cnt += 1
      elif code == 2:
        if self.__assocTriads2((x[0], x[1], x[2]), True, basis, mBasis) \
          -self.__assocTriads2((x[0], x[1], x[2]), False, basis, mBasis) == 0:
          buf.append(c)
          cnt += 1
      elif code == 3:
        if self.__assocTriads2((x[1], x[0], x[2]), True, basis, mBasis) \
          -self.__assocTriads2((x[1], x[0], x[2]), False, basis, mBasis) == 0:
          buf.append(c)
          cnt += 1
    out[a *lr +b] = buf
    return cnt

  def zeroTriads(self, basis, dump=False):
    """zeroTriads(basis, [dump])
       Return all zero divisors (a+b)(c+d) as list of (b,c,d1,d2,...) with
       a=bcd non-scalar and unique where b > c > d range through the table."""
    self.__checkSquare(basis, "zeroTriads", "basis")
    Lib._checkType(dump, bool, "zeroTriads")
    pBasis = list((str(x) for x in basis))
    mBasis = list((x[1:] if x[:1] == "-" else "-" +x for x in pBasis))
    if isinstance(self[0][0], Lib._basestr):
      basis = pBasis
    return Lib.triadPairs(self.__zeroTriads, basis, "zeroTriads", dump, mBasis)
  def __zeroTriads(self, out, basis, lr, b, c, params):
    buf = []
    cnt = 0
    cnt1 = cnt2 = cnt3 = cnt4 = 0
    scalar = lr +1  # Represent +-1 as +-(len(basis) +1)
    bb,cc,mBasis = params
    b1,c1 = b +1, c +1
    for d in range(c1, lr):
      d1 = d +1
      a1 = self.__assocTriads2((b1, c1, d1), True, basis, mBasis)
      aa = abs(a1)
      a0 = aa -1
      if aa != scalar:
        if d not in buf:
          # (a+b)(c+d) = ac +bd +ad +bc
          ac = self.__assocTriads1(a1, c1, basis, mBasis)
          ad = self.__assocTriads1(a1, d1, basis, mBasis)
          bc = self.__assocTriads1(b1, c1, basis, mBasis)
          bd = self.__assocTriads1(b1, d1, basis, mBasis)
          if (ac +bd == 0 and ad +bc == 0) or (ac -bd == 0 and ad -bc == 0):
            addit = True
            acBuf = out[a0 *lr +c if a0 < c else c *lr +a0]
            if b in acBuf or d in acBuf:
              addit = False
              cnt1 += 1
            else:
              bdBuf = out[b *lr +d]
              if a0 in bdBuf or c in bdBuf:
                cnt2 += 1
                addit = False
            if addit:
              buf.append(d)
              cnt += 1
        if a0 not in buf:
          # (d+b)(c+a) = dc +ba +da +bc
          d1 = self.__assocTriads2((b1, c1, a1), True, basis, mBasis)
          dd = abs(d1)
          d0 = dd -1
          dc = self.__assocTriads1(d1, c1, basis, mBasis)
          ba = self.__assocTriads1(b1, a1, basis, mBasis)
          da = self.__assocTriads1(d1, a1, basis, mBasis)
          if (dc +ba == 0 and da +bc == 0) or (dc -ba == 0 and da -bc == 0):
            addit = True
            abBuf = out[a0 *lr +b if a0 < b else b *lr +a0]
            if c in abBuf or d0 in abBuf:
              cnt3 += 1
              addit = False
            else:
              cdBuf = out[c *lr +d0 if c < d0 else d0 *lr +c]
              if a0 in cdBuf or b in cdBuf:
                cnt4 += 1
                addit = False
            if addit:
              buf.append(a0)
              cnt += 1
    out[b *lr +c] = buf
    return(cnt, cnt1, cnt2, cnt3, cnt4)
    #return cnt

  def compare(self, cmp, over=False):
    """compare(cmp,[over])
       Return differences or overlap if over set."""
    Lib._checkList(cmp, None, "compare")
    Lib._checkType(over, bool, "compare")
    out = []
    dups = (not cmp)
    for val in self:
      if val in cmp:
        if over:
          out.append(val)
      else:
        if not over:
          out.append(val)
        if dups:
          cmp.append(val)
    return Tensor(*out)

  def unique(self, dups=False):
    """unique([dups])
       Return unique or duplicate elements if dups set."""
    Lib._checkType(dups, bool, "unique")
    return self.compare([], dups)

  def isomorph(self, basis, perm):
    """isomorph(basis, perm)
       Return self permuted and cells swapped by signed, inverted perm."""
    self.__checkSquare(basis, "isomorph", "basis")
    Lib._checkList(perm, None, "isomorph", len(basis))
    if len(basis) == 0 or len(basis) != len(self):
      raise Exception("Invalid length for isomorph")
    if isinstance(basis[0], Lib._basestr):
      pBasis,mBasis = Lib._basisStrs(basis)
      pBasis = list((pBasis[x-1] if x>0 else mBasis[-x-1] for x in perm))
    else:
      pBasis = list((basis[x-1] if x>0 else -basis[-x-1] for x in perm))
    out = self.permute(perm, True)
    return Tensor(*out.morph(basis, pBasis))

  def permInvert(self):
    """permInvert()
       Return inverted vector permutation."""
    compPerm = list(x+1 for x in range(max(self)))
    for idx,x in enumerate(range(1, len(self) +1)):
      if x in self:
        compPerm[idx] = self.index(x) +1
      elif -x in self:
        compPerm[idx] = -self.index(-x) -1
    return Tensor(compPerm)

  def permute(self, perm, invert=False):
    """permute(perm, [invert])
       Return self with rows and columns swapped and signed by perm."""
    Lib._checkList(perm, None, "permute")
    if max(self.shape) +2 < max(perm):
      raise Exception("Permute parameter length is not valid")
    isStr = isinstance(self[0] if self.shape[1]<2 else self[0][1],
                       Lib._basestr)
    compPerm = Tensor(perm).permInvert() if invert else perm
    rows = []
    for idx in compPerm: # Swap rows
      if idx > 0:
        rows.append(self[idx -1])
      elif isStr:
        val = self[-idx -1][:]
        if self.shape[1] == 1:
          rows.append(val[1:] if val=="-" else "-" +val)
        else:
          rows.append(list((v[1:] if v[0]=="-" else "-" +v for v in val)))
      elif self.shape[1] == 1:
        rows.append(-self[-idx -1])
      else:
        rows.append(list((-v for v in self[-idx -1])))
    out = []  # Swap columns
    if self.shape[1] == 1:
      if len(self) < max(compPerm):
        raise Exception("Invalid permute row length")
      out = rows
    else:
      for idx in range(len(rows)):
        out.append([]) 
        if len(self[idx]) != len(compPerm):
          raise Exception("Invalid permute row length")
      for idx1,row in enumerate(out):
        for idx2 in compPerm:
          if idx2 > 0:
            row.append(rows[idx1][idx2 -1])
          elif isStr:
            val = rows[idx1][-idx2 -1]
            row.append(val[1:] if val[0]=="-" else "-" +val)
          else:
            row.append(-rows[idx1][-idx2 -1])
    return self.copy(out)

  def permCycles(self):
    """permCycles()
       Change permutation into relative cycles as a list."""
    out = []
    tmpPerm = self[:]
    for idx1 in range(len(self)):
      idx2,nxt = idx1,tmpPerm[idx1]
      if idx1 +1 == abs(nxt):
        if nxt < 0:
          out.append([nxt])
      else:
        out1 = [idx1 +1]
        while nxt:
          if abs(nxt) > len(self) or idx2 +1 == abs(nxt):
            raise Exception("Invalid self for permCycles")
          out1.append(nxt)
          tmpPerm[idx2] = 0
          idx2 = abs(nxt) -1
          nxt = tmpPerm[idx2]
          if abs(nxt) == out1[0]:
            if nxt < 0:
              out1[0] = nxt
            tmpPerm[idx2] = 0
            out.append(out1)
            nxt = 0
    return out

  def morphCycles(self, perm, basis=None, tri=False):
    """morphCycles(perm, [basis,tri])
       Return cycle sign permuted by perm as a list from (1,2,3,...)."""
    s0 = []
    iso = []
    for row in self:
      cycLen = 5 if len(row) > 3 else 3
      isoRow = row[:]
      if basis:
        pBasis,mBasis = Lib._basisStrs(basis)
        for idx in range(cycLen):
          tmp = str(isoRow[idx])
          isoRow[idx] = (-mBasis.index(tmp) -1) if tmp[:1]=='-' \
                        else (pBasis.index(tmp) +1)
      #for idx in range(cycLen):
      #  tmp = perm[abs(isoRow[idx]) -1]
      #  isoRow[idx] = tmp *(-1 if row[idx] < 0 else 1)
      while True:
        for idx,neg in {0: (2,3), 1: (2,4)}.items():
          if isoRow[idx] < 0:
            isoRow[idx] *= -1
            if cycLen > 3:
              isoRow[neg[0]] *= -1
              isoRow[neg[1]] *= -1
        cpy = isoRow[:]
        if isoRow[0] > isoRow[1]:
          tmp = isoRow[0]; isoRow[0] = isoRow[1]; isoRow[1] = -tmp
          if cycLen > 3:
            tmp = isoRow[3]; isoRow[3] = isoRow[4]; isoRow[4] = -tmp
        elif isoRow[1] > abs(isoRow[2]):
          tmp = isoRow[1]; isoRow[1] = -isoRow[2]; isoRow[2] = tmp
          if cycLen > 3:
            isoRow[4] = isoRow[1]
        else:
          break
      if tri:
        isoRow = isoRow[:3]
      if basis:
        if isinstance(basis[0], Lib._basestr):
          isoRow = list(pBasis[x-1] if x>0 else mBasis[-x-1] for x in isoRow)
        else:
          isoRow = list(-basis[x-1] if x>0 else basis[-x-1] for x in isoRow)
      iso.append(isoRow)
    return Tensor(*sorted(iso))

  def allSigns(self, half=False, dump=False):
    """allSigns([half,dump])
       Generate a list of all, half [boolean] or a single indexed term [half=
       int] of the signed combinations of self, (eg allSigns(e1)=[e1,-e1]).
       If dump log progress and abort if memory is below the limit."""
    dim = len(self)
    if dim > 0 and 1 not in self.shape:
      raise Exception("Tensor must be a vector for allSigns")
    if dim > 0 and not hasattr(self, "__sub__"):
      raise Exception("Tensor needs to be able to subtract for allSigns")
    stopCnt = -1
    if not isinstance(half, bool):
      Lib._checkType(half, int, "allSigns", (0, pow(2, dim) -1))
      stopCnt = half +1
      half = False
    Lib._checkType(half, bool, "allSigns")
    halfStop = (half and dim %2 == 0)
    halfDim = (int(dim /2) if half else dim)
    halfComb = int(Lib.comb(dim, halfDim) /2)
    for n in range(halfDim +1):
      for cnt,sgns in enumerate(Lib.comb(dim, n, True, dump)): # For n -sign combos
        if n == halfDim and halfStop and cnt == halfComb:
          break
        stopCnt -= 1
        if stopCnt == -1:
          break
        p0 = self[:]
        for sgn in sgns:
          p0[sgn -1] *= -1
        yield p0
      if stopCnt == -1:
        break

  def allSignsIndices(self):
    """allSignsIndices()
       Return index and minus sign count of self in allSigns."""
    dim = len(self)
    if dim > 0 and 1 not in self.shape:
      raise Exception("Tensor must be a vector for allSigns")
    if dim > 0 and not hasattr(self, "__sub__"):
      raise Exception("Tensor needs to be able to subtract for allSigns")
    cnt,sgns = 0,[]
    for idx,term in enumerate(self):
      if term < 0:
        sgns.append(idx +1)
      cnt += 1
    offs = 0
    for dim in range(len(sgns)):
      offs += Lib.comb(cnt, dim)
    for idx,allSgns in enumerate(Lib.comb(cnt, len(sgns), True)):
      if allSgns == sgns:
        break
    return idx +offs, len(sgns)

  ############ Other Creators ############
  @staticmethod
  def Resolution(digits):
    """Resolution([digits])
       Set print format digits or reset to Lib.resolution default."""
    return Lib.resolution(digits)

  @staticmethod
  def NED(lat, lng):
    """BasisNED(lat, lng)
       Lat/long Earth Centred-Earth Fixed (ECEF) basis changed to
       North-East-Down returned as a 3x3 Matrix [NT,ET,DT]T. From
       onlinelibrary.wiley.com/doi/pdf/10.1002/9780470099728.app3.
       This is introduced to check NED() by rotating i, j & k."""
    Lib._checkType(lat, (int, float), "NED")
    Lib._checkType(lng, (int, float), "NED")
    sLat,cLat = Lib._sincos(math.radians(lat))
    sLng,cLng = Lib._sincos(math.radians(lng))
    return Tensor((-cLng *sLat, -sLng *sLat, cLat),
                  (-sLng, cLng, 0),
                  (-cLng *cLat, -sLng *cLat, -sLat))

  @staticmethod
  def Table(basis, rhsBasis=None, lie=0):
    """Table(basis, [rhsBasis, lie])
       Return Matrix mult.or Lie/lie table for basis times rhsBasis or basis."""
    out = []
    if rhsBasis is None:
      rhsBasis = basis
    Lib._checkList(basis, None, "Table")
    Lib._checkList(rhsBasis, None, "Table")
    Lib._checkType(lie, (int, float), "Table")
    if len(basis) > 0 and not hasattr(basis[0], "asym"):
      raise Exception("Table parameter is not a list of basis elements")
    for idx,bas1 in enumerate(rhsBasis):
      row = []
      if lie:
        if lie < 0:
          for bas2 in basis:
            val = abs((bas1.asym(bas2))/lie)
            row.append(val)
        else:
          for bas2 in basis:
            val = (bas1.asym(bas2))/lie
            row.append(val)
      else:
        for bas2 in basis:
          val = bas1 * bas2
          row.append(val)
      out.append(row)
    return Tensor(out)

  @staticmethod
  def Diag(diag):
    """Diag(diag)
       Return the zero matrix with diag as diagonal entries."""
    out = []
    Lib._checkList(diag, None, "Diag")
    for ii in range(len(diag)):
      out.append([0] *len(diag))
      if isinstance(diag[ii], (list, tuple, Matrix)):
        raise Exception("Invalid Diag element")
      out[ii][ii] = diag[ii]
    return Tensor(*out)

  @staticmethod
  def Rotations(vect1, vect2):
    """Rotations(vect1, vect2)
       Return G,P,M for orthonormal vectors vect1 and vect2. Rotation matrix is
       I-P +Pcos(a) +Gsin(a) == (a +Q(*a)*Q(*b)).frameMatrix() for angle a
       == (a/2 +Q(*a)*Q(*b)).versor().eulerMatrix(). Use versor().rotate.
       P is projection into plane of vectors. P=-G*G, P*P=P, N*N=N.
       From Wikipedia matrix exponential: Any matrix X in M(R,nxn): X = A+N 
       where A diagonalisable, N nilpotent N^m = 0, m <= n, A & N commute. Then
                exp(X) = exp(A + N) + exp(A) + exp(N). 
       A = UDU^-1, exp(A) = Uexp(D))U^-1, D = tr(a1,...,an) (diagonal),
            exp(D) = tr(exp(a1),...,exp(an)), det(exp(X)) = exp(tr(X)).
       For P^2 = P, exp(P) = I +(exp(1) -1)*P   by Fourier expansion, so
       R(a) = exp(G*a) = I+G*sin(a)+G*G*(1-cos(a)) = I-P +P*cos(a) +G*sin(a)."""
    I = Matrix.Diag([1.0] *vect1.shape[0])
    G = vect2.dot(vect1.transpose()) - vect1.dot(vect2.transpose())
    P = vect1.dot(vect1.transpose()) + vect2.dot(vect2.transpose())
    M = I - P
    return G,P,M

  @staticmethod
  def Triads(triList, basis, _int=int):
    """Triads(triList, basis)
       Turn triad list into Table using basis of assumed square -1."""
    Lib._checkList(triList, None, "Triads")
    Lib._checkList(basis, None, "Triads")
    isStr = isinstance(basis[0], Lib._basestr) if len(basis) > 0 else False
    if isStr:
      tt = Tensor.Diag(["-1"] *len(basis))
    else:
      tt = Tensor.Diag([-1] *len(basis))
    for tri in triList:
      if not isinstance(tri, (list, tuple, Matrix)) or len(tri) < 3:
        raise Exception("Invalid Triads length: %s" %tri)
      for idx,val in enumerate(tri[:3]):
        Lib._checkType(val, (int, _int), "Triads")
        if abs(val) > len(basis) or val == 0 or (val < 0 and idx < 2):
          raise Exception("Invalid Triads value: %s" %tri)
      if tri[2] < 0:
        tmp = tri[0]; tri[0] = tri[1]; tri[1] = tmp
        tri[2] = -tri[2]
      tt[tri[0]-1][tri[1]-1] = basis[tri[2]-1]
      tt[tri[1]-1][tri[2]-1] = basis[tri[0]-1]
      tt[tri[2]-1][tri[0]-1] = basis[tri[1]-1]
      if isStr:
        tt[tri[1]-1][tri[0]-1] = "-" +basis[tri[2]-1]
        tt[tri[2]-1][tri[1]-1] = "-" +basis[tri[0]-1]
        tt[tri[0]-1][tri[2]-1] = "-" +basis[tri[1]-1]
      else:
        tt[tri[1]-1][tri[0]-1] = -basis[tri[2]-1]
        tt[tri[2]-1][tri[1]-1] = -basis[tri[0]-1]
        tt[tri[0]-1][tri[2]-1] = -basis[tri[1]-1]
    return Tensor(*tt)
 
###############################################################################
## Matrix = numpy if loaded else use a development class based on lists.      #
###############################################################################
if np and "numpy" in sys.modules:
  import numpy   # Rename
  numpy.set_printoptions(edgeitems=30, linewidth=100000)
  class Matrix(numpy.ndarray):
    """Class Matrix interfaces & extends numpy instead of using Tensor class."""
    __transpose = False
    def __init__(self, *args):
      """Matrix(ndarray)
         Define a nx1 or nxm matrix as an numpy.ndarray. Needs Matrix(*list)!"""
      self.__transpose = False
    def __new__(self, *args):
      """Define a nx1 or nxm matrix as a list or list of lists."""
      self.__transpose = False
      arr = numpy.array(args)
      return arr.view(Matrix)

    def __eq__(self, mat):
      """Return True if 2 matricies are equal within precision."""
      if isinstance(mat, numpy.ndarray):
        if (len(self.shape) > 1 and isinstance(self[0][0], Lib._basestr)) or \
           (len(mat.shape) > 1 and isinstance(mat[0][0], Lib._basestr)) or \
           isinstance(self[0], Lib._basestr) or \
           isinstance(mat[0], Lib._basestr):
          return super(Matrix, self).__eq__(mat)
      return ((self - mat) < Lib._getPrecision()).all()

    def get(self, x, y):
      """get(x,y)
         Return the value of a matrix element. Used for FrameMatrix."""
      return self[x][y]

    def transpose(self, axes=None):
      """transpose([axes])
         Return transpose matrix."""
      if len(self.shape) == 1:
        out = self[:]
        out.__transpose = not self.__transpose
        return out
      return super(Matrix, self).transpose(axes)

    def dot(self, b, out=None):
      """dot(b, [out])
         Return inner product or outer product for vector with transpose on RHS."""
      if len(self.shape)==1 and len(b.shape)==1 and b.__transpose:
        return Matrix(*Tensor(list(self)).dot(Tensor(list(b)).transpose()))
      return super(Matrix, self).dot(b,out)

    def copy(self, arr=None):
      """copy(arr)
         Return deep copy at both levels and shape set with optional overwrite."""
      return Matrix(*Tensor(list(self)).copy(arr))

    def sym(self, mat):
      """sym(mat)
         Return self*mat +mat*self."""
      return np.dot(self, mat) +np.dot(mat, self)

    def asym(self, mat):
      """asym(mat)
         Return self*mat -mat*self."""
      return np.dot(self, mat) -np.dot(mat, self)

    def multiply(self, mat):
      """multiply(mat)
         Return Hadamard product matrix."""
      return Matrix(*Tensor(list(self)).multiply(mat))

    def cayleyDicksonMult(self, vector, baezRule=False):
      """cayleyDicksonMult(vector, [baezRule])
         Multiply Tensor pairs using Cayley-Dickson rule, Wikipedia or J.C.Baez:
         q1=(p,q); q2=(r,s); q1*q2 = (pr -s*q, sp +qr*) [wikiRule]
         q1=(p,q); q2=(r,s); q1*q2 = (pr -sq*, p*s +rq) [baezRule]."""
      return Matrix(*Tensor(list(self)).cayleyDicksonMult(vector, baezRule))

    def diag(self, vector=None):
      """diag([vector])
         Return diagonal of square matrix or diagonal of self * vector.transpose
         as vector. Hence trace=sum(matrix.diag()) and dot product is
         sum(v.diag(vector)). This allows Dickson algebra (the product of Real,
         Complex, Quaternion and Octernion numbers) as Tensor(R,Q(1),Q(2), O(3)).
         Addition is via + and multiplication via diag(v)."""
      return Matrix(*Tensor(list(self)).diag(vector))

    def trim(self):
      """trim()
         Return copy with cells smaller than precision set to zero."""
      return Matrix(*Tensor(list(self)).trim())

    def slice(self, offset, shape=None):
      """slice(offset, [shape])
         Return copy of 2-D selection or square if offset,shape are integers."""
      if not isinstance(offset, (list, tuple, Matrix)):
        offset = (offset, offset)
      Lib._checkList(offset, int, "slice", 2)
      if shape is None:
        shape = (self.shape[0], 1 if len(self.shape)==1 else self.shape[1])
      elif not isinstance(shape, (list, tuple, Matrix)):
        shape = (shape, shape)
      Lib._checkList(shape, int, "slice", 2)
      if len(self.shape) ==1:
        return self[offset[0]:shape[0]]
      return self[offset[0]:shape[0], offset[1]:shape[1]]

    def reshape(self, shape):
      """reshape(shape)
         Return copy with new 2-D shape or square if shape is integer."""
      return Matrix(*Tensor(list(self)).reshape(shape))

    def scalar(self):
      """scalar()
         Return copy with non-scalar cells set to the scalar part."""
      return Matrix(*Tensor(list(self)).scalar())

    def reduce(self, fn, init=0):
      """reduce(fn, [init=0])
         Return matrix with x=fn(x[=init],y) applied to each element of self."""
      return self.__reduce(fn, init)

    def __reduce(self, fn, init):
      """Internal reduce(fn) method."""
      out = init
      if len(self.shape) > 1:
        for val1 in range(self.shape[0]):
          for val2 in range(self.shape[1]):
            out = fn(out, self[val1][val2])
        return out
      for val1 in range(self.shape[0]):
        out = fn(out, self[val1])
      return out

    def all(self):
      """all()
         Return True if all matrix values are True else False."""
      return self.__reduce(lambda x,y: x and y, True)

    def any(self):
      """any()
         Return True if any matrix values are True else False."""
      return self.__reduce(lambda x,y: x or y, False)

    def product(self):
      """prod[uct]()
         Return the product of the elements."""
      return Tensor(list(self)).product()
    prod = product

    def function(self, fn):
      """function(fn)
         Return matrix with fn() applied to each element of self."""
      return Matrix(*Tensor(list(self)).function(fn))

    def __function(self, fn):
      """Internal function(fn) method."""
      out = []
      if len(self.shape) > 1:
        for idx1 in range(self.shape[0]):
          out.append([])
          for idx2 in range(self.shape[1]):
            out[idx1].append(fn(self[idx1][idx2]))
        return self.copy(out)
      for idx1 in range(self.shape[0]):
        out.append(fn(self[idx1]))
      return self.copy(out)

    def abs(self):
      """abs()
         Return matrix with absolute value applied to each element of self."""
      return self.function(abs)
    __abs__ = abs

    def pow(self, exp):
      """pow(exp)
         Return matrix with power applied to each element of self."""
      return self.function(lambda x: numpy.pow(x, exp) if isinstance(x, \
                    (numpy.integer, int, float)) else x.__class__.pow(x, exp))
    __pow__ = pow

    def exp(self):
      """exp()
         Return matrix with exponentiation applied to each element of self."""
      return self.function(lambda x: numpy.exp(x) if isinstance(x, \
                    (numpy.integer, int, float)) else x.__class__.exp(x))

    def log(self):
      """log()
         Return matrix with logarithm applied to each element of self."""
      return self.function(lambda x: numpy.log(x) if isinstance(x, \
                    (numpy.integer, int, float)) else x.__class__.log(x))

    def morph(self, basis, labels=None):
      """morph(basis, [labels])
         Return self morphed using a list of pairs or basis->labels. Pairs are
         string names mapped as first->second. Basis & ones are replaced by
         labels and +-1 of labels type so may be of basis or string type."""
      return Matrix(*Tensor(list(self)).morph(basis, labels))

    def differences(self, mat, ignore=None):
      """diff[erences](mat,[ignore])
         Return list of indicies for differences of 2 matricies & ignore value."""
      return Matrix(*Tensor(list(self)).differences(mat, ignore))

    def dump(self, xLabels=[], yLabels=[], name=None):  # Use resolution TBD XXX
      """dump([xLabels,yLabels,name])
         Pretty print of Matrix with optional labels."""
      return Matrix(*Tensor(list(self)).dump(xLabels, yLabels, name))

    def search(self, basis, cf, cfBasis=None, num=-2, diffs=-1, cycles=True,
               initPerm=[], permCycle=False, noAntiIso=False):
      """search(basis, cf, [cfBasis,num,diffs,cycles,initPerm,permCycle])
       Find self in signed permutation of cf for basis optionallly replacing
       cfBasis in cf with basis. All n! permutations and 2**n combinations of
       signs are searched and the histogram of difference counts is returned
       (default num=-2). If num=-1 then return the first match while num>-1
       returns this signed permutation. Otherwise diffs>=0 can be used to
       report each permutation for this many differences or less. Parameter cf
       is swapped as cfBasis -> basis and text searching is quicker. Parameter
       cycle is also quicker (default) because it converts to Tensor.cycles
       and uses morphCycles instead of isomorph. Parameter initPerm is used
       to fix the first part of all permutations such as octonians when
       looking for sedenions. permCycle outputs cycles instead of maps."""
      return Matrix(Tensor(list(self)).search(basis, cf, cfBasis, num, diffs,
                   cycles, initPerm, permCycle, noAntiIso))

    def cycles(self, basis, degree=0, indices=False):
      """cycles(basis, [degree,indices])
         Return a list of multiplication triads for degree using basis."""
      return Matrix(*Tensor(list(self)).cycles(basis, degree, indices))

    def assocTriads(self, basis, nonAssoc=False, alternate=False, dump=False):
      """assocTriads(basis,[nonAssoc,alternate,dump])
         Return assoc traids [a,b,c]=0 or !=0 if nonAssoc. Alternate associativity
         is [a,b,c] = 0 if any two of a,b,c are equal. See Lib.triadDump()."""
      return Matrix(*Tensor(list(self)).assocTriads(basis, nonAssoc, alternate, dump))

    def moufangTriads(self, basis, moufang=0, dump=False):
      """moufangTriads(basis,[moufang,dump])
         Return moufang traids depending on moufang=0-5 where 0 is none and 5 is
         the sum of all four. See Lib.triadDump() for return and other values.
           1: c*(a*(c*b)) -((c*a)*c)*b, 2: a*(c*(b*c)) -((a*c)*b)*c,
           3: (c*a)*(b*c) -(c*(a*b))*c, 4: (c*a)*(b*c) -c*((a*b)*c)."""
      return Tensor(list(self)).moufangTriads(basis, moufang, dump)

    def abcTriads(self, basis, abc=0, nonAssoc=False, dump=False):
      """abcTriads(basis,[abc,nonAssoc,dump])
         Return traids for abc=0-3 where 0 is triple associator and 1-3: [a,c,b],
         [a,b,c], [b,a,c]. See Lib.triadDump() for return and other params."""
      return Tensor(list(self)).abcTriads(basis, abc, nonAssoc, dump)

    def zeroTriads(self, basis, dump=False):
      """zeroTriads(basis, [dump])
         Return all zero divisors (a+b)(c+d) as list of (b,c,d1,d2,...) with
         a=bcd non-scalar and unique where b > c > d range through the table."""
      return Tensor(list(self)).zeroTriads(basis, dump)

    def compare(self, cmp, over=False):
      """compare(cmp,[over])
         Return differences or overlap if over set."""
      return Matrix(*Tensor(list(self)).compare(cmp, over))

    def unique(self, dups=False):
      """unique([dups])
         Return unique or duplicate elements if dups set."""
      return Matrix(*Tensor(list(self)).unique(dups))

    def isomorph(self, basis, perm):
      """isomorph(basis, perm)
         Return self permuted and cells swapped by signed, inverted perm."""
      return Matrix(*Tensor(list(self)).isomorph(basis, perm))

    def permInvert(self):
      """permInvert()
         Return inverted vector permutation."""
      return Matrix(*Tensor(list(self)). permInvert())

    def permute(self, perm, invert=False):
      """permute(perm, [invert])
         Return self with rows and columns swapped and signed by perm."""
      return Matrix(*Tensor(list(self)).permute(self, perm, invert))

    def permCycles(self):
      """permCycles()
         Change permutation into relative cycles as a list."""
      return Matrix(*Tensor(list(self)). permCycles())

    def morphCycles(self, perm, tri=False):
      """morphCycle(perm, [tri])
         Return cycle sign permuted by perm as a list from (1,2,3,...)."""
      return Matrix(*Tensor(list(self)).morphCycles(perm, tri))

    def allSigns(self, half=False, dump=False):
      """allSigns([half,dump])
         Generate a list of all, half [boolean] or a single indexed term [half=
         int] of the signed combinations of self, (eg allSigns(e1)=[e1,-e1]).
         If dump log progress and abort if memory is below the limit."""
      return Matrix(*Tensor(list(self)).allSigns(half, dump))

    def allSignsIndices(self):
      """allSignsIndices()
         Return index and minus sign count of self in allSigns."""
      return Matrix(*Tensor(list(self)).allSigns(half, dump))

     ############ Other Creators ############
    @staticmethod
    def Resolution(digits):
      """Resolution([digits])
         Set print format digits or reset to Lib.resolution default."""
      numpy.set_printoptions(Lib._getResolutions()[0])
      return Lib.resolution(digits)

    @staticmethod
    def NED(lat, lng):
      """BasisNED(lat, lng)
         Lat/long Earth Centred-Earth Fixed (ECEF) basis changed to
         North-East-Down returned as a 3x3 Matrix [NT,ET,DT]T. From
         onlinelibrary.wiley.com/doi/pdf/10.1002/9780470099728.app3.
         This is introduced to check NED() by rotating i, j & k."""
      return Matrix(*Tensor.NED(lat, lng))

    @staticmethod
    def Table(basis, rhsBasis=None, lie=0):
      """Table(basis, [rhsBasis, lie])
         Return Matrix mult.or Lie/lie table for basis times rhsBasis or basis."""
      return Matrix(*Tensor.Table(basis, rhsBasis, lie))

    @staticmethod
    def Diag(diag):
      """Diag(diag)
         Return the zero matrix with diag as diagonal entries."""
      return Matrix(*Tensor.Diag(diag))

    @staticmethod
    def Rotations(vect1, vect2):
      """Rotations(vect1, vect2)
         Return G,P,M for orthonormal vectors vect1 and vect2. Rotation matrix is
         I-P +Pcos(a) +Gsin(a) == (a +Q(*a)*Q(*b)).frameMatrix() for angle a
         == (a/2 +Q(*a)*Q(*b)).versor().eulerMatrix(). Use versor().rotate.
         P is projection into plane of vectors. P=-G*G, P*P=P, N*N=N.
         From Wikipedia matrix exponential: Any matrix X in M(R,nxn): X = A+N 
         where A diagonalisable, N nilpotent N^m = 0, m <= n, A & N commute. Then
                  exp(X) = exp(A + N) + exp(A) + exp(N). 
         A = UDU^-1, exp(A) = Uexp(D))U^-1, D = tr(a1,...,an) (diagonal),
              exp(D) = tr(exp(a1),...,exp(an)), det(exp(X)) = exp(tr(X)).
         For P^2 = P, exp(P) = I +(exp(1) -1)*P   by Fourier expansion, so
         R(a) = exp(G*a) = I+G*sin(a)+G*G*(1-cos(a)) = I-P +P*cos(a) +G*sin(a)."""
      G,P,M = Tensor.Rotations(vect1, vect2)
      return Matrix(*G), Matrix(*P), Matrix(*M)

    @staticmethod
    def Triads(triList, basis):
      """Triads(triList, basis)
         Turn triad list into Table using basis of assumed square -1."""
      return Matrix(*Tensor.Triads(triList, basis, numpy.integer))

    @staticmethod
    def FromNumpy(array):
      """FromNumpy(array)
         Convert from numpy.ndarray to Matrix to process."""
      return array.view(Matrix)
else:
  Matrix = Tensor

################################################################################
class Euler(list):
  """Class to store roll, pitch and yaw and perform Euler Angle rotations for Q.
     Euler angles in higher dimensions (eg SO(4)) is supported for CA & O."""
  __HEX_BASIS   = 15                # Max. rotation matrix rank
  def __init__(self, *args, **kwargs): #roll=0, pitch=0, yaw=0):
    """Euler([<angles>,...][roll,pitch,yaw])
       Create an n-D Euler object for angles as radians with defaults 0."""
    if len(args) < 3:
      if len(args) > 0 and isinstance(args[0], Matrix): # Very strange
        args = args[0]
      args = list(args)
      args.extend([0] *(3 -len(args)))
    super(Euler, self).__init__(args)
    names = ("roll", "pitch", "yaw")
    for param,val in kwargs.items():
      if param not in names:
        raise TypeError("Euler() got unexpected keyword argument %s" %param)
      idx = names.index(param)
      self[idx] = val
    for val in self:
      Lib._checkType(val, (int, float), "Euler")
  def __repr__(self):
    """Overwrite object output using __str__ for print if verbose."""
    if (Lib._isVerbose()):
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __str__(self):
    """Overload string output. Printing taking resolution into account."""
    return "%s" %Tensor(*self)
  def __eq__(self, cf):
    """Return True if 2 Eulers are equal within precision."""
    if not isinstance(cf, Euler) or len(self) != len(cf):
      return False
    precision = Lib._getPrecision()
    for idx,val in enumerate(self):
      if abs(val -cf[idx]) >= precision:
        return False
    return  True
  def copy(self):
    """copy()
       Deep copy of self."""
    return Euler(*self)
  def degrees(self):
    """degrees()
       Return Matrix of euler angles as degrees instead of radians."""
    return Matrix(list(map(math.degrees, self)))

  def trim(self):
    """trim()
       Return copy with elements smaller than precision set to zero."""
    angles = self
    for idx,val in enumerate(self):
      if abs(val) < Lib._getPrecision():
        angles[idx] = 0.0
    return Euler(*angles)

  def matrix(self, order=[], implicit=False, offset=0):
    """matrix([order, implicit, offset])
       Return a Direction Cosine Matrix from Euler angles in extrinsic ZYX
       form from self which is optimised for the default - ie Rxyz: rotate roll
       around x then pitch around y then yaw about z. Opposite of Matrix().
       The rotation order can change and can contain names x, y, z, X, Y, Z or
       roll, pitch, yaw. Four dimensions or more need hexadecimal numbers.
       If explicit then order can't have repeats. If inplicit then subsequent
       rotations use the new rotated axis for rotation so default is Rz''y'x.
       Contents can have an x & y index offset."""
    Lib._checkList(order, None, "Euler.matrix")
    Lib._checkType(implicit, bool, "Euler.matrix")
    rank = len(self)
    if not order:
      if rank == 3 and not implicit and offset == 0:
        sx,cx = Lib._sincos(self[0])
        sy,cy = Lib._sincos(self[1])
        sz,cz = Lib._sincos(self[2])
        return Matrix(\
                 (cz*cy, cz*sy*sx -sz*cx, cz*sy*cx +sz*sx),
                 (sz*cy, sz*sy*sx +cz*cx, sz*sy*cx -cz*sx),
                 (-sy, cy*sx, cy*cx))
      order = list(range(1, rank +1))
    if not (isinstance(order, list) and len(order) in range(1, len(self) +1)):
      raise Exception("Invalid order size for Euler.matrix")
    xyz = [(2,1), (0,2), (1,0)]
    dim = int((math.sqrt(8*rank +1) +1) /2 +0.9) # l=comb(dim,2)
    for j in range(4, dim +1):
      for i in range(1, j):
        if i < 4 or j < 4:
          xyz.append((i -1,j -1))
        else:
          xyz.append((j -1,i -1))  # rotated
    names = ('xXroll', 'yYpitch', 'zZyaw')
    blank = Matrix.Diag([1.0] *(dim +offset))
    mat = blank.copy()
    implicitRot = blank.copy()
    store = []
    for key in order:
      if isinstance(key, Lib._basestr):
        for i,code in enumerate(names):
          if code.find(key) >= 0:
            key = i +1
            break
      if key in store or key not in range(1, len(self) +1):
        raise Exception("Invalid order index for Euler.matrix: %s" %key)
      sw,cw = Lib._sincos(self[key -1])
      idx1,idx2 = xyz[key -1]
      idx1 += offset
      idx2 += offset
      rot = blank.copy()
      rot[idx1][idx1] = cw
      rot[idx1][idx2] = sw
      rot[idx2][idx1] = -sw
      rot[idx2][idx2] = cw
      if implicit:
        tmpRot = rot.copy()
        rot = implicitRot.dot(rot).dot(implicitRot.transpose())
        implicitRot = implicitRot.dot(tmpRot)
      else:
        store.append(key)
      mat = rot.dot(mat)
    return mat

  @staticmethod
  def Matrix(mat, offset=0):
    """Matrix(mat, [offset])
       Create an Euler object from an n-D direction cosine matrix (DCM) in
       extrinsic form (XYZ=Rxyz for 3-D). Opposite of matrix() but sometimes
       angles are out by pi. If pitch is +-90 deg then yaw is arbitrary. Need
       to extract sine terms from the last column at each rank and reduce by
       multiplying by the inverse for each rank until 3-D is reached. Eg 4-D:
       R4[:][4] = (-s14, -s24*c14, -s34*c24*c14, c34*c24*c14), other n-D signs
       alternate apart from when xyz versor indicies are swapped.
       Rzx'z'' roll=atan2(r(3,2),r(3,3))
               pitch=asin(r(3,1))
               yaw=atan2(r(2,1),r(1,1)).
       Rzyx roll=atan2(r(2,3),r(3,3))
            pitch=asin(r(1,3))
            yaw==atan2(r(1,2),r(1,1)).
       Rxy'z''=Z''Y'X, Y=XY'X*, Y=X*Y'X (*=inverse)
       Rz''=Z''=(Y'X)Z(Y'X)*, Z=(Y'X)*Z''(Y'X).
       Rzyx=XYZ=X(X*Y'X)(Y'X)*Z''(Y'X)=Z''Y'X..
       Only handles 3-D matricies. For discriminant see SE(3) transformations at
       https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.468.5407."""
    Lib._checkType(mat, (Matrix, Tensor), "Matrix")
    rank = len(mat) -offset
    blank = Matrix.Diag([1.0] *(rank +offset))
    dim = int((math.sqrt(8*(rank -offset) +1) +1) /2 +0.9) # l=comb(dim,2)
    angles = [0] *(int(Lib.comb(rank, 2)))
    cnt = len(angles)
    for jj in reversed(range(4, rank +1)):
      accum = 1.0
      cnt -= jj -1
      for idx in range(jj -1):
        val = mat[idx +offset][jj -1 +offset]
        sgn = 1 if idx < 3 or jj < 3 else -1 # rotation
        if abs(val) > 1.0:
          raise Exception("Invalid Matrix for Euler""")
        angles[cnt +idx] = math.asin(sgn *val /accum)
        accum *= math.cos(angles[cnt +idx])
      xyz = []
      for ii in range(1, jj):
        if ii < 4 or jj < 4:
          xyz.append((ii -1,jj -1))
        else:
          xyz.append((jj -1,ii -1))  # rotated
      for idx in reversed(range(len(xyz))):
        sw,cw = Lib._sincos(angles[cnt +idx])
        idx1,idx2 = xyz[idx]
        idx1 += offset
        idx2 += offset
        rot = blank.copy()
        rot[idx1][idx1] = cw   # Transpose = inverse to .matrix
        rot[idx1][idx2] = -sw
        rot[idx2][idx1] = sw
        rot[idx2][idx2] = cw
        mat = rot.dot(mat)
    v0 = mat.get(offset, offset)
    v1 = mat.get(offset +1, offset)
    pitch = -math.atan2(mat.get(offset +2, offset), math.sqrt(v0 *v0 +v1 *v1))
    angles[1] = Lib._piRange(pitch)
    if abs(abs(pitch) -math.pi /2) < Lib._getPrecision():
      sgn = -1.0 if pitch < 0 else 1.0
      angles[0] = Lib._piRange(math.atan2(sgn *mat.get(offset +1, offset +2),
                                             sgn *mat.get(offset,offset +2)))
    else:
      angles[0] = Lib._piRange(math.atan2(mat.get(offset +2, offset +1),
                                             mat.get(offset +2, offset +2)))
      angles[2] = Lib._piRange(math.atan2(v1, v0))
    return Euler(*angles)

################################################################################
if __name__ == '__main__': # Might as well run calcR
  import traceback
  from math import *
  from calcR import *
  exp = Lib.exp
  log = Lib.log
  try:
    import importlib
  except:
    importlib = None          # No changing of calculators

  calc = Calculator(Real)
  calc.processInput(sys.argv)

elif sys.version_info.major != 2:  # Python 3
  def execfile(fName):
    """To match Python2's execfile need: from pathlib import Path
       exec(Path(fName).read_text())."""
    exec(Lib.readText(fName))

