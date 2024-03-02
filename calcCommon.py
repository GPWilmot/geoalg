#!/usr/bin/env python
################################################################################
## File: calcCommon.py is the basis for calcR.py and is part of GeoAlg.
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
## Assumes quat.py is in the same directory and this has more documentation.
## Start with either calcR.py, python calcR.py or see ./calcR.py -h.
## This module contains classes (run help for more info):
##   * Common - miscellaneous functions for all calculators
##   * Tensor - simple 1&2-D matricies used for testing and algebra elements
##   * Matrix - interface to numpy if it exists otherwise Tensor is used
##   * Euler  - extended Euler angles (calcQ uses 3 angles, calcCA uses more)
################################################################################
__version__ = "0.3"
import math, sys

################################################################################
class Common():
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
  __precision  = 1E-15          # Precision used for equality
  __resolution  = 3             # Digits for float display
  __resol_form  = r"%0.3G"      # Format string for float display
  __resol_float = r"%0.3f"      # Format string for float display
  __verbose     = False         # Traceback logging
  __info        = False         # User info logging
  __plotFigure  = 0             # Matplotlib unique figures count
  __lastTime    = 0             # Store epoch for time
  __lastProcTime= 0             # Store program time
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
    Common._checkType(a, (int, float), "sincos")
    return Common._sincos(a)

  @staticmethod
  def readText(filename):
    """readTest(filename)
       Local read_text(filename) to return contents of a text file."""
    with open(filename) as fp:
      return fp.read()

  @staticmethod
  def nextFigure():
    """nextFigure()
       Return next Matplotlib unique figure count."""
    Common.__PlotFigures += 1
    return Common.__PlotFigure 

  @staticmethod
  def getMainDoco(filt="test"): #TBD
    """inDoco([filt])
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
    """info se([info])
       Toggle or set info for user reporting."""
    if info is None:
      info = not Common.__info
    Common._checkType(info, bool, "info")
    Common.__info = info
    return verbosity
  @staticmethod
  def verbose(verbosity=None):
    """verbose([verbosity])
       Toggle or set verbosity for traceback reporting."""
    if verbosity is None:
      verbosity = not Common.__verbose
    Common._checkType(verbosity, bool, "verbose")
    Common.__verbose = verbosity
    return verbosity

  @staticmethod
  def resolution(digits=0):
    """resolution([digits])
       Set print format digits or reset to default 17."""
    if not isinstance(digits, int) or digits < 0:
      raise Exception("Invalid printing resolution")
    if digits == 0:
      digits = 17
    if digits < 4:
      digits = 4
    Common.__resolution = digits
    Common.__resol_form = "%%0.%dG" %digits
    Common.__resol_float = "%%0.%df" %digits

  @staticmethod
  def precision(precise=None):
    """precision([precise])
       Set equality precision or reset to default e-15."""
    if precise is None:
      precise = 1E-15
    Common._checkType(precise, float, "precision")
    Common.__precision = precise

  @staticmethod
  def isInfo():
    """isInfo()
       Return true if verbosity is set."""
    return Common.__info
  @staticmethod
  def isVerbose():
    """isVerbose()
       Return true if verbosity is set."""
    return Common.__verbose
  _isVerbose = isVerbose
  @staticmethod
  def getResolNum(val):
    """getResolNum(val)
       Return int or the float rounded to resolution or in exponent format."""
    num = str(val)
    pos = num.find('.')
    if pos < 0 or len(num) -pos -1 < Common.__resolution:
      return num
    else:
      return Common.__resol_float %val
  @staticmethod
  def _checkType(arg, typ, method):
    """Raise exception if not the correct type."""
    if isinstance(arg, bool):
      if not isinstance(False, typ):   # Filter out int for bool or bool in list
        tmp = str(arg)
        if len(tmp) > 9:
          tmp = str(type(arg))
        raise Exception("Invalid parameter type (%s) for %s" %(tmp, method))
    if not isinstance(arg, typ):
      tmp = str(arg)
      if len(tmp) > 9:
        tmp = str(type(arg))
      raise Exception("Invalid parameter type (%s) for %s" %(tmp, method))
  @staticmethod
  def _getResolutions():
    """Internal method to return the digits and print format."""
    return Common.__resolution, Common.__resol_form, Common.__resol_float
  @staticmethod
  def _getPrecision():
    """Internal method to return the precision."""
    return Common.__precision
  @staticmethod
  def _resolutionDump(sign, val, basis):
    """Internal method to return a formated number and basis or blank."""
    out = ""
    num = "%s" %val
    pos = num.find('.')
    if num[0] == '-' and sign:
      sign = " "
    if pos < 0 or len(num) -pos -1 < Common.__resolution:
      if abs(val) == 1 and basis:
        num = "" if val > 0 else "-"
      if val != 0:
        out = "%s%s%s" %(sign, num, basis)
    elif val != 0.0:
      flt = Common.__resol_form %val
      if flt.find(".") < 0:
        flt = Common.__resol_float %val
      resolForm = r"%%s%s%%s" %Common.__resol_form
      out = sign +flt +basis
    return out
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
    out = []
    if arr1 and arr2:
      for ii in arr1:
        for jj in arr2:
          if jj[0] == "-":
            out.append("-" +ii +jj[1:])
          else:
            out.append(ii +jj)
    elif arr1:
      out.extend(arr1)
    else:
      out.extend(arr2)
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
      Common._checkType(pairs[0], (list, tuple), "_unzipBasis")
      Common._checkType(pairs[0][0], Common._basestr, "_unzipBasis")
      Common._checkType(pairs[0][1], int, pairs[0][0])
      for val in args:
        Common._checkType(val, int, pairs[0][0])
        if val > pairs[0][1] or val < 0:
          raise Exception("Invalid %s parameter size" %pairs[0][0])
    return args
  @staticmethod
  def _unzipPairs(dim, pairs):
    """Return pairs or sets of pairs as a permutation list."""
    Common._checkType(pairs, (list, tuple), "_unzipPairs")
    flat = []
    for pair in pairs:
      if isinstance(pair, (list, tuple)):
        if len(flat) %2:
          raise Exception("Pairs index must be even")
        flat.extend(pair)
      else:
        flat.append(pair)
    if len(flat) %2:
      raise Exception("Pairs length must be even")
    if flat and dim < max(flat):
      raise Exception("Pairs dim must be greater then indicies")
    perm = list(x +1 for x in range(dim))
    for idx in range(0, len(flat), 2):
      idx0, idx1 = flat[idx] -1, flat[idx +1] -1
      tmp = perm[idx0]; perm[idx0] = perm[idx1]; perm[idx1] = tmp
    return perm

  @staticmethod
  def _morph(basisNames, value, pairs):
    """Internal utility to perform a single basis swap for each pair in a list
       of basis pair names."""
    Common._checkType(pairs, (list, tuple), "morph")
    if isinstance(basisNames, (list, tuple)):
      basisNames = "".join(basisNames)
    if len(pairs) %2:
      raise Exception("Pairs in morph needs to be of even length")
    morphed = False
    out = {}
    for idx in range(len(pairs) //2):
      x,y = pairs[idx *2:idx *2 +2]
      Common._checkType(x, Common._basestr, "morph")
      Common._checkType(y, Common._basestr, "morph")
      if x in pairs[:idx] or y in pairs[:idx +1]:
        raise Exception("Pairs in morph need to be unique")
      if basisNames == x:
        out[y] = value
        morphed = True
    if not morphed:
      out[basisNames] = value
    return out

  @staticmethod
  def chain(*iterables):
    """chain(list)
       Same as itertools.chain(). Concatenate list of lists or generators."""
    for it in iterables:
      for element in it:
        yield element

  @staticmethod
  def freeMemMB():
    import os
    return os.sysconf('SC_AVPHYS_PAGES')//256

  @staticmethod
  def date(noMs=False):
    """date([noMs=False])
       Return the datetime object for now with str() formated as date_time."""
    import datetime
    now = str(datetime.datetime.today())
    if noMs:
      idx = now.find(".")
      now = now[:idx]
    return now

  @staticmethod
  def time(epoch=False):
    """time([epoch])
       Return seconds since epoch or difference to previous call as a float."""
    import time
    if Common.__lastTime == 0 or epoch:
      Common.__lastTime = time.time()
    lastTime = Common.__lastTime
    Common.__lastTime = time.time()
    return Common.__lastTime -(0 if epoch else lastTime)

  @staticmethod
  def procTime(start=False):
    """procTime([start])
       Return program user+sys seconds since start or diff. to previous call."""
    import time
    if Common.__lastProcTime == 0 or start:
      Common.__lastProcTime = time.process_time()
    lastTime = Common.__lastProcTime
    Common.__lastProcTime = time.process_time()
    return Common.__lastProcTime -(0 if start else lastTime)

  @staticmethod
  def comb(n, r, perms=False, dump=False):
    """comb(n, r, [perms])
       Return number of combinations of r in n or perm(n,r) set generator."""
    Common._checkType(n, int, "comb")
    Common._checkType(r, int, "comb")
    Common._checkType(perms, bool, "comb")
    Common._checkType(dump, bool, "comb")
    if n < r:
      raise Exception("Invalid parameter for comb(%s,%s)" %(n, r))
    if perms:
      if dump:
        dump = [r,0]
      return Common.__perm(n, [1] *r, 1, dump)
    return math.factorial(n) /math.factorial(n-r) /math.factorial(r) 
  
  @staticmethod
  def perm(n, r=None):
    """perm(n, [r])
       Return the number of permutations of n elements or set of r in n perms
       as a generator."""
    Common._checkType(n, int, "perm")
    if isinstance(r, int) and not isinstance(r, bool):
      return Common.__perm(n, [1] *r, 0, None)
    elif r is None:
      return math.factorial(n) 
    raise Exception("Invalid r parameter for perm")

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
          for more in Common.__perm(n, arr[1:], recuse if offset else 0, dump):
            arr = [recuse] +more
            dup = False
            for idx,elem in enumerate(arr):
              if elem in arr[idx+1:]:
                dup = True
                break
            if dump and len(arr) == dump[0] and arr[0] != dump[1]:
              dump[1] = arr[0]
              sys.stdout.write("%s %s %s\n" %(Common.date(), dup, arr))
            if not dup:
              yield arr
        else:
          yield [recuse]

  @staticmethod
  def additionTree(dim, split, maxs=()):
    """additionTree(dim,split, [maxs])
       Split dim into split parts with options maxs list."""
    Common._checkType(dim, int, "additionTree")
    Common._checkType(split, int, "additionTree")
    Common._checkType(maxs, (list, tuple), "additionTree")
    if dim < 1 or split < 2:
      raise Exception("additionTree has invalid dim or split size")
    if maxs and len(maxs) != split:
      raise Exception("additionTree has invalid maxs lens")
    for val in maxs:
      Common._checkType(val, int, "additionTree")
      if val < 0:
        raise Exception("additionTree has invalid maxs value")
    out = []
    if split == 2:
      for i in range(dim +1):
        if maxs[0] >= dim-i and maxs[1] >= i:
          out.append([dim-i, i])
    else:
      splits = [0] *split
      splits[0] = dim
      Common.__additionTree(splits, out, maxs, 0)
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
      out.append(splits)
    splits = splits[:]
    if splits[idx] > 0:
      splits[idx] -= 1
      if idx < 2:
        splits[idx +1] += 1
        Common.__additionTree(splits, out, maxs, idx)
        if idx < 1:
          splits = splits[:]
          splits[idx +1] -= 1
          splits[idx +2] += 1
          Common.__additionTree(splits, out, maxs, idx +1)
      elif idx < 1:
        splits[idx +2] += 1
        Common.__additionTree(splits, out, maxs, idx +1)
    
  @staticmethod
  def triads(dim):
    """triads(dim)
       Generate a list of all independent triads for a basis of dim > 2."""
    Common._checkType(dim, int, "triads")
    if dim < 3:
      raise Exception("Invalid triad dimension")
    pAll = list(Common.comb(dim, 3, True))
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
      for triads in Common.comb(len(pAll), facesLen, True):
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
  def assocTriads(basis, code=0, nonAssoc=False, all=False, dump=False):
    """assocTriads(basis,[code=0,all=False,dump=False])
       Return assoc traids [a,b,c] or [b,c,d] or [c,b,d]=0 for code=1-3 where
       a=b*c*d is pure, code=4 returns scalars, code=5-8 returns moufang != 0
       for types=1-4, code=0,9 prints abc,moufang counts only. nonAssoc inverts
       associative test, all ranges c and d over whole range and dump prints
       progress and aborts if memory limit is exceeded."""
    Common._checkType(basis, (list,tuple), "assocTriads")
    Common._checkType(code, int, "assocTriads")
    Common._checkType(nonAssoc, bool, "assocTriads")
    Common._checkType(all, bool, "assocTriads")
    Common._checkType(dump, bool, "assocTriads")
    if code not in range(10):
      raise Exception("Invalid code in assocTriads")
    lr = len(basis)
    total = pairs = soles = 0
    acnt = bcnt = ccnt = scnt = 0
    mcnt = [0] *4
    Common.time()
    out = []
    for b in range(lr):
      if dump and b %10 == 0:
        sys.stdout.write("%s %d %d..%d\n" %(Common.date(True),
                         int(Common.time()), b, lr))
        if Common.freeMemMB() < Common._memLimitMB:
          sys.stdout.write("ABORT: Memory limit reached\n")
          break
      cr = 0 if all else b +1
      for c in range(cr, lr):
        bb,cc = basis[b], basis[c]
        dr = 0 if all else c +1
        for d in range(dr, lr):
          dd = basis[d]
          aa = bb *cc *dd
          buf = (bb, cc, dd)
          total += 1
          if all and (b==c or c==d or d==b):
            if b==c and c==d:
              soles += 1
            else:
              pairs += 1
          elif aa.isScalar():
              scnt += 1
              if code == 4:
                out.append(buf)
          elif code > 4:
            if code == 9:
              for num in range(4):
                nass = (bb.moufang(cc,dd,num +1) == 0)
                if (nass and nonAssoc) or not (nass or nonAssoc):
                  mcnt[num] += 1
            else:
              nass = (bb.moufang(cc, dd, code -4) == 0)
              if (nass and nonAssoc) or not (nass or nonAssoc):
                out.append((bb,cc,dd))
          else:
            nass = (aa.assoc(bb,cc) != 0)
            if (nass and nonAssoc) or not (nass or nonAssoc):
              acnt += 1
              if code == 1:
                out.append(buf)
            nass = (bb.assoc(aa,cc) != 0)
            if (nass and nonAssoc) or not (nass or nonAssoc):
              bcnt += 1
              if code == 2:
                out.append(buf)
            nass = (cc.assoc(bb,dd) != 0)
            if (nass and nonAssoc) or not (nass or nonAssoc):
              ccnt += 1
              if code == 3:
                out.append(buf)
    if code == 0:
      sys.stdout.write("total=%d, %sscalar=%d, a#=%d, b#=%d, c#=%d\n"
          %(total, ("pairs=%d, soles=%d, " %(pairs, soles) if all else ""),
                  scnt, acnt, bcnt, ccnt))
    elif code == 9:
      sys.stdout.write("total=%d, %sscalar=%d, m#1=%d, m#2=%d, m#3=%d, m#4=%d\n"
          %(total, ("pairs=%d, soles=%d, " %(pairs, soles) if all else ""),
                  scnt, mcnt[0], mcnt[1], mcnt[2], mcnt[3]))
    return out

################################################################################
class Tensor(list):
  """Class for development and test instead of using numpy. May contain basis
     numbers like i, j, k. These are tensors since they may contain basis
     vectors. If g_lo = Tensor(e0,e1,e2,e3) and g_hi = Tensor(-e0,e1,e2,e3) then
     metric tensor is (g_lo*g_hi.transpose()).scalar() = Tensor.Diag([1]*4)."""
     
  def __init__(self, *args):
    """Tensor(list)
       Define a nx1 or nxm matrix as a list or list of lists."""
    super(Tensor, self).__init__()
    Common._checkType(args, (list, tuple), "Tensor")
    if not args:
      self.__size = (0, 0)
    else:
      if not args or len(args) == 1 and not args[0]:
        args = (0,)
      self.__size = (len(args), 1)
      if isinstance(args[0], (list, tuple)):
        if len(args) == 1:
          args = args[0]
          self.__size = (len(args), 1)
        if isinstance(args[0], (list, tuple)):
          self.__size = (len(args), len(args[0]))
    for row in args:
      if isinstance(row, (list, tuple)):
        if len(row) == 1 and self.__size[1] == 1:
          self.append(row[0])
        else:
          self.append(list(row))
          if len(self[-1]) != self.__size[1]:
            raise Exception("Inconsistant Tensor row lengths")
      else:
        self.append(row)

  def get(self, x, y):
    """get(x,y)
       Return the value of a matrix element. Used for FrameMatrix."""
    return self[x][y]

  def shape(self):
    """size()
       Return the dimensions of a matrix. Used for FrameMatrix."""
    return self.__size

  def __str__(self):
    """Overload standard string output."""
    out = "(" if self else "( \n"
    sep = ""
    for row in self:
      if isinstance(row, list):
        sep = "("
        for col in row:
          if isinstance(col, float):
            out += sep +Common.getResolNum(col)
          else:
            out += sep +str(col)
          sep = ", "
        out += '),\n'
      else:
        if isinstance(row, float):
          out += sep +Common.getResolNum(row)
        else:
          out += sep +str(row)
        sep = ", "
    return (out[:-2] +')') if out[-1]=='\n' else out +')'

  def __eq__(self, mat):
    """Return True if 2 matricies are equal taking resolution into account."""
    if not isinstance(mat, Tensor):
      mat = Tensor(mat).reshape(self.__size)
    if self.__size != mat.__size:
      return False
    precision = Common._getPrecision()
    for idx1,val1 in enumerate(self):
      if isinstance(val1, (list, tuple)):
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
    """Matrix multiplication for ranks 3x3 * 3x3 and 3x3 * 3x1."""
    if isinstance(mat, (list, tuple)) and not isinstance(mat, Tensor):
      mat = Tensor(*mat)
    a = []
    if isinstance(mat, Tensor):
      if self.__size[1] != mat.__size[0]:
        raise Exception("Invalid Matrix sizes for multiplying: %sx%s" \
                     %(self.__size, mat.__size))
      if self and isinstance(self[0], (list, tuple)): # self.__size > (1,1)
        if mat.__size[1] > 1:                         # matrix * matrix
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
      elif isinstance(mat[0], (list, tuple)):        # mat.__size > (1,1)
        for row,val1 in enumerate(mat):              # vector * matrix
          a.append(self[0] *val1[0])
          for col,val2 in enumerate(val1[1:]):
            a[row] += self[col +1] *val2
      elif self.__size[1] != mat.__size[0]:
        raise Exception("Invalid Matrix sizes for product: %sx%s" \
                     %(self.__size, mat.__size))
      elif self.__size[0] == 1:
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
      if isinstance(val1, (list, tuple)):
        a.append([None] *len(val1))
        for col,val2 in enumerate(val1):
          if isinstance(mat, Common._basestr):
            a[row][col] = self.__mulStr(str(val2), mat)
          elif isinstance(val2, Common._basestr):
            a[row][col] = self.__mulStr(val2, str(mat))
          else:
            a[row][col] = val2 *mat
      elif isinstance(mat, Common._basestr):
        a.append(self.__mulStr(str(val1), mat))
      elif isinstance(val1, Common._basestr):
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
          if isinstance(mat, Common._basestr):
            a[row][col] = self.__mulStr(str(val2), mat)
          elif isinstance(val2, Common._basestr):
            a[row][col] = self.__mulStr(val2, str(mat))
          else:
            a[row][col] = mat * val2
      elif isinstance(mat, Common._basestr):
        a.append(self.__mulStr(mat, val1))
      elif isinstance(val1, Common._basestr):
        a.append(self.__mulStr(mat, val1))
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
    if isinstance(den, (list, tuple)):
      raise Exception("Invalid Matrix division")
    a = []
    for row,val1 in enumerate(self):
      if isinstance(val1, (list, tuple)):
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
    if isinstance(mat, (list, tuple)) and not isinstance(mat, Tensor):
      mat = Tensor(*mat)
    Common._checkType(mat, Tensor, "add")
    if self.__size != mat.__size:
      raise Exception("Invalid Matrix size for add/sub")
    if self and isinstance(self[0], (list, tuple)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          tmp = mat[idx1][idx2]
          if isinstance(val2, Common._basestr) or \
             isinstance(tmp, Common._basestr):
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
    return self.__mul__(-1)
  def __sub__(self, mat):
    """Subtract 2 matricies."""
    if isinstance(mat, (list, tuple)) and not isinstance(mat, Tensor):
      mat = Tensor(*mat)
    Common._checkType(mat, Tensor, "sub")
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
    if self and isinstance(self[0], (list, tuple)):
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
    Common._checkType(arr, (list, tuple), "copy")
    if arr and isinstance(arr[0], (list, tuple)):  # arr.__size > (1,1)
      return Tensor([row[:] for row in arr])
    out = Tensor(*arr)
    out.__size = self.__size  # Keep transpose
    return out

  def sym(self, mat):
    """Return self*mat +mat*self."""
    return self *mat +mat *self

  def asym(self, mat):
    """Return self*mat -mat*self."""
    return self *mat -mat *self

  def diag(self, vector=None):
    """diag([vector])
       Return diagonal of square matrix or diagonal of self * vector.transpose
       as vector. Hence trace=sum(matrix.diag()) and dot product is
       sum(v.diag(vector)). This allows Dickson algebra (the product of Real,
       Complex, Quaternion and Octernion numbers) as Tensor(R,Q(1),Q(2), O(3)).
       Addition is via + and multiplication via diag(v)."""
    out = []
    shape = self.shape()
    if vector is None:
      if shape[0] != shape[1]:
        raise Exception("Matrix for diag must be square")
      for idx in range(shape[0]):
        out.append(self[idx][idx])
    else:
      Common._checkType(vector, (list, tuple) , "diag")
      if len(self) != len(vector):
        raise Exception("Vectors for diag must be the same length")
      for idx,val in enumerate(self):
        out.append(val *vector[idx])
    return Tensor(*out)

  def multiply(self, mat):
    """Return Hadamard product matrix."""
    out = []
    if isinstance(mat, (list, tuple)) and not isinstance(mat, Tensor):
      mat = Tensor(*mat)
    Common._checkType(mat, Tensor, "multiply")
    if self.__size != mat.__size:
      raise Exception("Invalid Matrix sizes for multiply %sx%s" \
                     %(self.__size, mat.__size))
    if self and isinstance(self[0], (list, tuple)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          out[idx1].append(val2 *mat[idx1][idx2])
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(val1 *mat[idx1])
    return self.copy(out)

  def cayleyDicksonMult(self, vector, wikiRule=False):
    """cayleyDicksonMult(vector, [wikiRule])
       Multiply Tensor pairs using Cayley-Dickson rule, Wikipedia or J.C.Baez:
       q1=(p,q); q2=(r,s); q1*q2 = (pr -s*q, sp +qr*) [wikiRule]
       q1=(p,q); q2=(r,s); q1*q2 = (pr -sq*, p*s +rq) [baezRule]."""
    out = []
    Common._checkType(vector, (list, tuple) , "bioct")
    if len(self) != len(vector) or len(self) != 2:
      raise Exception("Vectors for bioct must have length two")
    if self and not isinstance(self[0], (list, tuple)):
      if wikiRule:
        out.append(self[0] *vector[0] -vector[1].conjugate() *self[1])
        out.append(vector[1] *self[0] +self[1] *vector[0].conjugate())
      else:
        out.append(self[0] *vector[0] -vector[1] *self[1].conjugate())
        out.append(self[0].conjugate() *vector[1] +vector[0] *self[1])
      return Tensor(*out)
    raise Exception("Self for bioct needs to be a vector")

  def trim(self):
    """trim()
       Return copy with cells smaller than precision set to zero."""
    out = []
    if self and isinstance(self[0], (list, tuple)):  # self.__size > (1,1)
      for idx1 in range(len(self[0])):
        out.append([])
      for idx1,val1 in enumerate(self):
        for val2 in val1:
          out[idx1].append(0.0 if abs(val2) < Common._getPrecision() else val2)
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(0.0 if abs(val1) < Common._getPrecision() else val1)
    return self.copy(out)

  def slice(self, offset, shape=None):
    """slice(offset, [size]):
       Return copy of 2-D selection or square if offset, shape are integers."""
    if not isinstance(offset, (list, tuple)):
      offset = (offset, offset)
    Common._checkType(offset, (list, tuple), "slice")
    if len(offset) != 2:
      raise Exception("Invalid offset length in slice")
    Common._checkType(offset[0], int, "slice")
    Common._checkType(offset[1], int, "slice")
    if shape is not None:
      if not isinstance(shape, (list, tuple)):
        shape = (shape, shape)
      Common._checkType(shape, (list, tuple), "slice")
      if len(shape) != 2:
        raise Exception("Invalid shape length in slice")
      Common._checkType(shape[0], int, "slice")
      Common._checkType(shape[1], int, "slice")
    out = []
    if self and isinstance(self[0], (list, tuple)):  # self.__size > (1,1)
      if shape is None:
        shape = (self.__size[0] -offset[0], self.__size[1] -offset[1])
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
      shape = (self.__size[0] -offset[0], )
    out = self[offset[0]:offset[0] +shape[0]]
    out = Tensor(*out)
    if self.__size[1] > 1:
      out.__size = (1, out.size[0])
    return out

  def reshape(self, shape):
    """reshape(shape)
       Return copy with new 2-D shape or square if shape is integer."""
    if not isinstance(shape, (list, tuple)):
      shape = (shape, shape)
    Common._checkType(shape, (list, tuple), "reshape")
    Common._checkType(shape[0], int, "reshape")
    if len(shape) != 2:
      raise Exception("Invalid shape length in reshape")
    out = []
    if self and isinstance(self[0], (list, tuple)):  # self.__size > (1,1)
      Common._checkType(shape[1], int, "reshape")
      for idx1 in range(max(1,shape[0])):
        out.append([])
      for idx1 in range(max(1,shape[0])):
        val1 = self[idx1] if idx1 < len(self[0]) else []
        for idx2 in range(shape[1]):
          out[idx1].append(val1[idx2] if idx2 < len(val1) else 0)
      return Tensor(*out)
    if shape[0] > 1 and shape[1] > 1:
      for idx1 in range(shape[0]):
        out.append([0] *shape[1])
    else:
      self.__size = (shape[0], shape[1])
      return Tensor(self)
    for idx,val in enumerate(self):
      if shape[0] > 1 and shape[1] > 1:
        if self.__size[0] > 1:
          out[idx][0] = val
        else:
          out[0][idx] = val
      else:
        out.append(val)
      if idx >= max(shape[0], shape[1]):
        break
    out = Tensor(*out)
    out.__size = (shape[0], shape[1])
    return out

  def scalar(self):
    """scalar()
       Return copy with non-scalar cells set to the scalar part."""
    out = []
    if self and isinstance(self[0], (list, tuple)):  # self.__size > (1,1)
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
    if self and isinstance(self[0], (list, tuple)):  # self.__size > (1,1)
      out = []
      for idx1 in range(len(self[0])):
        out.append([])
      for idx1,val1 in enumerate(self):
        for idx2,val2 in enumerate(val1):
          out[idx2].append(val2)
      return Tensor(*out)
    out = Tensor(self)
    out.__size = (self.__size[1], self.__size[0])
    return out

  def pow(self, exp):
    """pow(exp)
       Return matrix with  power applied to each element of self."""
    Common._checkType(exp, (int, float), "pow")
    out = []
    if self and isinstance(self[0], (list, tuple)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          out[idx2].append(pow(val2, exp))
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      out.append(pow(val1, exp))
    return self.copy(out)
  __pow__ = pow

  def exp(self):
    """exp()
       Return matrix with exponentiation applied to each element of self."""
    out = []
    if self and isinstance(self[0], (list, tuple)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          if isinstance(val2, (float, int)):
            out[idx2].append(exp(val2))
          else:
            out[idx2].append(exp(val2))
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      if isinstance(val1, (float, int)):
        out.append(val1.exp())
      else:
        out.append(val1.exp())
    return self.copy(out)

  def log(self):
    """log()
       Return matrix with logarithm applied to each element of self."""
    out = []
    if self and isinstance(self[0], (list, tuple)):
      for idx1,val1 in enumerate(self):
        out.append([])
        for idx2,val2 in enumerate(val1):
          if isinstance(val2, (float, int)):
            out[idx2].append(log(val2))
          else:
            out[idx2].append(val2.log())
      return Tensor(*out)
    for idx1,val1 in enumerate(self):
      if isinstance(val1, (float, int)):
        out.append(log(val1))
      else:
        out.append(val1.log())
    return self.copy(out)

  def morph(self, basis, labels=None):
    """morph(basis, [labels])
       Return self morphed using a list of pairs or basis->labels. Pairs are
       string names mapped as first->second. Basis & ones are replaced by
       labels and +-1 of labels type so may be of basis or string type."""
    Common._checkType(basis, (list, tuple), "morph")
    if labels is None:
      out = []
      if self and isinstance(self[0], (list, tuple)):  # self.__size > (1,1)
        for idx1,val1 in enumerate(self):
          out.append([])
          for val2 in val1:
            out.append(Common._morph(val2, 1, basis)[val2])
        return Tensor(*out)
      for idx1,val1 in enumerate(self):
        out.append(Common._morph(val1, 1, basis)[val1])
      return self.copy(out)
    Common._checkType(labels, (list, tuple), "morph")
    if len(basis) != len(labels) or len(self) < len(basis):
      raise Exception("Swap length is not valid")
    val1 = self[0]
    if isinstance(val1, (list, tuple)) and len(val1) > 0:
      for val2 in val1[:]:
        if type(val1) == type(basis[0]):
          break
        val1 = val2
    if type(val1) != type(basis[0]):
      raise Exception("Invalid swap basis type: %s !~ %s" %(type(val1),
                       type(basis[0])))
    return self.__morph(basis, labels)

  def __morph(self, basis, labels):
    """Internal routine to return self with basis & ones replaced by labels
       and +-1 of labels type."""
    isStrBasis = isinstance(basis[0], Common._basestr)
    if isStrBasis:
      pBasis = (str(x) for x in basis)
      mBasis = list((x[1:] if x[0]=="-" else "-"+x for x in pBasis))
    else:
      mBasis = list((-x for x in basis))
    isStrLabel = isinstance(labels[0], Common._basestr)
    if isStrLabel:
      pLabels = (str(x) for x in labels)
      mLabels = list((x[1:] if x[0]=="-" else "-"+x for x in pLabels))
    else:
      mLabels = list((-x for x in labels))
    out = []
    for val1 in self:
      row = []
      if isinstance(val1, (list, tuple)):
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
              raise Exception("Swap element not found in basis: %s" %val2)
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
    """differences(mat,[ignore])
       Return list of indicies for differences of 2 matricies & ignore value."""
    Common._checkType(mat, (list, tuple), "diff")
    out = []
    isStr = (len(self) > 0 and isinstance(self[0], Common._basestr))
    for idx1,val1 in enumerate(self):
      if isinstance(val1, (list, tuple)):
        isStr = (len(self[0]) > 0 and isinstance(self[0][0], Common._basestr))
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

  def dump(self, xLabels=[], yLabels=[], name=""):  # Use resolution TBD XXX
    """dump([xLabels,yLabels,name])
       Pretty print of Matrix with optional labels."""
    s = t = 0
    if xLabels:
      Common._checkType(xLabels, (list, tuple), "dump")
      if not self or len(xLabels) != self.__size[1]:
        raise Exception("Invalid xLabels length for Matrix dump")
      if not yLabels and len(xLabels) == self.__size[0]:
        yLabels = xLabels
      s = max(map(lambda x :len(str(x)), xLabels))
    if yLabels:
      Common._checkType(yLabels, (list, tuple), "dump")
      if len(yLabels) != self.__size[0]:
        raise Exception("Invalid yLabels length for Matrix dump")
      t = max(map(lambda x :len(str(x)), yLabels))
    if isinstance(self, list) and len(self) > 0 and isinstance(self[0], list):
      size = max(map(max, (map(lambda x: len(str(x)), row) for row in self)))
    else:
      size = max(map(lambda x :len(str(x)), self))
    formX = " %%%ds" %(max(s,size))
    nameX = "%%-%ds" %(max(t,len(str(name))) +3)
    formY = " %%%ds" %max(t,len(str(name)))
    if xLabels or name:
      sys.stdout.write(nameX %str(name) +"%s\n" %"".join(formX %x for x in xLabels))
      if xLabels:
        sys.stdout.write(formY %"" +'-' *(len(xLabels) *(max(s,size)+1)+2)+'\n')
    for ii,vals in enumerate(self):
      if yLabels:
        sys.stdout.write(formY %yLabels[ii] +"| ")
      if self.__size[1] == 1:
        sys.stdout.write(formX %vals)
      elif self.__size[0] == 1:
        for val in self:
          sys.stdout.write(formX %val)
        sys.stdout.write("\n")
        break
      else:
        for val in vals:
          sys.stdout.write(formX %val)
      sys.stdout.write("\n")

  def __checkSquare(self, basis, name, basisName):
    """Internal method to raise exception for invalid Tensor."""
    Common._checkType(basis, (list, tuple), name)
    if len(self) != len(basis) or len(self) == 0:
      raise Exception("Parameter %s length invalid for %s" \
                       %(basisName, name))
    if self.__size[0] != self.__size[1]:
      raise Exception("Tensor is not square for %s" %name)
    if hasattr(self[0][0], "grades") != hasattr(basis[0], "grades"):
      raise Exception("Tensor is not the same type as basis for %s" %name)

  def search(self, basis, cf, cfBasis=None, num=-2, diffs=-1, cycles=True, initPerm=[]):
    """search(basis, cf, [cfBasis,num,diffs,cycles,initPerm])
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
       looking for sedenions."""
    self.__checkSquare(basis, "search", "basis")
    Common._checkType(cf, (list, tuple), "search")
    cf = Tensor(*cf)
    if cfBasis:
      cf.__checkSquare(cfBasis, "search", "cf")
    Common._checkType(num, int, "search")
    Common._checkType(diffs, int, "search")
    Common._checkType(cycles, bool, "search")
    Common._checkType(initPerm, (list, tuple), "search")
    chkPerm = list(x for x in sorted(list(abs(y) for y in initPerm)))
    for idx in range(len(initPerm)):
      if chkPerm[idx] != idx +1:
        raise Exception("Search initPerm needs numbers %s" %range(len(initPerm)))
    dim = len(basis)
    val1 = self[0]
    if isinstance(val1, (list, tuple)) and len(val1) > 0:
      for val2 in val1[:]:
        if type(val1) == type(basis[0]):
          break
        val1 = val2
    if type(val1) != type(basis[0]):
      raise Exception("Invalid search basis type: %s !~ %s" %(type(val1),
                       type(basis[0])))
    if cfBasis:
      if dim != len(cfBasis):
        raise Exception("Search cfBasis length is not valid")
      if not cycles:
        cf = cf.morph(cfBasis, basis)
    dim -= len(initPerm) 
    perms = Common.perm(dim, dim)
    difHisto = {}
    difRange = [99999, 0]
    isStr = isinstance(basis[0], Common._basestr)
    if isStr:
      mBasis = list((x[1:] if x[:1] == "-" else ("-" +x) for x in basis))
    else:
      mBasis = list((-x for x in basis))
    if cycles:
      cf = cf.cycles(cfBasis if cfBasis else basis)
      cycleIso = self.cycles(basis)
      if len(cf) < len(self) or len(cycleIso) < len(self):
        print(len(cf), len(self), len(cycleIso),cf)
        raise Exception("Invalid basis for table.cycles()")
    cnt = 0
    for p in perms:                # For all permutations
      for n in range(dim +1):      # For all negative sign combinations
        for sgns in Common.comb(dim, n, True):
          p0 = list(x +len(initPerm) for x in p)
          for sgn in sgns:
            p0[sgn -1] *= -1
          p0 = initPerm +p0

          # iso = self.isomorph(basis, p0)   # Signed swap rows, columns & cells
          if cycles:
            iso = cycleIso.morphCycles(p0)
          else:
            pBasis = list((basis[x-1] if x>0 else mBasis[-x-1] for x in p0))
            morphed = self.__morph(basis, pBasis)
            iso = morphed.permute(p0, True)

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
          if diffs >= 0:
            if len(dif) == 0: # iso == cf
              sys.stdout.write("FOUND at %d %s\n" %(cnt, p0))
            elif len(dif) <= diffs:
              sys.stdout.write("DIFFS at %d %s has %d: %s\n" %(cnt, p0, len(dif), dif))
          if cnt == num or (num == -1 and len(dif) == 0):
            if diffs < 0:
              sys.stdout.write("%s at %d %s\n" \
                   %("FOUND" if num == -1 else "GOT", cnt, p0))
            return iso
          cnt += 1
    i = difRange[0]
    p0 = [0] *i
    while i < difRange[1] +1:
      j = difHisto[i] if i in difHisto else 0
      p0.append(j)
      i += 1
    if num == -1:
      sys.stdout.write("NOT FOUND for %d: %s\n" %(cnt, p0))
    return p0
  
  def cycles(self, basis, degree=0, dump=False, indicies=True):
    """cycles(basis, [grade,dump])
       Return or dump a list of multiplication triads for degree using basis."""
    self.__checkSquare(basis, "cycles", "basis")
    Common._checkType(dump, bool, "cycles")
    if degree and not hasattr(basis[0], "grades"):
      raise Exception("Parameter grade for cycles needs graded basis")
    pBasis = list((str(x) for x in basis))
    mBasis = list((x[1:] if x[:1] == "-" else "-" +x for x in pBasis))
    prod = {}
    out = []
    for i1 in range(len(basis)):
      for i2 in range(i1 +1, len(basis)):
        p1 = pBasis[i1]
        p2 = pBasis[i2]
        p3 = self[i1][i2]  # p1 * p2
        isGrade = (not degree or p3.grades(degree)[degree])
        p3 = str(p3)
        if p1 not in prod:
          prod[p1] = []
          prod[mBasis[i1]] = []
        if p2 not in prod:
          prod[p2] = []
          prod[mBasis[i2]] = []
        sgn3 = "-" if p3 in mBasis else ""
        i3 = mBasis.index(p3) if sgn3 else pBasis.index(p3)
        mp3 = pBasis[i3] if sgn3 else mBasis[i3]
        if p3 not in prod:
          prod[p3] = []
          prod[mp3] = []
        if not (p1 in prod[p2] or p1 in prod[p3] \
             or p2 in prod[p1] or p2 in prod[p3] \
             or p3 in prod[p1] or p3 in prod[p2]) and isGrade:
          p4 = sgn3 +str(self[i2][i3])  #p2 *p3
          p4 = p4[2:] if p4[:2] == "--" else p4
          sgn4 = "-" if p4 in mBasis else ""
          if p4 == p1 or p4 == mBasis[i1]:
            i4 = mBasis.index(p4) if sgn4 else pBasis.index(p4)
            p5 = sgn3 +sgn4 +str(self[i3][i4])  #p3 *p4
            p5 = p5[2:] if p5[:2] == "--" else p5
            sgn5 = "-" if p5 in mBasis else ""
            if p5 == p2 or p5 == mBasis[i2]:
              i5 = mBasis.index(p5) if sgn5 else pBasis.index(p5)
              prod[p1].extend([p2, mBasis[i2], p3, mp3])
              prod[mBasis[i1]].extend([p2, mBasis[i2], p3, mp3])
              prod[p2].extend([p1, mBasis[i1], p3, mp3])
              prod[mBasis[i2]].extend([p1, mBasis[i1], p3, mp3])
              prod[p3].extend([p1, mBasis[i1], p2, mBasis[i2]])
              prod[mp3].extend([p1, mBasis[i1], p2, mBasis[i2]])
              com = 0
              if str(self[i2][i1]) == p3: com += 1 # p2 *p1
              if str(self[i3][i2]) == str(self[i2][i3]): com += 1 # p3 *p2
              if str(self[i4][i3]) == str(self[i3][i4]): com += 1 # p4 *p3
              row = (i1 +1, i2 +1, (-i3 -1) if sgn3 else i3 +1,
                  (-i4 -1) if sgn4 else i4 +1, (-i5 -1) if sgn5 else i5 +1, com)
              out.append(row)
    if dump:
      if len(out) > 1:
        out0 = []
        for row in out:
          p3 = pBasis[row[2]-1] if row[2] > 0 else mBasis[-row[2] -1]
          p4 = pBasis[row[3]-1] if row[3] > 0 else mBasis[-row[3] -1]
          p5 = pBasis[row[4]-1] if row[4] > 0 else mBasis[-row[4] -1]
          cycle = (row[3]==row[0] and row[4]==row[1])
          out0.append(("%s*%s = %s" %(pBasis[row[0] -1], pBasis[row[1] -1], p3),
                       "%s*%s = %s" %(pBasis[row[1] -1], p3, p4),
                       "%s*%s = %s" %(p3, p4, p5), "%s & %d" %(cycle, row[5])))
        Tensor(out0).dump(("Cycle1", "Cycle2", "Cycle3", "Rotation&Commutes"),
                         list(range(1, len(out0)+1)))
      else:
        Tensor(out).dump()
      return None
    if not indicies:
      out0 = []
      isStr = isinstance(basis[0], Common._basestr)
      for row in out:
        p1 = basis[row[0]-1]
        p2 = basis[row[1]-1]
        p3 = basis[row[2]-1] if row[2] > 0 else (mBasis[-row[2] -1] \
                                              if isStr else -basis[-row[2] -1])
        out0.append((p1, p2, p3))
      out = out0
    return Tensor(*out)

  def __assocTriads1(self, x1, x2, basis, mBasis):
    """Internal function to return +/- index of x1 * x2 from square table."""
    scalar = len(basis) +1  # Represent +-1 as +-(len(basis) +1)
    if abs(x1) == scalar: return x2 *(-1 if x1 < 0 else 1)
    if abs(x2) == scalar: return x1 *(-1 if x2 < 0 else 1)
    mul = self[abs(x1) -1][abs(x2) -1]
    if abs(x1) == abs(x2):
      neg = (mul[0] == "-") if isinstance(mul, Common._basestr) else (mul < 0)
      idx = scalar * (-1 if neg else 1)
    else:
      idx = (basis.index(mul) +1) if mul in basis else -mBasis.index(str(mul)) -1
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

  def assocTriads(self, basis, nonAssoc=False, nonAlternate=False,
                  nonPower=False, moufang=0, offs=None):
    """assocTriads(basis, [nonAssoc,nonAlternate,nonPower,moufang,offs])
       Return list of 3 +/- basis indicies associative, nonAssociative, nonPower
       nonAlternate or moufang associativity entries in self table. Any mutually
       exclusive, non-associative parameter enabled will return such triples.
       Associator is [a,b,c] = a(bc) -(ab)c and associativity means [a,b,c] = 0.
       Alternate associativity is [a,b,c] = 0 if any two of a,b,c are equal and
       Power associativity means [a,a,a] = 0 and are counted only if associator
       is non-zero. Optional parameters [only one allowed] are boolean apart
       from moufang which is a number 0-5 where 0 is off and 5 is the sum of all
       4 Moufang checks:
         1: a*(b*(a*c)) -((a*b)*a)*c, 2: b*(a*(c*a)) -((b*a)*c)*a,
         3: (a*b)*(c*a) -(a*(b*c))*a, 4: (a*b)*(c*a) -a*((b*c)*a).
       For large basis, offs=0, ... will break when free memory is less than
       1GB and print status periodically including the next offs."""
    self.__checkSquare(basis, "assocTriads", "basis")
    Common._checkType(nonAssoc, bool, "assocTriads")
    Common._checkType(nonAlternate, bool, "assocTriads")
    Common._checkType(nonPower, bool, "assocTriads")
    Common._checkType(moufang, int, "assocTriads")
    if moufang not in range(6):
      raise Exception("Invalid vaue for number in moufang")
    assoc = int(nonAssoc) +int(nonAlternate) +int(nonPower) +int(moufang>0)
    if assoc > 1:
      raise Exception("Invalid number of options in assocTriads")
    for b in basis:
      if not hasattr(b, "assoc"):
        raise("Invalid basis element for assocTriads: %d" %b)
    dim = int(math.log(len(basis))/math.log(2) +1)       # l=pow(2,dim)
    accum = []
    pBasis = list((str(x) for x in basis))
    mBasis = list((x[1:] if x[:1] == "-" else "-" +x for x in pBasis))
    for x in Common.comb(len(basis), 3, True, (offs is not None)):
      if offs is not None:
        if cnt1 == 1000000:
          cnt1 = 0
          cnt2 += 1
          sys.stdout.write("%s %d %d %dMB" %(Common.date(), cnt2, len(accum),
                           Common.freeMemMB()))
          if Common.freeMemMB() < Common._memLimitMB:
            sys.stdout.write("ABORT: Memmory limit reached\n")
            break
        if offs > cnt2:
          continue
        cnt1 += 1
      out = 0
      if moufang:
        if moufang in (1, 5):
          lhs = self.__assocTriads2((x[1], x[0], x[2]), False, basis, mBasis)
          rhs = self.__assocTriads2((x[0], x[1], x[0]), True, basis, mBasis)
          if self.__assocTriads1(x[0], lhs, basis, mBasis) \
            -self.__assocTriads1(rhs, x[2], basis, mBasis):
            accum.append(x)
        if moufang in (2, 5):
          lhs = self.__assocTriads2((x[0], x[2], x[0]), False, basis, mBasis)
          rhs = self.__assocTriads2((x[1], x[0], x[2]), True, basis, mBasis)
          if self.__assocTriads1(x[1], lhs, basis, mBasis) \
            -self.__assocTriads1(rhs, x[0], basis, mBasis):
            if x not in accum:
              accum.append(x)
        if moufang in (3, 5):
          lhs = self.__assocTriads1(x[0], x[1], basis, mBasis)
          rhs = self.__assocTriads1(x[2], x[0], basis, mBasis)
          ths = self.__assocTriads2(x, False, basis, mBasis)
          if self.__assocTriads1(lhs, rhs, basis, mBasis) \
            -self.__assocTriads1(x[0], ths, basis, mBasis):
            if x not in accum:
              accum.append(x)
        if moufang in (4, 5):
          lhs = self.__assocTriads1(x[0], x[1], basis, mBasis)
          rhs = self.__assocTriads1(x[2], x[0], basis, mBasis)
          ths = self.__assocTriads2((x[1], x[2], x[0]), True, basis, mBasis)
          if self.__assocTriads1(lhs, rhs, basis, mBasis) \
            -self.__assocTriads1(x[0], ths, basis, mBasis):
            if x not in accum:
              accum.append(x)
      else:
        out = self.__assocTriads2(x, True, basis, mBasis) \
             -self.__assocTriads2(x, False, basis, mBasis)
        if out and nonAlternate:
          for y in ((x[0], x[0], x[1]), (x[1], x[0], x[0]), (x[1], x[1], x[2])):
             if self.__assocTriads2(y, True, basis, mBasis) \
               -self.__assocTriads2(y, False, basis, mBasis):
               if y not in accum:
                 accum.append(y)
          for y in ((x[2], x[1], x[1]), (x[0], x[0], x[2]), (x[2], x[0], x[0])):
             if self.__assocTriads2(y, True, basis, mBasis) \
               -self.__assocTriads2(y, False, basis, mBasis):
               if y not in accum:
                 accum.append(y)
        elif out and nonPower:
          for y in ((x[0], x[0], x[0]), (x[1], x[1], x[1]), (x[2], x[2], x[2])):
            if self.__assocTriads2(y, True, basis, mBasis) \
              -self.__assocTriads2(y, False, basis, mBasis):
              if x not in accum:
                accum.append(x)
        elif out and nonAssoc:
          accum.append(x)
        elif assoc == 0 and not out:
          accum.append(x)
    ret = []
    for x in accum:
      ret.append(list(basis[y -1] for y in x))
    return ret

  def isomorph(self, basis, perm):
    """isomorph(basis, perm)
       Return self permuted and cells swapped by signed, inverted perm."""
    self.__checkSquare(basis, "isomorph", "basis")
    Common._checkType(perm, (list, tuple), "isomorph")
    if len(basis) == 0 or len(basis) != len(self) or len(perm) != len(basis):
      raise Exception("Invalid length for isomorph")
    isStr = isinstance(basis[0], Common._basestr)
    if isStr:
      mBasis = list((x[1:] if x[:1] == "-" else ("-" +x) for x in basis))
    else:
      mBasis = list((-x for x in basis))
    pBasis = list((basis[x-1] if x>0 else mBasis[-x-1] for x in perm))
    out = self.permute(perm, True)
    return Tensor(*out.morph(basis, pBasis))

  def permInvert(self):
    """permInvert()
       Return inverted vector permutation."""
    compPerm = []
    for x in range(1, len(self) +1):
      if x in self:
        compPerm.append(self.index(x) +1)
      else:
        compPerm.append(-self.index(-x) -1)
    return Tensor(compPerm)

  def permute(self, perm, invert=False):
    """permute(perm, [invert])
       Return self with rows and columns swapped and signed by perm."""
    Common._checkType(perm, (list, tuple), "permute")
    if self.__size[0] <= 1 or self.__size[0] != len(perm):
      raise Exception("Swap parameter length is not valid")
    if not all(map(lambda x:abs(x) in range(1,len(perm) +1), perm)):
      raise Exception("Swap perm index is out of range")
    isStr = isinstance(self[0] if self.__size[1]<2 else self[0][1],
                       Common._basestr)
    compPerm = Tensor(perm).permInvert() if invert else perm
    rows = []
    for idx in compPerm: # Swap rows
      if idx > 0:
        rows.append(self[idx -1])
      elif isStr:
        val = self[-idx -1][:]
        if self.__size[1] == 1:
          rows.append(val[1:] if val=="-" else "-" +val)
        else:
          rows.append(list((v[1:] if v[0]=="-" else "-" +v for v in val)))
      elif self.__size[1] == 1:
        rows.append(-self[-idx -1])
      else:
        rows.append(list((-v for v in self[-idx -1])))
    out = []  # Swap columns
    if self.__size[1] == 1:
      if len(self) != len(compPerm):
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

  def morphCycles(self, perm, tri=False):
    """morphCycle(perm)
       Return cycle sign permuted by perm as a list from (1,2,3,...)."""
    s0 = []
    iso = []
    for row in self:
      cycLen = 5 if len(row) > 3 else 3
      isoRow = row[:]
      for idx in range(cycLen):
        tmp = perm[abs(row[idx]) -1]
        isoRow[idx] = tmp *(-1 if row[idx] < 0 else 1)
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
      iso.append(isoRow)
    return Tensor(*sorted(iso))

  ############ Other Creators ############
  @staticmethod
  def Resolution(digits):
    """Resolution([digits])
       Set print format digits or reset to default 17."""
    Common.resolution(digits)

  @staticmethod
  def NED(lat, lng):
    """BasisNED(lat, lng)
       Lat/long Earth Centred-Earth Fixed (ECEF) basis changed to
       North-East-Down returned as a 3x3 Matrix [NT,ET,DT]T. From
       onlinelibrary.wiley.com/doi/pdf/10.1002/9780470099728.app3.
       This is introduced to check NED() by rotating i, j & k."""
    Common._checkType(lat, (int, float), "NED")
    Common._checkType(lng, (int, float), "NED")
    sLat,cLat = Common._sincos(math.radians(lat))
    sLng,cLng = Common._sincos(math.radians(lng))
    return Tensor((-cLng *sLat, -sLng *sLat, cLat),
                  (-sLng, cLng, 0),
                  (-cLng *cLat, -sLng *cLat, -sLat))

  @staticmethod
  def Table(basis, rhsBasis=None):
    """Table(basis, [rhsBasis])
       Return Matrix multiplication table for basis times rhsBasis or basis."""
    out = []
    if rhsBasis is None:
      rhsBasis = basis
    Common._checkType(basis, (list, tuple), "Table")
    Common._checkType(rhsBasis, (list, tuple), "Table")
    if len(basis) > 0 and len(rhsBasis) > 0 and not \
          (hasattr(basis[0], "grades") and hasattr(rhsBasis[0], "grades")):
      raise Exception("Table parameter is not a list of basis elements")
    for idx,bas1 in enumerate(rhsBasis):
      row = []
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
    Common._checkType(diag, (list, tuple), "Diag")
    for ii in range(len(diag)):
      out.append([0] *len(diag))
      if isinstance(diag[ii], (list, tuple)):
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
    I = Tensor.Diag([1] *vect1.shape()[0])
    G = vect2 *vect1.transpose() - vect1 *vect2.transpose()
    P = vect1 *vect1.transpose() + vect2 *vect2.transpose()
    M = I - P
    return G,P,M

  @staticmethod
  def Triads(triList, basis):
    """Triads(triList, basis)
       Turn triad list into Table using basis assumed imaginary."""
    Common._checkType(triList, (list, tuple), "Triads")
    Common._checkType(basis, (list, tuple), "Triads")
    isStr = isinstance(basis[0], Common._basestr) if len(basis) > 0 else False
    if isStr:
      tt = Tensor.Diag(["-1"] *len(basis))
    else:
      tt = Tensor.Diag([-1] *len(basis))
    for tri in triList:
      if not isinstance(tri, (list, tuple)) or len(tri) < 3:
        raise Exception("Invalid Triads length: %s" %tri)
      for idx,val in enumerate(tri[:3]):
        Common._checkType(val, int, "Triads")
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
if "numpy" in sys.modules:
  import numpy   # Rename
  class Matrix(numpy.ndarray):
    """Class to interface to numpy instead of using a development class."""
    def __init__(self, *args):
      """Matrix(ndarray)
         Define a nx1 or nxm matrix as an numpy.ndarray."""
      super(Matrix, self).__init__(args)
    def __new__(self, *args):
      """Define a nx1 or nxm matrix as a list or list of lists."""
      arr = numpy.array(args)
      return arr.view(Matrix)

    def __eq__(self, mat):
      """Return True if 2 matricies are equal within precision."""
      if not isinstance(mat, (numpy.ndarray, list, tuple, Matrix)):
        return False
      return ((self - mat) < Common._getPrecision()).all()

    def get(self, x, y):
      """get(x,y)
         Return the value of a matrix element. Used for FrameMatrix."""
      return self[x][y]

    def slice(self, offset, shape):
      """slice(offset, size):
         Return copy of 2-D selection or square if offset,shape are integers."""
      return self._getitem(offset, shape) # TBD XXX

    @staticmethod
    def FromNumpy(array):
      """FromNumpy(array)
         Convert from numpy.ndarray to Matrix to process."""
      return array.view(Matrix)

    @staticmethod
    def Resolution(digits):
      """Resolution([digits])
         Set print format digits or reset to default 17."""
      Common.resolution(digits)
      if digits == 0:
        digits = 18
      numpy.set_printoptions(digits)

    @staticmethod
    def Diag(diag):
      """Diag(diag)
         Return the zero matrix with diag as diagonal entries."""
      Common._checkType(diag, (list, tuple, Matrix), "Diag")
      size = diag.shape[0] if isinstance(diag, Matrix) else len(diag)
      out = numpy.identity(size)
      for ii in range(size):
        if isinstance(diag[ii], (list, tuple)):
          raise Exception("Invalid Diag element")
        out[ii][ii] = diag[ii]
      return Matrix(*out)
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
      Common._checkType(val, (int, float), "Euler")
  def __repr__(self):
    """Overwrite object output using __str__ for print if verbose."""
    if (Common._isVerbose()):
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __str__(self):
    """Overload string output. Printing taking resolution into account."""
    return "%s" %Matrix(*self)
  def __eq__(self, cf):
    """Return True if 2 Eulers are equal within precision."""
    if not isinstance(cf, Euler) or len(self) != len(cf):
      return False
    precision = Common._getPrecision()
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
      if abs(val) < Common._getPrecision():
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
    Common._checkType(order, (list, tuple), "Euler.matrix")
    Common._checkType(implicit, bool, "Euler.matrix")
    rank = len(self)
    if not order:
      if rank == 3 and not implicit and offset == 0:
        sx,cx = Common._sincos(self[0])
        sy,cy = Common._sincos(self[1])
        sz,cz = Common._sincos(self[2])
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
    blank = Matrix.Diag([1] *(dim +offset))
    mat = blank.copy()
    implicitRot = blank.copy()
    store = []
    for key in order:
      if isinstance(key, Common._basestr):
        for i,code in enumerate(names):
          if code.find(key) >= 0:
            key = i +1
            break
      if key in store or key not in range(1, len(self) +1):
        raise Exception("Invalid order index for Euler.matrix: %s" %key)
      sw,cw = Common._sincos(self[key -1])
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
        rot = implicitRot *rot *implicitRot.transpose()
        implicitRot *= tmpRot
      else:
        store.append(key)
      mat = rot *mat
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
    Common._checkType(mat, (Matrix, Tensor), "Matrix")
    rank = len(mat) -offset
    blank = Matrix.Diag([1] *(rank +offset))
    dim = int((math.sqrt(8*(rank -offset) +1) +1) /2 +0.9) # l=comb(dim,2)
    angles = [0] *(int(Common.comb(rank, 2)))
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
        sw,cw = Common._sincos(angles[cnt +idx])
        idx1,idx2 = xyz[idx]
        idx1 += offset
        idx2 += offset
        rot = blank.copy()
        rot[idx1][idx1] = cw   # Transpose = inverse to .matrix
        rot[idx1][idx2] = -sw
        rot[idx2][idx1] = sw
        rot[idx2][idx2] = cw
        mat = rot *mat
    v0 = mat.get(offset, offset)
    v1 = mat.get(offset +1, offset)
    pitch = -math.atan2(mat.get(offset +2, offset), math.sqrt(v0 *v0 +v1 *v1))
    angles[1] = Common._piRange(pitch)
    if abs(abs(pitch) -math.pi /2) < Common._getPrecision():
      sgn = -1.0 if pitch < 0 else 1.0
      angles[0] = Common._piRange(math.atan2(sgn *mat.get(offset +1, offset +2),
                                             sgn *mat.get(offset,offset +2)))
    else:
      angles[0] = Common._piRange(math.atan2(mat.get(offset +2, offset +1),
                                             mat.get(offset +2, offset +2)))
      angles[2] = Common._piRange(math.atan2(v1, v0))
    return Euler(*angles)

################################################################################
if __name__ == '__main__': # Might as well run calcR
  import traceback
  import sys, os
  from math import *
  from calcR import *
  exp = Common.exp
  log = Common.log
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
    exec(Common.readText(fName))

