#!/usr/bin/env python
################################################################################
## File: calcCA.py needs calcR.py and is part of GeoAlg.
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
## CalcCA is a commnd line calculator that converts basis numbers into 
## Clifford Algebra (CA), Quaternions, Versors and Euler Angle rotations.
##
## CA multivectors are of the form p = w +p e0..F i1..F +... where e0=i1.
## A quaternion maps as i->e32, j->e13, k->e21 & e123 maps to vector basis.
## Quaternions are of the form q = w + _v_ where _v_ is a i,j,k vector
## s.t. _v_*_v_ >= 0 and w is a scalar. A pure quaternion has w==0.
## A unit quaternion has _n_*_n_ = -1. A versor has len = 1 which means
## r = cos(a) + sin(a) * _n_ where _n_ in a unit. This is extended to CA for
## the n-D rotation group, O(n) where _n_ = e12, e13, e14, ... e(n-1)n.
## This gives a rank-n rotation matrix with twice the rotation angle as the
## versor angle a due to the rotation group operation p' = r *p *r.inverse().
## Can be included as a module or run as a command line calculator.
## Assumes calcR.py is in same directory & numpy is imported first, if required.
## CA & optional Quaternion tests are included at the end of this file.
## Start with either calcCA.py, python calcCA.py or see ./calcR.py -h.
################################################################################
__version__ = "0.5"
import math
from calcLib import *

################################################################################
class CA():
  """Class to process Clifford Algebra consisting of 32-D numbers, half being
     positive definite (e1..eF) and half being imaginary (i1..iF). Basis e0=i1
     and shorthand for multibasis forms exist (eg eij=eiej). Multiplication of
     basis elements negates all index swaps and negates imaginaes squares. 
     Use CA.Basis(p,q) to discover the basis elements for CA(p,q). Note that
     CA(0,1) is complex numbers, CA(0,2) is quaternions and CA(3) is the 
     product of these since every even CA sub-algebra is another CA. CA(3,1)
     is space-time. Just enter arbirary elements and the highest dimensions are
     remembered for producing matricies. Package math methods only work on the
     scalar part eg sin(ca.scalar()).
     """
  __HEX_BASIS   = 15                     # e and i basis size excluding 0
  __HEX_CHARS   = ('A', 'B', 'C', 'D', 'E', 'F')
  __BASIS_CHARS = ('e', 'i')             # CA basis chars only
  __maxBasis    = ['0', '0']             # Store the max dimensions
  dumpRepr      = False                  # Repr defaults to str

  class Grade:
    """Each CA has an list of e & i basis elements ordered by grade. Each
       basis is a list of ordered hex digits."""
    def __init__(self, value, bases):
      self.value = value
      self.__eBase = bases[0]
      self.__iBase = bases[1]
    def new(self, value):
      inherit = self.__new__(CA.Grade)
      inherit.value = 0
      inherit.__eBase = ""
      inherit.__iBase = ""
      return inherit
    def bases(self):
      return (self.__eBase, self.__iBase)
    def lens(self):
      return (len(self.__eBase), len(self.__iBase))
    def strs(self, entered0=0):
      """Convert e1/i1 to i0/e0 depending on entered0 bits."""
      eOut = ('e' +self.__eBase) if self.__eBase else ""
      iOut = ('i' +self.__iBase) if self.__iBase else ""
      if entered0:
        eBase = self.__eBase
        iBase = self.__iBase
        sgn = False
        if entered0 == 3 and eBase[:1] == '1' and iBase[:1] == '1':
          eOut = 'e0' +eBase[1:]
          iOut = 'i0' +iBase[1:]
          sgn = not sgn
        else:
          if entered0 &1 and iBase[:1] == '1':
            eOut = 'e0' +eBase
            iOut = ('i' +iBase[1:]) if iBase[1:] else ""
            if len(eOut) %2 == 1:
              sgn = not sgn
          if entered0 &2 and eBase[:1] == '1':
            iOut = 'i0' +iBase
            eOut = ('e' +eBase[1:]) if eBase[1:] else ""
            if len(eBase) %2 == 0:
              sgn = not sgn
        if sgn:
          eOut = "-" +eOut
      return (eOut, iOut)
    def __str__(self):
      return "%s[%s,%s]" %(self.value, self.__eBase, self.__iBase)
    __repr__ = __str__
    def copy(self, value=None):
      return CA.Grade(self.value if value is None else value,
                     (self.__eBase[:], self.__iBase[:]))

    def commutes(self, rhs):
      """Return boolean of self commutes with rhs."""
      cnt = 0
      for bas in rhs.__eBase:
        cnt += len(self.__eBase) -(1 if bas in self.__eBase else 0) \
              +len(self.__iBase)
      for bas in rhs.__iBase:
        cnt += len(self.__iBase) -(1 if bas in self.__iBase else 0) \
              +len(self.__eBase) +len(rhs.__eBase)
      return cnt %2 == 0

    def isEq(self, cf, precision):
      """Return true if the grades are equal within precision."""
      return abs(self.value -cf.value) <= precision \
             and self.__eBase == cf.__eBase and self.__iBase == cf.__iBase

    def order(self, cf):
      """Find the placement of a single CA term in self taking into account
         the base signature and sign change under swapping."""
      if sum(self.lens()) < sum(cf.lens()):
        return -1
      if sum(self.lens()) > sum(cf.lens()):
        return 1
      if len(self.__eBase) < len(cf.__eBase):
          return -1
      if len(self.__eBase) > len(cf.__eBase):
          return 1
      if self.__eBase < cf.__eBase:
        return -1
      if self.__eBase > cf.__eBase:
        return 1
      if self.__iBase < cf.__iBase:
        return -1
      if self.__iBase > cf.__iBase:
        return 1
      return 0

    def mergeBasis(self, value, rhs):
      """Multiply graded basis self by rhs."""
      value *= self.value
      lhs = self.bases()
      bases = [None, None]  # Basis for output
      sgn = 0
      offs = len(lhs[1])
      for index,rBase in enumerate(rhs): # Iterate rhs e and i
        lBase = lhs[index]
        base = ""
        if rBase:
          pos = 0
          for char in rBase:
            while pos < len(lBase):
              ch = lBase[pos]
              if ch >= char:
                break
              base += ch
              pos += 1
            if pos < len(lBase):
              if ch > char:
                sgn += len(lBase) -pos +offs
                base += char
              elif ch == char:
                sgn += len(lBase) -pos +1 +offs
                if index == 1:
                  sgn += 1
                pos += 1
              else:
                sgn += offs
            else:
              base += char
              sgn += offs
          while pos < len(lBase):
            base += lBase[pos]
            pos += 1
          bases[index] = base
        else:
          bases[index] = lhs[index][:]
        offs = 0
      return CA.Grade(-value if sgn %2 else value, bases)

  ##############################################################################
  ## Class overwritten functionality methods
  ##############################################################################
  def __init__(self, *args, **kwargs):
    """CA([scalar, e1 multiplier, ...][basis=multiplier, ...])
       The scalar is followed by values for each vector dimension. Vectors and
       other grades can also be entered in dictionary form as e<hex>=<value>
       with higher and imaginary grades entered with more hex digits eg:
       e12i1=1.0. Repeated bases are not allowed and e0 is i1 and i0 is e1.
       See Basis and BasisArgs for a list of basis numbers and names."""
    self.w = args[0] if args else 0             # Scalar
    self.__g = []                               # Array of ordered Grades
    self.__currentAdd = -1                      # Previous add index
    self.__entered0 = 0                         # Remember for printing e0/i0
    Lib._checkType(self.w, (int, float), "CA")
    if len(args) > self.__HEX_BASIS *2:
      raise Exception("Too many basis elements")
    for idx,val in enumerate(args[1:]):
      Lib._checkType(val, (int, float), "CA")
      if val:     # Single vector
        base = hex(idx +1)[-1]
        self.__g.append(CA.Grade(val, [base,""]))
        if base > CA.__maxBasis[0]:
          CA.__maxBasis[0] = base
    for key,value in kwargs.items():
      Lib._checkType(value, (int, float), "CA")
      if not key:
        self.w += value
      elif value:
        lGrade,entered0 = CA._init(key, value, self.__entered0)
        self.__entered0 = entered0
        self.__add(lGrade)

  @staticmethod
  def _init(key, value, entered0):
    """Return the Grade for basis string key and value +/-1."""
    lGrade = CA.Grade(value, ("", ""))  # Base for e and i, resp
    rBases = ["", ""]
    typ = None
    baseCh = False
    lastChar = ['', '']
    for char in key:
      offset = int(typ == CA.__BASIS_CHARS[1]) # i
      if typ and char.isdigit():
        if char == '0':   # Store e0,i0 as i1,e1 and remember for printing
          entered0 |= (offset +1)
          offset = 1 -offset
          char = '1'
          lastChar[offset] = max(char, lastChar[offset])
        elif char == '1': # Reset remembering
          entered0 &= (offset +1)
        if char <= lastChar[offset]:
          lGrade = lGrade.mergeBasis(1, rBases)
          rBases = ["", ""]
        else:
          lastChar[offset] = char
          if char > CA.__maxBasis[offset]:
            CA.__maxBasis[offset] = char
        rBases[offset] += char
        baseCh = False
      elif typ and char in CA.__HEX_CHARS:
        if char <= lastChar[offset]:
          lGrade = lGrade.mergeBasis(1, rBases)
          rBases = ["", ""]
        else:
          lastChar[offset] = char
          if char > CA.__maxBasis[offset]:
            CA.__maxBasis[offset] = char
        rBases[offset] += char
        baseCh = False
      elif char in CA.__BASIS_CHARS and not baseCh:
        if rBases[0] +rBases[1]:
          lGrade = lGrade.mergeBasis(1, rBases)
          rBases = ["", ""]
        typ = char
        baseCh = True
        lastChar[offset] = ''
      else:
        raise Exception("Invalid basis: %s" %key)
    if typ and baseCh:
      raise Exception("Invalid last basis: %s" %key)
    return lGrade.mergeBasis(1, rBases), entered0

  def __float__(self):
    return float(self.w)
  def __int__(self):
    return math.trunc(self.w)
  def __str__(self):
    """Overload string output. Printing taking resolution into account."""
    out = ""
    sign = ""
    for grade in [None] +self.__g:
      if grade:
        eOut,iOut = grade.strs(self.__entered0)
        if eOut[:1] == "-":
          val = -grade.value
          eOut = eOut[1:]
        else:
          val = grade.value
      else:
        val = self.w
        eOut = iOut = ""
      out += Lib._resolutionDump(sign, val, eOut +iOut)
      if out:
        sign = " +"
    return out if out else "0"
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if Lib._isVerbose() and CA.dumpRepr:
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __hash__(self):
    """Allow dictionary access for basis objects."""
    return hash(str(self))

  def __eq__(self, cf):
    """Return True if 2 CAs are equal within precision."""
    precision = Lib._getPrecision()
    if isinstance(cf, (int, float)):
      return not self.__g  and abs(self.w -cf) <= precision
    elif not isinstance(cf, CA):
      return False
    if abs(self.w -cf.w) > precision:
      return False
    idx = 0
    cfIdx = 0
    while True:
      base = self.__g[idx] if idx < len(self.__g) else None
      cfBase = cf.__g[cfIdx] if cfIdx < len(cf.__g) else None
      if not (base and cfBase):
        if not base:
          if not cfBase:
            return True
          if abs(cfBase.value) > precision:
            return False
          cfIdx += 1
        else:
          if abs(base.value) > precision:
            return False
          idx += 1
      else:
        order = base.order(cfBase)
        if order == 0:
          if abs(base.value -cfBase.value) > precision:
            return False
          idx += 1
          cfIdx += 1
        else:
          if order < 0:
            if abs(cfBase.value) > precision:
              return False
            idx += 1
          else:
            if abs(base.value) > precision:
              return False
            cfIdx += 1
    raise Exception("Programming error for equality")

  def __ne__(self, cf):
    """Not equal is not automatic. Need this."""
    return not self.__eq__(cf)
  def __lt__(self, cf):
    """Return True if all scalar and graded parts smaller."""
    return self.__cf(cf, lambda x,y: x < y)
  def __le__(self, cf):
    """Return True if all scalar and graded parts less than or equal."""
    return self.__cf(cf, lambda x,y: x <= y)
  def __gt__(self, cf):
    """Return True if all scalar and graded parts greater."""
    return self.__cf(cf, lambda x,y: x > y)
  def __ge__(self, cf):
    """Return True if all scalar and graded parts greater than or equal."""
    return self.__cf(cf, lambda x,y: x >= y)

  def __add__(self, ca):
    """Add 2 CAs or a scalar from w."""
    if isinstance(ca, CA):
      out = self.dup(self.w +ca.w)
      for grade in ca.__g:
        out.__add(grade)
    elif isinstance(ca, Tensor):
      out = ca.__add__(self)
    else:
      Lib._checkType(ca, (int, float), "add")
      out = self.dup(self.w +ca)
    return out
  __radd__ = __add__

  def __sub__(self, ca):
    """Subtract 2 CAs or a scalar from w."""
    if isinstance(ca, CA):
      lhs = self.__copy()
      rhs = ca.__copy()
      for key,val in rhs.items():
        if key in lhs:
          lhs[key] -= val
        else:
          lhs[key] = -val
      return self.copy(self.w -ca.w, **lhs)
    if isinstance(ca, Tensor):
      return ca.__add__(-self)
    Lib._checkType(ca, (int, float), "sub")
    out = self.dup(self.w -ca)
    return out
  def __rsub__(self, sa):
    """Subtract CA from scalar with CA output."""
    return self.__neg__().__add__(sa)

  def __neg__(self):
    """Unitary - operator for CA."""
    out = self.dup(-self.w)
    for grade in out.__g:
      grade.value = -grade.value
    return out
  def __pos__(self):
    """Unitary + operator for CA."""
    return self
  def __abs__(self):
    """Unitary abs operator for CA."""
    out = self.dup(abs(self.w))
    for grade in out.__g:
      grade.value = abs(grade.value)
    return out
  abs = __abs__

  def __mul__(self, ca):
    """Multiplication of 2 CAs or self by scalar."""
    if isinstance(ca, CA):
      out = CA(self.w *ca.w)
      out.__entered0 = self.__entered0 | ca.__entered0
      if self.w:
        for grade2 in ca.__g:
          grade = self.Grade(self.w, ("", ""))
          out.__add(grade.mergeBasis(grade2.value, grade2.bases()))
      if ca.w:
        for grade1 in self.__g:
          out.__add(grade1.mergeBasis(ca.w, ("","")))
      for grade1 in self.__g:
        for grade2 in ca.__g:
          out.__add(grade1.mergeBasis(grade2.value, grade2.bases()))
    elif isinstance(ca, Tensor):
      return ca.__rmul__(self)
    else:
      Lib._checkType(ca, (int, float), "mul")
      out = CA(self.w *ca)
      out.__entered0 = self.__entered0
      if ca:
        for grade in self.__g:
          out.__g.append(self.Grade(grade.value *ca, grade.bases()))
    return out
  __rmul__ = __mul__

  def __bool__(self):
    return self != 0
  __nonzero__ = __bool__

  def __div__(self, ca, isFloor=False):
    """Attempted division for 2 versors or self by scalar."""
    if isinstance(ca, CA):
      out = self.__mul__(ca.inverse())
      if isFloor:
        out.w = int(out.w)
        for grade in out.__g:
          grade.value = int(grade.value)
      return out
    Lib._checkType(ca, (int, float), "div")
    if abs(ca) < Lib._getPrecision():
      raise Exception("Illegal divide by zero")
    if sys.version_info.major == 2 or isFloor:  # Python v2 to v3
      if isinstance(ca, int) or isFloor:
        out = CA(int(self.w /ca))
        for grade in self.__g:
          out.__g.append(self.Grade(int(grade.value /ca), grade.bases()))
      else:
        out = CA(float(self.w) /ca)
        for grade in self.__g:
          out.__g.append(self.Grade(float(grade.value) /ca, grade.bases()))
    else:
      out = CA(self.w /ca)
      for grade in self.__g:
        out.__g.append(self.Grade(grade.value /ca, grade.bases()))
    out.__entered0 = self.__entered0
    return out 
  __truediv__ = __div__
  def __floordiv__(self, ca): return self.__div__(ca, True)

  def __rdiv__(self, ca): return self.inverse().__mul__(ca) # Scalar / CA
  __rtruediv__ = __rdiv__
  def __rfloordiv__(self, ca): return CA(ca).__div__(self, True)

  def __cf(self, cf, oper):
    """Return inside/outside graded comparisons for operator."""
    if isinstance(cf, (int, float)):
      res = True
      if not self.__g:
        if not oper(self.w, cf):
          res = False
      if res:
        for g in self.__g:
          if not oper(g.value, 0.0):
            res = False
            break
      return res
    elif not isinstance(cf, CA):
      raise Exception("Invalid comparison for O: %s" %type(cf))
    cfIdx = 0
    idx = 0
    res = True
    while True:
      base = self.__g[idx] if idx < len(self.__g) else None
      cfBase = cf.__g[cfIdx] if cfIdx < len(cf.__g) else None
      if not (base and cfBase):
        if not base:
          if not cfBase:
            if (self.w or cf.w) and not oper(self.w, cf.w):
              res = False
            return res
          if not oper(0.0, cfBase.value):
            res = False
          cfIdx += 1
        else:
          if not oper(base.value, 0.0):
            res = False
          idx += 1
      else:
        order = base.order(cfBase)
        if order == 0:
          if not oper(base.value, cfBase.value):
            res = False
          idx += 1
          cfIdx += 1
        else:
          if (base.value < 0) != (cfBase.value < 0):
            if not (oper(base.value, 0.0) and oper(0.0, cfBase.value)):
              return False
          if order < 0:
            if not oper(0.0, cfBase.value):
              res = False
            idx += 1
          else:
            if not oper(base.value, 0.0):
              res = False
            cfIdx += 1
    return res

  def __add(self, grade):
    """Add a single CA term to self placing it in the correct order. Optimise
       storage by remembering last addition position."""
    if sum(grade.lens()) == 0:
      self.w += grade.value
      return
    if self.__currentAdd >= 0: 
      pos = self.__currentAdd
    else:
      pos = 0
    pos = 0
    for idx,base in enumerate(self.__g[pos:]):
      order = base.order(grade)
      if order == 0:
        self.__g[idx].value += grade.value
        if not self.__g[idx].value:
          del(self.__g[idx])
          self.__currentAdd -= 1
        return
      elif order > 0:
        break
      pos = idx +1
    if grade.value:
      self.__currentAdd = pos
      self.__g.insert(pos, grade)

  def __copy(self):
    """Used by copy() to turn the basis into a kwargs dictionary."""
    v = {}
    for grade in self.__g:
      eStr,iStr = grade.strs()
      v["%s%s" %(eStr, iStr)] = grade.value
    return v

  def _copyGrades(self):
    """Used by other calculators to copy the grades."""
    out = []
    for grade in self.__g:
      out.append(grade.copy())
    return out

  def _commutes(self, rhs):
    """Return array of booleans for commutes for each term by term expansion."""
    out = ([True] *(len(rhs.__g) +(1 if rhs.w else 0))) if self.w else []
    for lhsBase in self.__g:
      if rhs.w:
        out += [True]
      for rhsBase in rhs.__g:
        out.append(lhsBase.commutes(rhsBase))
    return out

  def __invertible(self, conj=True):
    """Return (conjugate, simple, even, hyperbolic, sum of basis squares).
       This is correct for simple forms but may fail otherwise.
       Flat = [number of imaginary terms, number of hyperbolic terms].
       Diff = Flat[0] == Flat[1] + 1 if scalar != 0.
       Simple = (Diff != Commutes) and 2 or less grades with scalar.
       Even = all even terms including scalar. Commutes = +ve/-ve terms commute.
       Hyperbolic = has x*x>0 terms but no imaginary terms."""
    sgnOut = CA()
    out = CA(self.w) 
    out.__entered0 = self.__entered0
    p2 = 0
    lastDim = (0, 0)
    cnt = 0        # Count of total basis terms
    flat = [0, 0]  # Count of Imaginaries, Hyperbolics with different basis dims
    even = True
    for grade in self.__g:
      dim = grade.lens()
      cnt += 1
      if dim != lastDim:
        lastDim = dim
        sgnVal = grade.copy(1)
        sgnVal = sgnVal.mergeBasis(1, grade.bases())
        flat[int(sgnVal.value > 0)] += 1
        if sum(dim) %2 == 1:
          even = False
      value = grade.value
      p2 += value *value
      if conj:
        if sgnVal.value < 0: # Conjugate if len < 0
          value *= -1
          sgnOut.__g.append(self.Grade(value, grade.bases()))
        else:
          out.__g.append(self.Grade(value, grade.bases()))
    scalar = (1 if self.w else 0)
    simple = False
    if conj:
      if cnt +scalar == 1:
        simple = True
      elif flat[0] == 1:
        if flat[1] == 0 and scalar:
          simple = True
        elif flat[1] + scalar == 1:
          commutes = out._commutes(sgnOut)
          simple = (sum(commutes) == len(commutes))
    return out +sgnOut, simple, even, (flat[1] > 0 and flat[0] == 0), p2

  def __versible(self, conj):
    """Try self*conj to see if it generates a single invertible term. Return
       self/this or 0 if not single. This tests for large versors."""
    tmp = (self * conj).trim()   # Could work anyway
    if sum(tmp.grades()) == 1:   # Single term
      if tmp < 0:
        return -conj *(-tmp).inverse(False)
      else:
        return conj *tmp.inverse(False)
    return 0

  def __bar(self, ca, sign):
    """Return semi-graded sym or asym product for sign=-1 or 1. Different
       definition to structure paper due to Pertti Lounesto feedback.
       Probably upsets Pfaffian proof though."""
    if sign == -1:
      out = CA(0)
      out.__entered0 = self.__entered0
    else:
      out = self.w *ca.w +self *ca.w +ca *self.w
    for grade1 in self.__g:
      l1 = sum(grade1.lens())
      for grade2 in ca.__g:
        l2 = sum(grade2.lens())
        out.__add(grade1.mergeBasis(grade2.value *0.5, grade2.bases()))
        sgn = sign if (l1 %2 == 0 or l2 %2 == 0) else -sign
        out.__add(grade2.mergeBasis(grade1.value *0.5 *sgn, grade1.bases()))
    return out

  def _vectorSizes(self):
    """Return the CA vector sizes. Can't handle negative signatures."""
    dims = self.basis()
    if dims[0] < 3:
      dims[0] = 3
    dims[1] = 0  # No negative signatures
    return dims

  @staticmethod
  def _BasisArg(dim, part):
    """Used by BasisArgs and other calcs and matches Grade.order. Yield
       digits list for the combinations of size part out of dim."""
    out = []
    if part > 0 and dim > 0:
      basis = list(map(lambda x: "%X" %x, range(1, dim +1)))
      for form in Lib.comb(dim, part, basis):
        yield "".join(list(form))

  @staticmethod
  def _BasisArgs(eDim, iDim, grade):
    """Used by other calcs and matches Grade.order. Yield (e,i) basis elements
       as a list of names in addition order."""
    minGrade = grade if grade else 1
    maxGrade = grade +1 if grade else eDim +iDim +1
    for n in range(minGrade, maxGrade): # Lib.additionTree
      for i in range(n +1):
        if eDim >= n-i and iDim >= i:
          oute = tuple(("e" +x for x in CA._BasisArg(eDim, n-i)))
          outi = tuple(("i" +x for x in CA._BasisArg(iDim, i)))
          for out in Lib._mergeBasis(oute, outi):
            yield out

  @staticmethod
  def _VersorArgs(eDim, iDim=0, rotate=False):
    """Used internally and externally - see VersorArgs."""
    xyz = ["32", "13", "21", "14"] if rotate else ["12", "13", "23", "14"]
    out = []
    cnt = 0
    for j in range(2, eDim +1):
      for i in range(1, j):
        if cnt < 4:
          out.append("e" +(xyz[cnt] if eDim > 2 else "12"))
          cnt += 1
        elif rotate and i > 3 and j > 3:
          out.append("e%X%X" %(j,i))
        else:
          out.append("e%X%X" %(i,j))
    cnt = 0
    for j in range(2, iDim +1):
      for i in range(1, j):
        if cnt < 4:
          out.append("i" +(xyz[cnt] if iDim > 2 else "12"))
          cnt += 1
        elif rotate and i > 3 and j > 3:
          out.append("i%X%X" %(j,i))
        else:
          out.append("i%X%X" %(i,j))
    for i in range(1, eDim +1):
      for j in range(1, iDim +1):
        out.append("e%Xi%X" %(i,j))
    return out

  ##############################################################################
  ## Class utility methods
  ##############################################################################
  @staticmethod
  def version():
    """version()
       Return the module version string."""
    return __version__

  def isScalar(self):
    """isScalar()
       Return true is there are no graded parts."""
    return not self.__g

  def isVersor(self, hyperbolic=False):
    """isVersor([hyperbolic])
       Return true if invertible (if hyperbolic) and even. See versor."""
    precision = Lib._getPrecision()
    conj,simple,even,isHyperbolic,p2 = self.__invertible()
    l2 = float(p2 + self.w *self.w)
    return abs(math.sqrt(l2) -1.0) <= precision and (simple or even) \
       and (not isHyperbolic or not hyperbolic)

  def degrees(self, ang=None):
    """degrees(deg, [ang])
       Return or set scalar part in degrees."""
    if ang:
      Lib._checkType(ang, (int, float), "degrees")
      self.w = math.radians(ang)
    return math.degrees(self.w)

  def scalar(self, scalar=None):
    """scalar([scalar])
       Return and/or set scalar part. Use float() [automatic] for return."""
    if scalar is not None:
      Lib._checkType(scalar, (int, float), "scalar")
      self.w = scalar
    return self.w

  def dup(self, scalar=None):
    """dup([scalar])
       Fast copy with optional scalar overwrite."""
    out = CA()
    if scalar is None:
      out.w = self.w
    else:
      Lib._checkType(scalar, (int, float), "dup")
      out.w = scalar
    out.__g = self._copyGrades()
    return out

  def copy(self, *args, **kwargs):
    """copy([scalar, e1 multiplier, ...][basis=multiplier, ...])
       Return clone with optional new basis values."""
    kw = self.__copy()
    kw.update(kwargs)
    if len(args) == 0:
      args = [self.w]
    out = CA(*args, **kw)
    out.__entered0 = self.__entered0
    return out

  def copyTerms(self):
    """copyTerms()
       Return terms as a list of pairs of (term, factor). Cf CA(**dict(...))."""
    v = [("", self.w)] if self.w else []
    for grade in self.__g:
      eStr,iStr = grade.strs()
      v.append(("%s%s" %(eStr, iStr), grade.value))
    return v

  def basisTerms(self):
    """basisTerms()
       Return self as 3 lists = a list of e-basis indicies, values & i-basis."""
    out1,out2,out3 = [],[],[]
    if self.w:
      out1,out2,out3 = [[]],[self.w],[[]]
    for grade in self.__g:
      eBase,iBase = grade.bases()
      basis = []
      for ch in eBase:
        basis.append(int(ch, self.__HEX_BASIS +1))
      out1.append(basis)
      out2.append(grade.value)
      basis = []
      for ch in iBase:
        basis.append(int(ch, self.__HEX_BASIS +1))
      out3.append(basis)
    return out1,out2,out3

  def trim(self, precision=None):
    """trim([precision])
       Return copy with elements smaller than precision removed."""
    if precision is None:
      precision = Lib._getPrecision()
    else:
      Lib._checkType(precision, float, "trim")
    out = CA(0 if abs(self.w) < precision else self.w)
    out.__entered0 = self.__entered0
    for grade in self.__g:
      if abs(grade.value) >= precision:
        out.__g.append(self.Grade(grade.value, grade.bases()))
    return out

  def pure(self, dim=[], even=False, odd=False):
    """pure([dim,even,odd])
       Return the pure dim part or parts if dim is a list (CA() if empty).
       Else all even or odd grades above dim."""
    Lib._checkType(dim, (int, tuple, list), "pure")
    Lib._checkType(even, (bool), "pure")
    Lib._checkType(odd, (bool), "pure")
    if not (dim or even or odd):
      return self.dup(0)
    maxDim = 0
    useDim = (not (even or odd) and isinstance(dim, int))
    if not isinstance(dim, (list, tuple)):
      dim = [dim]
    for i in dim:
      Lib._checkType(i, int, "pure")
      if i > maxDim:
        maxDim = i
    out = CA(self.w) if 0 in dim else CA()
    out.__entered0 = self.__entered0
    for grade in self.__g:
      l2 = sum(grade.lens())
      if dim and l2 > maxDim and useDim:
        break
      parity = ((even and l2 %2 == 0) or (odd and l2 %2))
      if (not dim and useDim) or l2 in dim or parity:
        out.__g.append(self.Grade(grade.value, grade.bases()))
    return out

  def vector(self, size=None):
    """vector([size])
       Return the pure vector part as a Matrix optionally reshaped."""
    v = [0] *(sum(self._vectorSizes()))
    for grade in self.__g:
      bases = grade.bases()
      if len(bases[0]) +len(bases[1]) > 1:
        break
      pos = 0
      for idx,base in enumerate(bases):
        if base:
          pos += int(base[0], self.__HEX_BASIS +1) -1
          pos += 1 if idx==1 and base[0] else 0
      v[pos] = grade.value
    if size:
      return Matrix(*v).reshape(size)
    return Matrix(*v)

  def grades(self, maxSize=0):
    """grades([maxSize])
       Return a list of basis terms count at each grade with scalar first."""
    Lib._checkType(maxSize, int, "grades")
    g = [1 if self.w else 0]
    for base in self.__g:
      l = sum(base.lens())
      if maxSize and l > maxSize:
        break
      if len(g) < l +1:
        g.extend([0] *(l -len(g) +1))
      g[l] += 1
    if maxSize > len(g) -1:
      g.extend([0] *(maxSize -len(g) +1))
    return g

  def dims(self):
    """dims()
       Return a list of maximum basis at each grade with scalar first (0/1)."""
    g = [1 if self.w else 0]
    b = '0'
    for grade in self.__g:
      eStr,iStr = grade.strs()
      l = (len(eStr) -1) if eStr else 0
      l += (len(iStr) -1) if iStr else 0
      if len(g) < l +1:
        g.extend(['0'] *(l -len(g) +1))
        b = '0'
      if eStr and eStr[-1] > b:
        b = eStr[-1]
      if iStr and iStr[-1] > b:
        b = iStr[-1]
      if b > g[l]:
        g[l] = b
    for l in range(1,len(g)):
      g[l] =  int(g[l], self.__HEX_BASIS +1)
    return g

  def basis(self, *maxBasis):
    """basis([maxBasis,...])
       Return the signature or maximum dimension basis of basis elements.
       Optionally set the maxBasis for matrix output. The basis is of form
       ('1','F') or just the first value."""
    dims = self.__class__.__maxBasis
    if maxBasis:
      if len(maxBasis) > len(dims):
        raise Exception("Invalid grade(): %s" %maxBasis)
      for idx in range(len(dims)):
        dims[idx] = '0'
      for idx,val in enumerate(maxBasis):
        if isinstance(val, int):
          val = hex(val).upper()[2:]
        if isinstance(val, Lib._basestr) and len(val) == 1 and \
              (val.isdigit or val in self.__HEX_CHARS):
          dims[idx] = val
        else:
          raise Exception("Invalid grade(): %s" %val)
    for grade in self.__g:
      eStr,iStr = grade.strs()
      dims[0] = max(dims[0], eStr[-1:])  # Convert max char to hex-digit
      dims[1] = max(dims[1], iStr[-1:])
    out0 = int(dims[0], self.__HEX_BASIS +1) # Convert max char to hex-digit
    out1 = int(dims[1], self.__HEX_BASIS +1) # Convert max char to hex-digit
    return [out0, out1]

  def len(self):
    """len()
       Return the signed scalar square root of the square."""
    n2 = self.w *self.w
    for grade in self.__g:
      sgnVal = grade.copy(1)
      sgnVal = sgnVal.mergeBasis(1, grade.bases())
      n2 += grade.value *grade.value *sgnVal.value
    if n2 < 0:
      return -math.sqrt(-n2)
    return math.sqrt(n2)

  def pureLen(self, maxGrade=0):
    """pureLen([maxGrade])
       Return the signed len of the pure part only (stopping at maxGrade)."""
    n2 = 0
    for grade in self.__g:
      if maxGrade and sum(grade.lens()) > maxGrade:
        break
      sgnVal = grade.copy(1)
      sgnVal = sgnVal.mergeBasis(1, grade.bases())
      n2 += grade.value *grade.value *sgnVal.value
    if n2 < 0:
      return -math.sqrt(-n2)
    return math.sqrt(n2)

  def canonical(self):
    """canon[ical]()
       Return canonical involution copy of self with basis parts negated.
       This is an automorphism of the algebra denoted with a tilde (~ accent)
       by R.Harvey so the tilde() method can be used. Grades p%4=(0123) have
       negation (+-+-). Also see composite() and conjugate()."""
    out = self.dup()
    out.__entered0 = self.__entered0
    for grade in out.__g:
      if sum(grade.lens()) %2 == 1:
        grade.value = -grade.value
    return out
  canon = canonical
  tilde = canonical

  def conjugate(self):
    """conj[ugate]()
       Return copy of self with imaginary parts negated. See canonical()."""
    out = self.dup()
    out.__entered0 = self.__entered0
    for grade in out.__g:
      sgnVal = grade.copy(1)
      sgnVal = sgnVal.mergeBasis(1, grade.bases())
      grade.value *= sgnVal.value
    return out
  conj = conjugate

  def reverse(self):
    """rev[erse]()
       Return reversal involution copy of self with basis parts reversed. This
       is an anti-automorphism of the algebra denoted with a check (v accent)
       by R.Harvey so the check() method can be used. Grades p%4=(0123) have
       negation (++--). Also see composite()."""
    out = self.dup()
    out.__entered0 = self.__entered0
    for grade in out.__g:
      if sum(grade.lens()) %4 > 1:
        grade.value = -grade.value
    return out
  rev = reverse
  check = reverse

  def composite(self):
    """compo[site]()
       Return composite involution copy of self applying canonical and reverse
       involutions. This is an anti-automorphism of the algebra denoted with a
       hat (^ accent) by R.Harvey so the hat() method can be used. Grades
       p%4=(0123) have negation (+--+)."""
    out = self.dup()
    out.__entered0 = self.__entered0
    for grade in out.__g:
      if (sum(grade.lens()) +1) %4 > 1:
        grade.value = -grade.value
    return out
  compo = composite
  hat = composite

  def norm(self):
    """norm()
       Return the scalar sqrt of the product with it's conjugate."""
    p2 = self.w *self.w
    for grade in self.__g:
      p2 += grade.value *grade.value
    return math.sqrt(p2)

  def inverse(self, noError=False):
    """inverse([noError])
       Return inverse of self which is conj()/len() if len()!=0 and a versor.
       Simple forms with only two grades can be checked immediately. Simple
       odd and mixed forms are also immediate. But with more than two grades
       an inverse is possible so if noError then try self*self & self*conj().
       If this produces a single term then divide by this term which captures
       examples 3e1234567 +e123 +e145 +e167 +e246 +e257 +e347 +e356 and
       (e0+e1+e2+e12)^2 = 2e012 (ie inverse -e0+e01-e02-e12). If not invertible
       then raise an exception unless !noError in which case 0 is returned.
       NB: spin7_g2.ca p=triads7all(10,56)*e1234567=(1+1236)(1+e1467)(1+e3567)
       ie 3 idempotents is octonion & (p//4-1)**2 = 1 ie invertible. Only 5 of
       each triadOhalf(x) work except x=10,19 have 7. All 35 term combinations
       work except (1,2,7),(1,3,6),(1,4,5),(2,3,5),(2,4,6),(3,4,7),(5,6,7))."""
    out,simple,even,isHyperbolic,p2 = self.__invertible()
    l2 = float(p2 +self.w *self.w)
    if l2 < Lib._getPrecision() or not simple:
      if l2 >= Lib._getPrecision() and out.w >= 0 and sum(out.grades()) == 1:
        return out *(1/l2)
      tmp = (self * self).trim()  # Could be antisymmetric - check
      if sum(tmp.grades()) == 1:  # Single term
        if tmp < 0:
          tmp = -(-tmp).inverse(True)
        else:
          tmp = tmp.inverse(True)
        out = self *tmp
      else:
        out = self.__versible(out)
    else:
      if out.w < 0 and not out.__g:
        out = 0
      else: # Rescale
        out.w /= l2
        for grade in out.__g:
          grade.value /= l2
    if out == 0 and not noError:
      raise Exception("Illegal form for inverse")
    return out

  def dot(self, ca):
    """dot(ca)
       Return odd overlap contraction. Dot product for vectors."""
    Lib._checkType(ca, CA, "dot")
    return self.__bar(ca, -1)

  def wedge(self, ca):
    """wed[ge](ca)
       Return even overlap & scalar expansion. Exterior part if no overlap."""
    Lib._checkType(ca, CA, "wedge")
    return self.__bar(ca, 1)
  wed = wedge

  def cross(self, ca):
    """cross(ca)
       Return cross product of pure(1) parts and using e321 * asym/2 product."""
    Lib._checkType(ca, CA, "cross")
    if len(self.grades()) > 2 or len(ca.grades()) > 2:
      raise Exception("Can only apply cross to 1-forms")
    x = self.pure(1)
    y = ca.pure(1)
    bases = (x+y).basisTerms()
    if len(bases) > 0:
      for base in bases[0]:
        if base[0] > 3:
          raise Exception("Invalid vector in cross")
    out = CA(e321=1) *(x *y -y *x) *0.5 
    out.__entered0 = self.__entered0
    return out

  def sym(self, ca):
    """sym(ca)
       Return symmetric product of two CAs. The dot product *2 for vectors."""
    Lib._checkType(ca, CA, "sym")
    out = self *ca +ca *self
    out.__entered0 = self.__entered0
    return out

  def asym(self, ca):
    """asym(ca)
       Return antisymmetric product of two CAs. Wedge product*2 for vectors."""
    Lib._checkType(ca, CA, "asym")
    out = self *ca -ca *self
    out.__entered0 = self.__entered0
    return out
 
  def associator(self, p, q):
    """associator(p,q)
       Return the associator [self,p,q] = (self * p) *q - self *(p * q),"""
    out = (self * p) *q - self *(p * q)
    out.__entered0 = self.__entered0
    return out
  assoc = associator

  def projects(self, ca):
    """projects(ca)
       Return (parallel, perpendicular) parts of ca projected onto 2-form self.
       Can project multiple grades onto the plane."""
    Lib._checkType(ca, CA, "projects")
    mix = self.grades()[2] if len(self.grades()) == 3 else 0
    if mix == 0 or self.grades() != [0,0,mix]:
      raise Exception("Can only apply projects to a 2-form")
    n1 = abs(self.pureLen())
    if n1 < Lib._getPrecision():
      raise Exception("Invalid length for projects")
    out = [0, 0]
    mul = self.pure()
    for idx,part in enumerate(ca.grades()):
      if idx > 0 and part:
        vect = ca.pure(idx)
        mul1 = mul *vect *mul /float(n1 *n1)
        out[0] += (vect +mul1)/2.0
        out[1] += (vect -mul1)/2.0
    return out

  def reflect(self, ca):
    """reflect(ca)
       Reflect ca by self taking into account self-form parity."""
    Lib._checkType(ca, CA, "reflect")
    parity = self.basisTerms()
    if len(parity[1]) == 0: # Ignore scalars
      return ca
    if len(parity[0]) != 1: # Ignore multiple terms
      raise Exception("Illegal basis for reflect()")
    inv,simple,even,isHyperbolic,p2 = self.__invertible()
    return self *ca *inv *(1 if len(parity[0][0] +parity[2][0]) %2 else -1)

  def reflection(self, ref):
    """reflection(ref)
       Reflect self inplace by ref taking into account ref-form parity."""
    if isinstance(ref, (int, float)):
      ref = CA(ref)
    Lib._checkType(ref, CA, "reflection")
    parity = ref.basisTerms()
    if len(parity[1]) == 0: # Ignore scalars
      return self
    if len(parity[0]) != 1:  # Ignore multiple terms
      raise Exception("Illegal basis for reflection")
    inv,simple,even,isHyperbolic,p2 = ref.__invertible()
    newSelf = ref *self *inv *(1 if len(parity[0][0] +parity[2][0]) %2 else -1)
    self.__g = newSelf.__g
    return self

  def rotate(self, ca):
    """rotate(q)
       Rotate ca by self. See rotation."""
    Lib._checkType(ca, CA, "rotate")
    precision = Lib._getPrecision()
    conj,simple,even,isHyperbolic,p2 = self.__invertible()
    l2 = float(p2 + self.w *self.w)
    if l2 <= precision or not (simple or even):
      conj = self.__versible(conj)
      if conj == 0:
        raise Exception("Illegal versor for rotate")
    if p2 <= precision:
      return ca.dup()
    if abs(math.sqrt(l2) -1.0) <= precision:
      l2 = 1.0
    return self *ca *conj /l2

  def rotation(self, rot):
    """rotation(rot)
       Rotate self inplace by rot converting rot to versor first, if necessary.
       Applying to versors rotates in the same sense as quaternions and frame.
       For CA vectors this is the same as rot.inverse()*self*rot. See versor."""
    Lib._checkType(rot, CA, "rotation")
    precision = Lib._getPrecision()
    conj,simple,even,isHyperbolic,p2 = rot.__invertible()
    l2 = float(p2 + rot.w *rot.w)
    if l2 <= precision or not (simple or even):
      conj = self.__versible(conj)
      if conj == 0:
        raise Exception("Illegal versor for rotation")
    if p2 <= precision:
      return
    if abs(math.sqrt(l2) -1.0) <= precision:
      l2 = 1.0
    newSelf = rot *self *conj /l2
    self.w = newSelf.w
    self.__g = newSelf.__g
    return self

  def frame(self, hyperbolic=False):
    """frame([hyperbolic])
       Return self=w+v as a frame=acos(w)*2 +v*len(w+v)/asin(w) for vector v.
       Ready for frameMatrix. See versor. Set hyperbolic to try hyperbolic
       solutions."""
    precision = Lib._getPrecision()
    conj,simple,even,isHyperbolic,p2 = self.__invertible()
    l2 = p2 +self.w *self.w
    if abs(math.sqrt(l2) -1.0) > precision:
      raise Exception("Illegal versor norm for frame")
    if isHyperbolic and not hyperbolic:
      raise Exception("Illegal hyperbolic versor for frame")
    if not (simple or even):
      if self.__versible(conj) == 0:
        raise Exception("Illegal versor for frame")
    if p2 < precision:
      return CA(1)
    if isHyperbolic:
      w = abs(self.w)
      if w < 1.0:
        raise Exception("Invalid hyperbolic frame angle")
      out = self.dup(math.acosh(w))
      if self.w < 0:
        out.w *= -1
    else:
      w = (self.w +1.0) %2.0 -1.0
      out = self.dup(math.acos(w) *2)
    p1 = math.sqrt(p2)
    if n1 > precision:
      p0 = 1.0 /p1
      for base in out.__g:
        base.value *= p0
    return out
    
  def versor(self, hyperbolic=False):
    """versor([hyperbolic])
       Return a versor of length 1 assuming w is the angle(rad) ready for
       rotation if of the correct form with even grade or equal mixed signature.
       Opposite of frame. See normalise. Hyperbolic versors use cosh and sinh
       expansions if hyperbolic is set."""
    precision = Lib._getPrecision()
    tmp,simple,even,isHyperbolic,p2 = self.__invertible(False)
    if not (simple or even):
      raise Exception("Illegal versor for versor")
    if isHyperbolic and not hyperbolic:
      raise Exception("Illegal hyperbolic versor for versor")
    if p2 <= precision:
      return CA(1)
    if isHyperbolic:
      sw = math.sinh(self.w /2.0)
      cw = math.cosh(self.w /2.0)
    else:
      sw,cw = Lib._sincos(self.w /2.0)
    sw /= math.sqrt(p2)
    out = self.dup(cw)
    for base in out.__g:
      base.value *= sw
    return out

  def unit(self):
    """unit()
       Return self with graded parts normalised to length one."""
    out = self.dup()
    n2 = 0
    for base in out.__g:
      n2 += base.value *base.value
    if n2 > Lib._getPrecision():
      n1 = math.sqrt(n2)
      for base in out.__g:
        base.value /= n1
    return out

  def distance(self, ca):
    """distance(ca)
       Return the Geodesic norm which is the half angle subtended by
       the great arc of the S3 sphere d(p,q)=|log(p.inverse())*q|. Both
       self & argument ca need to be non-hyperbolic versors."""
    Lib._checkType(ca, CA, "distance")
    if self.isVersor(True) and ca.isVersor(True):
      return abs((self.inverse() *ca).log().len())
    raise Exception("Invalid non-hyperbolic, non-versor for distance")

  def normalise(self):
    """normalise()
       Reduces error accumulation. Versors have len 1."""
    n = self.norm()
    if n <= Lib._getPrecision():
      return CA(1.0)
    out = self.dup(self.w /n)
    for base in out.__g:
      base.value /= n
    return out

  def pow(self, exp):
    """pow(exp)
       For even q=w+v then a=|q|cos(a) & v=n|q|sin(a), n unit."""
    # Look for even, non-hyperbolic form
    Lib._checkType(exp, (int, float), "pow")
    if isinstance(exp, int):
      out = CA(1.0)
      for cnt in range(exp):
        out *= self
      return out
    tmp,simple,even,isHyperbolic,p2 = self.__invertible(False)
    if (simple or even) and not isHyperbolic:
      l1 = math.sqrt(p2 +self.w *self.w)
      w = pow(l1, exp)
      if l1 <= Lib._getPrecision():
        return CA(w)
      a = math.acos(self.w /l1)
      s,c = Lib._sincos(a *exp)
      s *= w /math.sqrt(p2)
      out = CA(w *c)
      out.__entered0 = self.__entered0
      for grade in self.__g:
        eStr,iStr = grade.strs()
        out += CA(**{eStr +iStr: grade.value *s})
      return out
    raise Exception("Invalid float exponent for non-hyperbolic, non-versor pow")
  __pow__ = pow

  def exp(self):
    """exp()
       For even q=w+v then exp(q)=exp(w)exp(v), exp(v)=cos|v|+v/|v| sin|v|."""
    # Look for even, non-hyperbolic form
    tmp,simple,even,isHyperbolic,p2 = self.__invertible(False)
    if p2 <= Lib._getPrecision():
      return CA(self.w)
    if even and not isHyperbolic:
      n1 = math.sqrt(p2)
      s,c = Lib._sincos(n1)
      exp = pow(math.e, self.w)
      s *= exp /n1
      out = CA(exp *c)
      out.__entered0 = self.__entered0
      for grade in self.__g:
        eStr,iStr = grade.strs()
        out += CA(**{eStr +iStr: grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for exp")

  def log(self):
    """log()
       The functional inverse of the even exp()."""
    tmp,simple,even,isHyperbolic,p2 = self.__invertible(False)
    l1 = math.sqrt(self.w *self.w +p2)
    if p2 <= Lib._getPrecision():
      return CA(math.log(l1))
    if even and not isHyperbolic:
      s = math.acos(self.w /l1) /math.sqrt(p2)
      out = CA(math.log(l1))
      out.__entered0 = self.__entered0
      for grade in self.__g:
        eStr,iStr = grade.strs()
        out += CA(**{eStr +iStr: grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for log")

  def latLonAlt(self):
    """latLonAlt()
       Return geodetic lat(deg)/long(deg)/altitude(m) on WGS-84 for an ECEF
       quaternion vector (see LatLonAlt()). From fossen.biz/wiley/pdf/Ch2.pdf."""
    precision = Lib._getPrecision()
    ee3 = 1 -Lib._EARTH_ECCENT2
    x = [0] *3
    for grade in self.__g:
      eStr,iStr = grade.strs()
      if eStr and eStr in ["e1", "e2", "e3"] and not iStr:
        x[int(eStr[1], CA.__HEX_BASIS +1) -1] = grade.value
    p = math.sqrt(x[0] *x[0] +x[1] *x[1])
    lat = math.atan2(x[2], p *ee3) # First approx.
    while True:
      lat0 = lat
      sLat,cLat = Lib._sincos(lat)
      N = Lib.EARTH_MAJOR_M /math.sqrt(cLat *cLat +sLat *sLat *ee3)
      if p > precision:
        h = p /cLat -N
        lat = math.atan(x[2] /p /(1 -Lib._EARTH_ECCENT2 *N/(N +h)))
      elif lat >= 0.0:
        h = x[2] -Lib.EARTH_MINOR_M
        lat = math.pi *0.5
      else:
        h = x[2] +Lib.EARTH_MINOR_M
        lat = -math.pi *0.5
      if abs(lat -lat0) <= precision:
        break
    return Matrix(math.degrees(lat),
                  math.degrees(math.atan2(x[1], x[0])), h)

  def euler(self, hyperbolic=False):
    """euler([hyperbolic])
       Versors can be converted to Euler Angles & back uniquely for default
       order. For n-D greater than 3 need to extract sine terms from the last
       basis at each rank and reduce by multiplying by the inverse for each rank
       until 3-D is reached. See Lib.Euler.Matrix. Once 3-D is reached the
       quaternion.euler() calculation can be used. Euler parameters are of the
       form cos(W/2) +n sin(W/2), n pure unit versor. Set hyperbolic to try
       hyperbolic angles."""
    precision = Lib._getPrecision()
    tmp,simple,even,isHyperbolic,p2 = self.__invertible(False)
    l2 = p2 +self.w *self.w
    if not (simple or even):
      raise Exception("Illegal versor for euler")
    if abs(math.sqrt(l2) -1.0) > precision:
      raise Exception("Illegal versor norm for euler")
    if isHyperbolic and not hyperbolic:
      raise Exception("Illegal hyperbolic versor for versor")
    if (isHyperbolic and not hyperbolic) or not (simple or even):
      raise Exception("Illegal versor for euler")
    if p2 <= precision:
      return Euler()
    dims = self._vectorSizes()
    xyz = CA._VersorArgs(*dims, rotate=True)
    cnt = len(xyz)
    angles = [0] *cnt
    selfRot = self.dup()
    for rank in reversed(range(4, dims[0] +1)): # For dims > 3
      base = CA(**{"e%X" %rank: 1})
      mul = selfRot.rotate(base)
      cnt -= rank -1
      for grade in mul.__g:
        if grade.lens()[0] != 1:
          break
        eStr,iStr = grade.strs()
        val = int(eStr[-1], CA.__HEX_BASIS +1)
        if val < rank:
          eB = xyz[cnt +val -1]     # rotation
          angles[cnt +val -1] = grade.value if eB[1] < eB[2] else -grade.value
      accum = 1.0
      mul = CA(1)
      wVal = 0.0
      for idx in range(cnt, rank +cnt -1):
        val = math.asin(angles[idx] /accum)
        angles[idx] = val
        accum *= math.cos(val)
        s,c = Lib._sincos(val *0.5)
        mul *=  CA(c, **{xyz[idx]: -s})
      selfRot = mul *selfRot
    args = [0] *3   # Now only 3 dims left for angles
    xyz = CA._VersorArgs(3)
    for grade in selfRot.__g:
      eStr,iStr = grade.strs()
      if eStr in xyz:
        idx = xyz.index(eStr)
        args[idx] = -grade.value if idx != 1 else grade.value #rotate
    w, x, y, z = selfRot.w, args[2], args[1], args[0] # rotate
    disc = w *y - x *z
    if abs(abs(disc) -0.5) < Lib._getPrecision():
      sgn = 2.0 if disc < 0 else -2.0
      angles[0] = sgn *math.atan2(x, w)
      angles[1] = -math.pi /sgn
      angles[2] = 0.0
    else:
      angles[0] = math.atan2(2.0 * (z * y + w * x), 1.0 - 2.0 * (x * x + y * y))
      angles[1] = math.asin(2.0 * (y * w - z * x))
      angles[2] = math.atan2(2.0 * (z* w + x * y), 1.0 - 2.0 * (y * y + z * z))
    return Euler(*angles)

  def versorMatrix(self, hyperbolic=False):
    """versorMatrix([hyperbolic])
       This is same as frameMatrix but for a versor with half the angle.
       For 3-D substitute 1-c = (w*w+b*b)-(w*w-b*b)=2b*b in frameMatrix
       where c=cosW=w*w-b*b, w=cos(W/2) and q=w+a*b, b=sin(W/2), a*a=-1.
       For self=w+v return I*(2w*w-1) +2*v*vT +2*w*vX, v=self.vector()
       since x*x = 1 -w*w -y*y -z*z, etc. & vX=[[0,-z,y],[z,0,-x],[-y,x,0]],
       and vX is the cross-product matrix ie vx*r = vxr.
       In general: (w1+u)(w2+v) = w1w2 -u.v +w1v +w2u +uxv.
       r' = (w +v)r(w -v)
          = (w*w -l*l)r +2(r.v)v +2w(vxr) where l=len(v).
       Rodrigue's formula for rotation vectors g=n tan(W/2) & f=m tan(V/2)
       where n,m unit vectors then (g,f) is rotation g followed by f:
       (g,f) = (g+f-fxg)/(1-g.f), derived from a=1+g, b=1+f, c=C+D
       c = ab = 1 +f +g +(gxf-g.f), scalar-part: C=1-g.f
       non-scalar part: D=f +g +gxf want c=1+(g,f) so (g,f)=D/C
       where (g,f)=e tan(U/2) as a Rodrigues vector, unit e. n, m & e
       form a spherical triangle with dihedral angles W/2, V/2 & U/2.
       Opposite of Euler.Matrix for default order with CA.Euler. Set
       hyperbolic to try hyperbolic angles."""
    return self.euler(hyperbolic).matrix()

  def frameMatrix(self, hyperbolic=False):
    """frameMatrix([hyperbolic])
       Rodriges for n-D. See https://math.stackexchange.com/questions/1288207/
       extrinsic-and-intrinsic-euler-angles-to-rotation-matrix-and-back for
       3-D. Converts self to versor then applies each even part. Set
       hyperbolic to try hyperbolic angles."""
    return self.versor().euler(hyperbolic).matrix()

  def morph(self, pairs):
    """morph(pairs)
       Morphism with a list of name pairs as ("e1","e2") to map only e1->e2."""
    out = CA(self.w)
    out.__entered0 = self.__entered0
    for grade in self.__g:
      out += CA(**Lib._morph(grade.strs(), grade.value, pairs))
    return out

  def swap(self, basisTerms, signTerms=[]):
    """swap(basisTerms,[signTerms])
       Morphism with a list of terms that contain even lists taken as pairs of
       basis dimension integers, strings or basis with (1,2) or e12 meaning
       map e13 -> -e23 and e23 -> e13 in self. Hence this is a rotation of 90
       degrees with the advantage that rounding errors do not need to be
       trimmed. Terms as lists of indicies are added without sign and a single
       term may contain indicies directly."""
    if abs(basisTerms) == 1:
      return self
    if isinstance(basisTerms, CA):
      basisTerms = basisTerms.basisTerms()
      if signTerms:
        raise Exception("Swap signTerms only valid for list basisTerms")
      if basisTerms[2] and basisTerms[2][0]:
        raise Exception("Swap basisTerms can't be imaginary")
      signTerms = basisTerms[1]
      basisTerms = basisTerms[0]
    else:
      Lib._checkType(signTerms, (list, tuple), "swap")
      signTerms = signTerms +[1] *(len(basisTerms) -len(signTerms))
    Lib._checkType(basisTerms, (list, tuple), "swap")
    if len(basisTerms) == 0:
      out = self.dup() 
    else:
      out = CA(self.w)
      out.__entered0 = self.__entered0
      if not isinstance(basisTerms[0], (list, tuple)):
        basisTerms = [basisTerms]
      newBasisTerms = []
      for swaps in basisTerms:
        newSwaps = []
        try:
          for base in swaps:
            if isinstance(base, Lib._basestr) and len(base) == 1:
              newBase = int(base, self.__HEX_BASIS +1)
            elif isinstance(base, int) and base < self.__HEX_BASIS +1:
              newBase = base
            else:
              newBase = None
            if newBase and newBase not in newSwaps:
              newSwaps.append(newBase)
        except ValueError:
          pass
        if len(newSwaps) != len(swaps):
          raise Exception("Invalid index for swap: %s" %str(swaps))
        if len(newSwaps) %2 or not newSwaps:
          raise Exception("Swap basis can't be empty or odd, use reflection")
        newBasisTerms.append(newSwaps)
      valTerms = self.basisTerms()
      for pos1,term in enumerate(valTerms[0]):
        val = valTerms[1][pos1]
        for pos2,swaps in enumerate(newBasisTerms):
          sgn = signTerms[pos2]
          newTerm = term[:]
          for idx in range(0, len(swaps) -1):
            if abs(swaps[idx]) in newTerm:
              term[newTerm.index(abs(swaps[idx]))] = swaps[idx +1]
              val = -val *sgn
          if abs(swaps[-1]) in newTerm:
            term[newTerm.index(abs(swaps[-1]))] = swaps[0]
            val = val *sgn
        out += val *CA.Eval(term)
    return out

  def allSigns(self, half=False, dump=False):
    """allSigns([half,dump])
       Generate a list of all, half [boolean] or a single indexed term [half=
       int] of the signed combinations of self, (eg allSigns(e1)=[e1,-e1])."""
    terms = Tensor(*list(CA(**dict((x,))) for x in self.copyTerms()))
    for p0 in terms.allSigns(half, dump):
      yield sum(p0)

  def allSignsIndices(self):
    """allSignsIndices()
       Return index and minus sign count of self in allSigns."""
    terms = Tensor(*list(CA(**dict((x,))) for x in self.copyTerms()))
    return terms.allSignsIndices()

  def spin(self, basis=[]):
    """spin([basis])
       Return the Lib.Table triad list and Basis list if basis else
       VersorArgs list from 3-form self finding the largest dimension."""
    maxBasis = 0
    sTriads = []
    triads = []
    for term in self.__g:
      bases = term.bases()[0]
      if len(bases) != 3 or term.bases()[1]:
        raise Exception("Invalid 3-form for spin: %s" %"".join(term.strs()))
      terms = []
      rTerms = []
      for idx,pairs in enumerate(((0,1), (1,2), (0,2))):
        maxBasis = max(maxBasis, int(bases[idx], self.__HEX_BASIS +1))
        terms.append(int(bases[idx], self.__HEX_BASIS +1))
        form = 'e%s%s' %(bases[pairs[0]], bases[pairs[1]])
        rTerms.append(form)
      if term.value < 0:
        tmp = terms[0]; terms[0] = terms[1]; terms[1] = tmp
        tmp = rTerms[0]; rTerms[0] = rTerms[1]; rTerms[1] = tmp
      triads.append(terms)
      sTriads.append(rTerms)
    if basis:
      Lib._checkType(basis, (list, tuple), "spin")
      if len(basis) != maxBasis:
        raise Exception("Invalid basis length for spin")
    else:
      sBasis = CA._VersorArgs(maxBasis)
      basis = list((CA(**{x: 1}) for x in sBasis))
      triads = []
      for terms in sTriads:
        triad = []
        for term in terms:
          triad.append(sBasis.index(term) +1)
        triads.append(triad)
    return (Tensor(*triads), basis)

  ##############################################################################
  ## Other creators and source inverters
  ##############################################################################
  @staticmethod
  def Basis(eDim, iDim=0, grade=0):
    """Basis(eDim, [iDim, grade])
       Return (e,i) basis elements with value one for all or one grade."""
    Lib._checkType(eDim, int, "Basis")
    Lib._checkType(iDim, int, "Basis")
    Lib._checkType(grade, int, "Basis")
    if eDim < 0 or iDim < 0 or eDim > CA.__HEX_BASIS or iDim > CA.__HEX_BASIS:
      raise Exception("Invalid Basis dimension size")
    if grade < 0 or grade > eDim +iDim:
      raise Exception("Invalid Basis grade size")
    return list(CA(**{bas: 1}) for bas in CA._BasisArgs(eDim, iDim, grade))

  @staticmethod
  def BasisArgs(eDim, iDim=0, grade=0):
    """BasisArgs(eDim, [iDim,grade])
       Yield (e,i) basis elements as a list of names in addition order."""
    Lib._checkType(eDim, int, "BasisArgs")
    Lib._checkType(iDim, int, "BasisArgs")
    Lib._checkType(grade, int, "BasisArgs")
    if eDim < 0 or iDim < 0 or eDim > CA.__HEX_BASIS or iDim > CA.__HEX_BASIS:
      raise Exception("Invalid BasisArgs dimension size")
    if grade < 0 or grade > eDim +iDim:
      raise Exception("Invalid BasisArgs grade size")
    for base in CA._BasisArgs(eDim, iDim, grade):
      yield base

  @staticmethod
  def VersorArgs(eDim, iDim=0, rotate=False):
    """VersorArgs(eDim, [iDim, rotate])
       Return the arguments used for Versor() and Euler() (eDim only) in the
       correct order for entering the required number of arguments. It starts
       with the names that map to Quaternion i, j, k with indicies in
       increasing order. This is changed to e32, e13, e21 if rotate is set
       since this maps to the same rotation direction as quaternions and to
       the right hand screw rule axial vectors under left multiplication by
       e123. Both orders exhibit the quaternion cyclic multiplication rule.
       Names > 3-D vary if a 3-D index is included to maintain this rule."""
    Lib._checkType(eDim, int, "VersorArgs")
    Lib._checkType(iDim, int, "VersorArgs")
    Lib._checkType(rotate, bool, "VersorArgs")
    if eDim < 0 or iDim < 0 or eDim > CA.__HEX_BASIS or iDim > CA.__HEX_BASIS:
      raise Exception("Invalid VersorArgs argument size")
    return CA._VersorArgs(eDim, iDim, rotate)

  @staticmethod
  def Versor(*args, **kwargs):
    """Versor([e12 multiplier, ...],[basis=multiplier, ...])
       Return versor(2-D +...) where ... is higher dimensions in the
       form e32=x, e13=y, e21=z, .... Each dimension has (D 2)T=D(D-1)/2
       parameters and these are added as e4 to xyz, e5 to these +e45, etc.
       Use VersorArgs() to see this list. Same result as Euler() but allows
       for any dimension angle instead of listing all dimensions. Bases
       above e34 has the sign reversed."""
    # See Wikipedia.org rotations in 4-dimensional Euclidean space
    # Number of versors in len(args)=comb(dim,2) inversed is:
    dim = int((math.sqrt(8*len(args) +1) +1) /2 +0.9) if args else 0
    if dim > CA.__HEX_BASIS:
      raise Exception("Invalid number of Versor euler angles")
    dim = max(dim, 2)
    for key,value in kwargs.items():
      Lib._checkType(value, (int, float), "CA")
      if value:
        grades = CA._init(key, value, False)[0].lens()
        if grades[0] != 2 or grades[1]:
          raise Exception("Invalid basis for Versor: %s" %key)
        val1 = int(key[1], CA.__HEX_BASIS +1)
        val2 = int(key[2], CA.__HEX_BASIS +1)
        dim = max(val1, val2, dim)
    xyz = CA._VersorArgs(dim, rotate=True)
    out = [0] *len(xyz)
    for key,value in kwargs.items():
      if key in xyz:
        idx = xyz.index(key)
      else:
        idx = xyz.index("e" +key[2] +key[1])
      if idx < len(args):
        raise Exception("Invalid Versor basis duplication: %s" %xyz[idx])
      out[idx] = value
    for idx,value in enumerate(args):
      out[idx] = value
    return CA.Euler(*out)

  @staticmethod
  def Euler(*args, **kwargs): #order=[], implicit=False):
    """Euler([angles, ...][e12=multiplier, ...][order, implicit])
       Euler angles in higher dimensions have (D 2)T=D(D-1)/2 parameters.
       SO(4) has 6 and can be represented by two CAs. Here they are
       changed to a Versor using explicit rotations & this is returned.
       Explicit order means E5 = R5 *R4 *Rz *Rx *Rx for Euler(x,y,z,4,5).
       For 3-D return (cz+sz e21) *(cy+sy e13) *(cx+sx e32).
       kwargs may contains "order" and "implicit". The args arguments
       are entered as radian angles and rotations applied in the given order
       as shown using VersorArgs(). This order can be changed using the order
       array which must be as long as the list of angles. The default is 
       [1,2,...] and must have unique numbers. If implicit is set True then
       previous rotations are applied to each basis and the order can have
       repeats. So Euler(x,y,z,order=[3,1,3]) is R=Z(x)X'(y)Z''(z). If the
       quat module is included then args can be a Euler object."""
    order = kwargs["order"] if "order" in kwargs else [] # for importlib
    implicit = kwargs["implicit"] if "implicit" in kwargs else False
    Lib._checkType(order, (list, tuple), "Euler")
    Lib._checkType(implicit, bool, "Euler")
    if len(args) == 1 and isinstance(args[0], Euler):
      args = list(args[0])
    out = CA(1.0)
    implicitRot = CA(1.0)
    store = []
    dim = 2
    xyz = CA._VersorArgs(dim, rotate=True)
    if args:
      dim = int((math.sqrt(8*len(args) +1) +1) /2 +0.9) # l=comb(dim,2)
      if dim > CA.__HEX_BASIS:
        raise Exception("Invalid number of Euler angles")
      xyz = CA._VersorArgs(dim, rotate=True)
      for idx,val in enumerate(args):
        if xyz[idx] in kwargs:
          raise Exception("Invalid Euler basis duplication: %s" %xyz[idx])
    xyz_in = CA._VersorArgs(3)
    for bi,val in kwargs.items():
      if bi not in ("order", "implicit"):
        while bi not in xyz +xyz_in:
          dim += 1
          if dim > CA.__HEX_BASIS:
            raise Exception("Invalid Euler parameter: %s" %bi)
          xyz = CA._VersorArgs(dim, rotate=True)
        if bi in xyz:
          idx = xyz.index(bi) 
        else:
          idx = 2 -xyz_in.index(bi) #rotate
          val = -val
        args.extend([0] *(idx -len(args) +1))
        args[idx] = val 
    if not order:
      order = range(1, len(args) +1)
    elif implicit and len(order) < len(args):
      raise Exception("Order size should be >= %d" %len(args))
    elif len(order) != len(args):
      raise Exception("Order size should be: %d" %len(args))
    for idx,key in enumerate(order):
      Lib._checkType(key, (float, int), "Euler")
      key = int(key)
      if key in store or key <= 0 or key > len(args):
        raise Exception("Invalid order index for rotation: %s" %key)
      ang = args[key -1]
      Lib._checkType(ang, (int, float), "Euler")
      s,c = Lib._sincos(ang *0.5)
      rot = CA(c, **{xyz[key -1]: s})
      if implicit:
        tmpRot = rot.copy()
        rot.rotation(implicitRot)
        implicitRot *= tmpRot
        #implicitRot = tmpRot *implicitRot
      else:
        store.append(key)
      out = rot * out
    return out

  @staticmethod
  def LatLon(lat, lng):
    """LatLon(lat, lng)
       Return Earth Centred, Fixed (ECEF) vector for geodetic WGS-84
       lat(deg)/long(deg). From Appendix C - Coorinate Transformations
       at onlinelibrary.wiley.com/doi/pdf/10.1002/9780470099728.app3."""
    Lib._checkType(lat, (int, float), "LatLon")
    Lib._checkType(lng, (int, float), "LatLon")
    sLat,cLat = Lib._sincos(math.radians(lat))
    sLng,cLng = Lib._sincos(math.radians(lng))
    major = Lib.EARTH_MAJOR_M
    minor = Lib.EARTH_MINOR_M
    latParametric = math.atan2(minor *sLat, major *cLat)
    sLat,cLat = Lib._sincos(latParametric)
    xMeridian = major *cLat
    return CA(0, e1=xMeridian *cLng, e2=xMeridian *sLng, e3=minor *sLat)

  @staticmethod
  def LatLonAlt(lat, lng, alt=0):
    """LatLonAlt(lat, lng, [alt])
       Return Earth Centred, Earth Fixed (ECEF) vector for geodetic WGS-84 
       lat(deg)/long(deg)/altitude(m). From fossen.biz/wiley/pdf/Ch2.pdf.
       EarthPolar/EarthMajor = sqrt(1-e*e), e=eccentricity."""
    Lib._checkType(lat, (int, float), "LatLonAlt")
    Lib._checkType(lng, (int, float), "LatLonAlt")
    Lib._checkType(alt, (int, float), "LatLonAlt")
    sLat,cLat = Lib._sincos(math.radians(lat))
    sLng,cLng = Lib._sincos(math.radians(lng))
    ee3 = 1 -Lib._EARTH_ECCENT2
    N = Lib.EARTH_MAJOR_M /math.sqrt(cLat *cLat +sLat *sLat *ee3)
    return CA(0, e1=(N +alt) *cLat *cLng, e2=(N +alt) *cLat *sLng,
                   e3=(N *ee3 +alt) *sLat)
  @staticmethod
  def NED(lat, lng):
    """NED(lat, lng)
       Lat/long Earth Centred-Earth Fixed (ECEF) to North-East-Down (NED)
       frame. Return a versor to perform this rotation. The inverse changes
       from NED to ECEF."""
    Lib._checkType(lat, (int, float), "NED")
    Lib._checkType(lng, (int, float), "NED")
    sLat,cLat = Lib._sincos(math.radians(-lat -90) *0.5)
    sLng,cLng = Lib._sincos(math.radians(lng) *0.5)
    return CA(cLng *cLat, e32=-sLng *sLat, e13=cLng *sLat, e21=sLng *cLat)

  @staticmethod
  def FrameMatrix(mat):
    """FrameMatrix(mat)
       Return the CA for a 3-D frame matrix ie opposite of frameMatrix()
       for a unit vector except that angles outside 90 deg are disallowed.
       tr(mat) = 2cosW +1. If W=0 R=Id. If W=pi R=?."""
    Lib._checkType(mat, (Matrix, Tensor), "FrameMatrix")
    if mat.shape() != (3, 3):
      raise Exception("Invalid FrameMatrix Matrix size")
    tr = mat.get(0,0) +mat.get(1,1) +mat.get(2,2)
    arg = (tr -1) *0.5
    if abs(arg) > 1:
      raise Exception("FrameMatrix rotation outside 90 deg is ambiguous")
    W = math.acos(arg)
    if W == 0:
      return CA(1)
    args = [-(mat.get(2,1) -mat.get(1,2)),
             (mat.get(0,2) -mat.get(2,0)),
            -(mat.get(1,0) -mat.get(0,1))]
    a = math.sqrt(args[0]*args[0] +args[1]*args[1] +args[2]*args[2])
    xyz = CA._VersorArgs(3, rotate=True)
    kw = {} 
    for idx,val in enumerate(xyz):
      kw[val] = args[idx] /a
    return CA(W, **kw)

  @staticmethod
  def Eval(terms):
    """Eval(terms)
       Return opposite of copyTerms/basisTerms()(or[0])(or[0][0] for basis)."""
    Lib._checkList(terms, None, "Eval", (1,0))
    if not isinstance(terms[0], (list, tuple)):
      terms = [terms]
    Lib._checkList(terms, (list, tuple), "Eval", (1,0))
    scalar = 0
    out = {}
    if terms[0]:
      if isinstance(terms[0][0], Lib._basestr):
        for item in terms:
          Lib._checkList(item, None, "Eval", 2)
          Lib._checkType(item[0], Lib._basestr, "Eval")
          if len(item[0]) == 0:
            scalar += item[1]
          elif isinstance(item[0], Lib._basestr):
            base = item[0]
            if base and base[0] not in CA.__BASIS_CHARS:
              base = "e" +base
            if base in out:
              out[base] += item[1]
            else:
              out[base] = item[1]
      else:
        if not isinstance(terms[0][0], (list, tuple)):
          terms = [terms]
        terms = list(terms)
        terms.extend([[]] *(3-len(terms)))
        if len(terms[1]) == 0:
          terms[1] = [1] *max(len(terms[0]), len(terms[2]))
        buf = [[None]] *max(map(len,terms))
        for term,base in enumerate(("e","","i")):
          for idx,item in enumerate(terms[term]):
            if buf[idx][0] is None:
              buf[idx] = ["", 0]
            if not base:
              Lib._checkType(item, (int, float), "Eval")
              buf[idx][1] = item
            elif item:
              basis = base
              for num in item:
                Lib._checkType(num, int, "Eval")
                basis += "%X" %num
              buf[idx][0] += basis
        for basis,item in buf:
          if basis in out:
            out[basis] += item
          else:
            out[basis] = item
    return CA(scalar, **out)

  @staticmethod
  def tri2str(tri):
    """tri2str(str)
       Return a string of 105 hex-digits from a triad15. See tri2str()."""
    return "".join(list(x[0][1:] for x in tri.copyTerms()))
  @staticmethod
  def str2tri(str, ):
    """str2tri(str)
       Return triad15 from a string of 105 hex-digits. See str2tri()."""
    ca = CA()
    for x in range(0,len(str),3):
      ca += CA.Eval(("e" +str[x:x+3],1))
    return ca

  @staticmethod
  def Q(*args):
    """Q([scalar, x, y, z])
       Map quaternion basis (w,i,j,k) to (w, e32, e13, e21) with up to 4
       arguments. If calc(Q) included then scalar may instead be a Q object."""
    if Lib.isCalc("Q"):      # If module calcQ included can use Q class
      if len(args) == 1 and isinstance(args[0], Q):
        q = args[0]
        args = [q.w, q.x, q.y, q.z]
    if len(args) > 4:
      raise Exception("Invalid Q parameters")
    xyz = CA._VersorArgs(3, rotate=True)
    kw = {} 
    for idx,val in enumerate(xyz):
      kw[val] = 0 if len(args) < 4 else args[idx +1]
    return CA(0 if len(args) < 1 else args[0], **kw)
  ###################################################
  ## Calc class help and basis processing methods  ##
  ###################################################
  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for CA."""
    calcHelp = """Clifford Algebra Calculator - Process 30-dimensional basis
          numbers (e0..F or i0..F) of signature (+,-) and multiples."""
    ijk = "i,j,k=CA(e32=1),CA(e13=1),CA(e21=1)"
    return (("CA", "Q", "R"), ("CA", "math"), ijk, "default.ca", calcHelp, "")

  @classmethod
  def _setCalcBasis(cls):
    """Load this other calculator. Quaternions are redefined."""
    return "e32,e13,e21"

  @classmethod
  def _validBasis(cls, value):
    """Used by Calc to recognise repeated basis forms e... and i...."""
    if len(value) == 1:
      return 0
    if value[0] not in cls.__BASIS_CHARS:
      return 0 
    isBasis = True
    for ch in value:
      if isBasis and ch in cls.__BASIS_CHARS:
        isBasis = False
      elif ch.isdigit() or ch in cls.__HEX_CHARS:
        isBasis = True
      else:
        return 0
    return 2

  @classmethod
  def _processStore(cls, state):
    """Convert the store array into CA(...) python code. If isMults1/2
       set then double up since MULTS are higher priority then SIGNS.
       The state is a ParseState from Calculator.processTokens()."""
    kw = {}
    line = ""
    signTyp = state.signVal
    firstCnt = 1 if state.isMults1 else -1
    lastCnt = len(state.store) -1 if state.isMults2 else -1
    for cnt,value in enumerate(state.store):
      val,key = value

      # If basis already entered or single so double up or accum scalar
      isMult = (cnt in (firstCnt, lastCnt) and lastCnt != 0)
      if key in kw or isMult:
        if key is None and not isMult:  # Duplicate scalar
          val = "+(%s%s)" %(kw[None], val)
        elif isMult and len(kw) == 1 and None in kw:
          line += kw[None]
          signTyp = "+"
          kw[None] = "0"
        else:
          scalar = ""
          if None in kw:
            scalar = "%s," %kw[None]
            del(kw[None])
          line += signTyp +"CA(%s%s)" %(scalar, ",".join(map(lambda x: \
                  ("%s=%s" %(x,kw[x])) if x else str(kw[x]), kw.keys())))
          signTyp = "+"
          kw = {}
      kw[key] = val

    # Dump the remainder
    if len(kw) == 1 and None in kw:
      signTyp = kw[None][0] if signTyp or kw[None][0] == "-" else ""
      line += signTyp +kw[None][1:]
    else:
      scalar = ""
      if None in kw:
        scalar = "%s, " %kw[None]
        del(kw[None])
      line += signTyp +"CA(%s%s)" %(scalar, ",".join(map(lambda x: \
              ("%s=%s" %(x,kw[x])) if x else str(kw[x]), kw.keys())))
    line += state.aftFill
    state.reset()
    return line

  @staticmethod
  def _processExec(isAns, code):
    """Run exec() or eval() within this modules scope."""
    if isAns:
      global ans
      ans = eval(code, globals())
      return ans
    else:
      exec(code, globals())
      return None

################################################################################
if __name__ == '__main__':
  from calcR import Calculator
  import traceback
  import sys, os
  from math import *
  exp = Lib.exp
  log = Lib.log
  try:
    import importlib
  except:
    importlib = None          # No changing of calculators

  # CA Unit test cases for Calc with Tests[0] being init for each case
  Tests = [\
    """d30=radians(30); d60=radians(60); d45=radians(45); d90=radians(90)
       e=Euler(pi/6,pi/4,pi/2); c=e1+2e2+3e3; c.basis(3)""",
    """# Test 1 Rotate via frameMatrix == versor half angle rotation.
       Rx=d60+e23; rx=(d60 +e23).versor()
       test = Rx.frameMatrix() *c.vector(); store = (rx*c*rx.inverse()).vector()
       Calculator.log(store == test, store)""",
    """# Test 2 Rotate via frameMatrix == versor.rotate(half angle)].
       Rx=d60+e13; rx=CA.Versor(e13=d60)
       test = Rx.frameMatrix() *c.vector(); store = (rx.rotate(c)).vector()
       Calculator.log(store == test, store)""",
    """# Test 3 Rotate versor rotate == rotation of copy.
       Rx=d60+e21; rx=math.cos(d30) +e21*math.sin(d30)
       test = Rx.frameMatrix() *c.vector(); store = (rx*c*rx.inverse()).vector()
       Calculator.log(store == test, store)""",
    """# Test 4 Quat Euler == CA Euler.
       test = CA.Euler(pi/6,pi/4,pi/2)
       store = CA.Euler(e,order=[1,2,3],implicit=False)
       Calculator.log( store == test, store)""",
    """# Test 5 Euler implicit rotation == other order, Rzyx==Rxy'z''.
       if Lib.isCalc("Q"):
         test = CA.Q(Q.Euler(e, order=[1,2,3], implicit=True))
       else:
         test = CA.Euler(e, order=[3,2,1])
         Lib.precision(1E-12)
       store = CA.Euler(e, order=[1,2,3], implicit=True)
       Calculator.log(store == test, store)""",
    """# Test 6 Versor squared == exp(2*log(e)).
       test = CA.Euler(e).pow(2); store = (CA.Euler(e).log() *2).exp()
       Calculator.log(store == test, store)""",
    """# Test 7 Rotate via frameMatrix == versor.versorMatrix(half angle).
       if Lib.isCalc("Q"):
         test = (d45+i+j+k).frameMatrix()
       else:
         test = (d45+i+j+k).frameMatrix()
       store = (d45+i+j+k).versor().versorMatrix()
       Calculator.log(store == test, store)""",
    """# Test 8 Rotate via versor.versorMatrix() == versor.euler().matrix().
       r = d45 +i +j +k; store = r.normalise().euler().matrix()
       if Lib.isCalc("Q"):
         test = r.normalise().versorMatrix()
       else:
         test = r.normalise().versorMatrix()
       Calculator.log(store == test, store)""",
    """# Test 9 Euler Matrix is inverse of versorMatrix.
       test=Tensor(pi/6, pi/4, pi/2)
       store=Euler.Matrix(CA.Euler(*test).versorMatrix())
       Calculator.log(store == Euler(*test), store)""",
    """# Test 10 Geodetic distance = acos(p.w *d.w -p.dot(d)).
       if Lib.isCalc("Q"):
         p = Q.Euler(e); d=(d45+i+2*j+3*k).versor()
         test = math.acos(p.w *d.w -p.dot(d))
         p = CA.Q(p); d = CA.Q(d)
       else:
         p = CA.Euler(e); d=(d45+e321*c).versor()
         test = math.acos(p.w *d.w -p.pure().wedge(d.pure())) # scalar part of even overlap
       store = p.distance(d)
       Calculator.log(abs(store - test) < 3E-5, store)""",
    """# Test 11 Length *2 == dot(self +self).
       store = (c *2).norm(); test = math.sqrt((c +c).dot(c +c))
       Calculator.log(abs(store - test) <1E-15, store)""",
    """# Test 12 Versor *3 /3 == versor.normalise
       Calculator.log(c/c.norm() == c.normalise(), c.normalise())""",
    """# Test 13 Check Rodriges formula
       def para(a,r,w): return a *a.dot(r)
       def perp(a,r,w): return r *math.cos(w) +CA(e321=1)*a.wedge(r) \\
               *math.sin(w) -a *a.dot(r) *math.cos(w)
       store = para(e1,e1+e2,d30)+perp(e1,e1+e2,d30)
       if Lib.isCalc("Q"):
         test = e123 *CA.Q((d30+i).versor().rotate(i+j))
       else:
         test = (d30+e32).versor().rotate(e1+e2)
       Calculator.log(store == test, store)""",
    """# Test 14 Compare Tensor projection and CA.projects.
       def Ptest(a, b, x):
         G,P,N = Tensor.Rotations(a.unit().vector(), b.unit().vector())
         p = (a * b).projects(x); x0 = P *x.vector()
         return [p[0].vector(), p[1].vector()] == [x0, x.vector()-x0]
       d2 = Ptest(CA(0,1), CA(0,0,1), CA(0,1,2))
       d3 = Ptest(CA(0,1,0,0), CA(0,0,1,2), CA(0,1,2,3))
       Calculator.log(d2 and d3, (d2, d3))""",
    """# Test 15 Euler Matrix is inverse of versorMatrix.
       if Lib.isCalc("Q"):
         test = Q.Euler(pi/6, pi/4, pi/2).versorMatrix()
       else:
         test = CA.Euler(pi/6, pi/4, pi/2).versorMatrix()
       store = Euler(pi/6, pi/4, pi/2).matrix()
       Calculator.log(store == test, store)""",
    """# Test 16 Check lat-long conversion to ECEF xyz and back.
       lat=45; lng=45; store = Tensor(lat,lng); Lib.precision(1E-8)
       if Lib.isCalc("Q"):
         test = Q.LatLon(lat,lng)
       else:
         test = CA.LatLon(lat,lng)
       Calculator.log(store == test.latLonAlt()[:2], store)""",
    """# Test 17 Check lat-long-alt conversion to ECEF xyz and back.
       lat=45; lng=45; test = Tensor(lat,lng,0)
       store = CA.LatLonAlt(lat,lng,0)
       Calculator.log(store.latLonAlt() == test, test)""",
    """# Test 18 Check lat-long conversion from ECEF to NED. Ryz==Rz'y.
       lat=45; lng=45; store = CA.NED(lat,lng)
       if Lib.isCalc("Q"):
         test = CA.Q(Q.Euler(0,-radians(lat+90), radians(lng), order=[1,2,3]))
       else:
         test = CA.Euler(0,-radians(lat+90), radians(lng), order=[1,2,3])
       Calculator.log(store == test, test)""",
    """# Test 19 Check lat-long conversion from ECEF to NED. Ryz==Rz'y.
       lat=45; lng=45; test = Tensor.NED(lat,lng); store = []
       def basis(x): r = CA.NED(lat,lng).rotate(x).trim(); \\
         return r.dot(e1),r.dot(e2),r.dot(e3)
       for x in (e1, e2, e3): store.append(list(basis(x)))
       Calculator.log(test == store, test)""",
    """# Test 20 Check real case lat-long conversions.
       roll,pitch,yaw=(0.1, 0.1, 0.1)
       lat,lng = (-34.9285, 138.6007)
       conv = Tensor((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
       model = CA.Euler(Euler.Matrix(conv))
       egNED = CA.Euler(Euler(roll, pitch, yaw), order=[3,2,1])
       model2NED = (egNED * model).normalise()
       ECEF = (CA.NED(lat, lng) * model2NED).normalise()
       store = ECEF.versorMatrix()
       test = Tensor(\
         ( 0.552107290714106247, 0.63168203529742073, -0.544201567273411735),\
         (-0.341090423374471263,-0.424463728320773503,-0.838741835383363443),\
         (-0.760811975866937606, 0.648697425324424537,-0.0188888261919459843))
       Calculator.log(store == test, store)""",
    """# Test 21 CA.Euler.euler 7-D is same as Euler.
       test = Tensor(list((x *0.01 for x in range(1,22))))
       store = CA.Euler(*test).euler()
       Calculator.log(Matrix(store) == test, store)""",
    """# Test 22 Euler Matrix 7-D Matrix is inverse of Euler.matrix.
       test = Euler(*list((x *0.01 for x in range(1,22))))
       store = Euler.Matrix(CA.Euler(test).versorMatrix())
       Calculator.log(store == test, store)""",
       ]

  calc = Calculator(CA, Tests)
  calc.processInput(sys.argv)
##############################################################################
