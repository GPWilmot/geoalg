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
## A unit quaternion has _n_*_n_ = -1. A versor has norm = 1 which means
## r = cos(a) + sin(a) * _n_ where _n_ in a unit. This is extended to CA for
## the n-D rotation group, O(n) where _n_ = e12, e13, e14, ... e(n-1)n.
## This gives a rank-n rotation matrix with twice the rotation angle as the
## versor angle a due to the rotation group operation p' = r *p *r.inverse().
## Can be included as a module or run as a command line calculator.
## Assumes calcR.py is in same directory & numpy is imported first, if required.
## CA & optional Quaternion tests are included at the end of this file.
## Start with either calcCA.py, python calcCA.py or see ./calcR.py -h.
################################################################################
__version__ = "0.1"
import math
from calcCommon import *

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
  __QUAT_CHARS  = ('i', 'j', 'k')        # Quaternion basis chars
  __BASIS_CHARS = ('e', 'i')             # CA basis chars only
  __allChars    = ['e', 'i']             # Maybe include quaternions
  __maxBasis    = ['0', '0']             # Store the max dimensions
  __useQuat     = False                  # Notify Q is included
  dumpRepr      = False                  # Repr defaults to str
  newMul = False
  dumpMul = False

  class Grade:
    """Each CA has an list of e & i basis elements ordered by grade. Each
       basis is a list of ordered hex digits."""
    def __init__(self, value, bases):
      self.value = value
      self.__eBase = bases[0]
      self.__iBase = bases[1]
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

    def isEq(self, cf, precision):
      """Return true if the grades are equal within precision."""
      return abs(self.value -cf.value) < precision \
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

    def mergeBasisNew(self, value, rhs):
      """Multiple graded basis elements."""
      lhs = self.bases()
      bases = [lhs[0], lhs[1]]  # Base for lhs e and i, resp
      offs = len(bases[1])
      sgn = 0
      if value and len(rhs[0]) +len(rhs[1]):
        for index,rBase in enumerate(rhs): # Iterate rhs e and i
          lBase = lhs[index]
          base = ""
          pos = 0
          for char in rBase +"0":
            while True:
              if pos == len(lBase):
                break
              nch = lBase[pos]
              if ch < char:
                base += ch
                pos += 1
                if Common._isVerbose():
                  print("X1 %s(%s)%d %s(%s)" %(ch, lBase, pos, char, rBase), sgn, base)
              elif ch == char:
                pos += 1
                sgn += len(lBase) -pos +offs     # Sign factor
                if index == 1:
                  sgn += 1
                if Common._isVerbose():
                  print("X2 %s(%s)%d %s(%s)" %(ch, lBase, pos, char, rBase), sgn, base)
                char = '0'
                break
              else: # ch > char
                if char == '0':
                  base += ch
                  pos += 1
                  if Common._isVerbose():
                    print("X3 %s(%s)%d %s(%s)" %(ch, lBase, pos, char, rBase), sgn, base)
                else:
                  sgn+= len(lBase) -pos +offs    # Sign factor
                  base += char
                  if Common._isVerbose():
                    print("X4 %s(%s)%d %s(%s)" %(ch, lBase, pos, char, rBase), sgn, base)
                  break
            if pos == len(lBase) and char != '0':
              base += char
          bases[index] = base
          offs = 0
      return CA.Grade(-value if sgn %2 else value, bases)

    def mergeBasis(self, value, rhs):
      """Multiply graded basis self by rhs."""
      value *= self.value
      if CA.newMul:
        return self.mergeBasisNew(value, rhs)
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
    self.w = 0.0 if len(args) == 0 else args[0] # Scalar  #__w TBD
    Common._checkType(self.w, (int, float), "CA")
    self.__g = []                               # Array of ordered Grades
    self.__currentAdd = -1                      # Previous add index
    self.__entered0 = 0                         # Remember for printing e0/i0
    if len(args) > self.__HEX_BASIS *2:
      raise Exception("Too many basis elements")
    for idx,val in enumerate(args[1:]):
      Common._checkType(val, (int, float), "CA")
      if val:     # Single vector
        base = hex(idx +1)[-1]
        self.__g.append(CA.Grade(val, [base,""]))
        if base > CA.__maxBasis[0]:
          U.__maxBasis[0] = base
    for key,value in kwargs.items():
      Common._checkType(value, (int, float), "CA")
      if value:
        lGrade,entered0 = CA._init(key, value, self.__entered0)
        self.__entered0 = entered0
        self.__add(lGrade)

  @staticmethod
  def _init(key, value, entered0):
    """Return the Grade for basis string key and value +/-1."""
    typ = None
    lGrade = CA.Grade(value, ("", ""))  # Base for e and i, resp
    rBases = ["", ""]
    typ = None
    lastChar = '0'
    for char in key:
      offset = int(typ == CA.__BASIS_CHARS[1]) # i
      if typ and char.isdigit():
        if char <= lastChar:
          lGrade = lGrade.mergeBasis(1, rBases)
          rBases = ["", ""]
          if char == '0':   # Store e0,i0 as i1,e1 and remember for printing
            entered0 |= (offset +1)
            offset = 1 -offset
            char = '1'
            rBases[offset] = char
            lGrade = lGrade.mergeBasis(1, rBases)
            rBases = ["", ""]
          else:
            if char == '1': # Reset remembering
              entered0 &= (offset +1)
            rBases[offset] += char
        else:
          rBases[offset] += char
        lastChar = char
        if char > CA.__maxBasis[offset]:
          CA.__maxBasis[offset] = char
      elif typ and char in CA.__HEX_CHARS:
        if char <= lastChar:
          lGrade = lGrade.mergeBasis(1, rBases)
          rBases = ["", ""]
        rBases[offset] += char
        lastChar = char
        if char > CA.__maxBasis[offset]:
          CA.__maxBasis[offset] = char
      elif char in CA.__BASIS_CHARS:
        if rBases[0] +rBases[1]:
          lGrade = lGrade.mergeBasis(1, rBases)
          rBases = ["", ""]
        typ = char
        lastChar = '0'
      else:
        raise Exception("Invalid basis: %s" %key)
    return lGrade.mergeBasis(1, rBases), entered0

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
      out += Common._resolutionDump(sign, val, eOut +iOut)
      if out:
        sign = " +"
    return out if out else "0"
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if Common._isVerbose() and CA.dumpRepr:
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __hash__(self):
    """Allow dictionary access for basis objects."""
    return hash(str(self))

  def __eq__(self, cf):
    """Return True if 2 CAs are equal within precision."""
    precision = Common._getPrecision()
    if isinstance(cf, (int, float)):
      return not self.__g  and abs(self.w -cf) < precision
    elif not isinstance(cf, CA):
      return False
    if abs(self.w -cf.w) >= precision or len(self.__g) != len(cf.__g):
      return False
    for idx,grade in enumerate(self.__g):
      if not grade.isEq(cf.__g[idx], precision):
        return False
    return True
  def __ne__(self, cf):
    """Not equal is not automatic. Need this."""
    return not self.__eq__(cf)
  def __lt__(self, cf):
    """Return True if all graded parts smaller or scalar smaller than cf.w."""
    return self.__cf(cf, lambda x,y: x < y)
  def __le__(self, cf):
    """Return True if all graded parts <= or scalar <= than cf.w."""
    return self.__cf(cf, lambda x,y: x <= y)
  def __gt__(self, cf):
    """Return True if all graded parts greater or scalar greater than cf.w."""
    return self.__cf(cf, lambda x,y: x > y)
  def __ge__(self, cf):
    """Return True if all graded parts >= or scalar >= than cf.w."""
    return self.__cf(cf, lambda x,y: x >= y)

  def __add__(self, ca):
    """Add 2 CAs or a scalar from w."""
    if isinstance(ca, CA):
      out = self.copy(self.w +ca.w)
      for grade in ca.__g:
        out.__add(grade)
    elif isinstance(ca, Tensor):
      out = ca.__add__(self)
    else:
      Common._checkType(ca, (int, float), "add")
      out = self.copy(self.w +ca)
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
    Common._checkType(ca, (int, float), "sub")
    out = self.copy(self.w -ca)
    return out
  def __rsub__(self, sa):
    """Subtract CA from scalar with CA output."""
    return self.__neg__().__add__(sa)

  def __neg__(self):
    """Unitary - operator for CA."""
    out = self.copy(-self.w)
    for grade in out.__g:
      grade.value = -grade.value
    return out
  def __pos__(self):
    """Unitary + operator for CA."""
    return self
  def __abs__(self):
    """Unitary abs operator for CA."""
    out = self.copy(abs(self.w))
    for grade in out.__g:
      grade.value = abs(grade.value)
    return out
  abs = __abs__

  def __mul__(self, ca):
    """Multiplication of 2 CAs or self by scalar."""
    if isinstance(ca, CA):
      if Common._isVerbose() and CA.dumpMul:
        print("MUL %s * %s" %(self, ca))
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
      Common._checkType(ca, (int, float), "mul")
      out = CA(self.w *ca)
      out.__entered0 = self.__entered0
      for grade in self.__g:
        out.__g.append(self.Grade(grade.value *ca, grade.bases()))
    return out
  __rmul__ = __mul__

  def __bool__(self):
    return self != 0
  __nonzero__ = __bool__

  def __div__(self, ca):
    """Attempted division for 2 versors or self by scalar."""
    if isinstance(ca, CA):
      return self.__mul__(ca.inverse())
    Common._checkType(ca, (int, float), "div")
    if abs(ca) < Common._getPrecision():
      raise Exception("Illegal divide by zero")
    if isinstance(ca, int): # Python v2 to v3
      tmp1 = int(self.w /ca)
      tmp2 = float(self.w) /ca
      out = CA(tmp1 if tmp1 == tmp2 else tmp2)
      for grade in self.__g:
        tmp1 = int(grade.value /ca)
        tmp2 = float(grade.value) /ca
        out.__g.append(self.Grade(tmp1 if tmp1==tmp2 else tmp2, grade.bases()))
    else:
      out = CA(self.w /ca)
      for grade in self.__g:
        out.__g.append(self.Grade(grade.value /ca, grade.bases()))
    out.__entered0 = self.__entered0
    return out 
  __truediv__ = __div__
  __floordiv__ = __div__

  def __rdiv__(self, ca):
    """Division for number, ca, divided by a CA."""
    return self.inverse().__mul__(ca)
  __rtruediv__ = __rdiv__
  __rfloordiv__ = __rdiv__

  def __cf(self, cf, oper):
    """Return inside/outside graded comparisons for operator."""
    if isinstance(cf, (int, float)):
      dims = self.grades()
      if not self.__g:
        return oper(self.w, cf)
      for g in self.__g:
        if oper(g.value, cf):
          return True
      return False
    elif not isinstance(cf, CA):
      return False
    cfIdx = 0
    idx = 0
    order = 0
    while True:
      base = self.__g[idx] if idx < len(self.__g) else None
      cfBase = cf.__g[cfIdx] if cfIdx < len(cf.__g) else None
      if base and cfBase:
        order = base.order(cfBase)
      elif not (base and cfBase):
        break
      if order < 0 or not base:
        if not oper(0.0, cfBase.value):
          return False
        idx += 1
      elif order > 0 or not cfBase:
        if not oper(base.value, 0.0):
          return False
        cfIdx += 1
      else:
        if not oper(base.value, cfBase.value):
          return False
        idx += 1
        cfIdx += 1
    return True

  def __add(self, grade):
    """Add a single CA term to self placing it in the correct order. Optimise
       storage by remembering last addition position."""
    if sum(grade.lens()) == 0:
      self.w += grade.value
      if Common._isVerbose() and CA.dumpMul:
        print("ADD0", self, "=", grade, '=>', str(self))
      return
    if self.__currentAdd >= 0: 
      order = self.__g[self.__currentAdd].order(grade)
      pos = self.__currentAdd
    else:
      order = self.__g[0].order(grade) if self.__g else 0
      pos = 0
    order = self.__g[0].order(grade) if self.__g else 0
    pos = 0
    for idx,base in enumerate(self.__g[pos:]):
      order = base.order(grade)
      pos = idx +1
      if order == 0:
        self.__g[idx].value += grade.value
        if not self.__g[idx].value:
          del(self.__g[idx])
          self.__currentAdd -= 1
        if Common._isVerbose() and CA.dumpMul:
          print("XXX", pos, self.__currentAdd)
          print("ADD", self, "=", grade, '=>', str(self))
        return
      elif order > 0:
        pos = idx
        break
      #order = base.order(grade)
      #pos += 1
    if grade.value:
      self.__currentAdd = pos
      self.__g.insert(pos, grade)
    if Common._isVerbose() and CA.dumpMul:
      print("XXX", pos, self.__currentAdd)
      print("NEW", self, "=", grade.strs(), '=>', str(self))
      if not self.__g and self.__currentAdd > 0:
        raise Exception("XXX")

  def __copy(self):
    """Used by copy() to turn the basis into a kwargs dictionary."""
    v = {}
    for grade in self.__g:
      eStr,iStr = grade.strs()
      v["%s%s" %(eStr, iStr)] = grade.value
    return v

  def copyTerms(self):
    """copyTerms()
       Return terms as a list of pairs of (term, factor). Cf CA(**dict(...))."""
    v = [("", self.w)] if self.w else []
    for grade in self.__g:
      eStr,iStr = grade.strs()
      v.append(("%s%s" %(eStr, iStr), grade.value))
    return v

  def _copyGrades(self):
    """Used by other calculators to copy the grades."""
    out = []
    for grade in self.__g:
      out.append(grade.copy())
    return out

  def __versor(self, inversed=False, both=False):
    """If inversed return the out=conjugate, signature, square and term cnt.
       Otherwise return out=None, sign=2, 0, 0 if not versor.
       Return square as sum of pure scalars squared and cnt as term count.
       The signature includes scalar and is flat=1 if more positive, flat=-1
       if more negative and flat=0 if equal. Hyperbolic versors have flat=1
       while rotation versors are 0 or +1 if cnt==1. If both check inverse
       is a versor with even parts only. Psuedo-versor has cnt < 0."""
    out = None
    if inversed:
      out = CA(self.w) 
      out.__entered0 = self.__entered0
    n2 = 0
    cnt = 0 if self.w == 0.0 else 1
    flat = [0, cnt]
    even = 1 if cnt else -1
    for grade in self.__g:
      dim = int(sum(grade.lens()) %2 == 0)
      if even == -1:
        even = dim
      cnt += 1
      if (inversed and not both) or dim == even:
        sgnVal = grade.copy(1)
        sgnVal = sgnVal.mergeBasis(1, grade.bases())
        n2 += grade.value *grade.value
        flat[int(sgnVal.value > 0)] += 1
        if inversed:
          value = grade.value
          if sgnVal.value < 0:
            value *= -1
          out.__g.append(self.Grade(value, grade.bases()))
      else:
        return None, 2, 0, 0  # Not a versor
    outFlat = 0
    if flat[1] > flat[0]:
      outFlat = 1 
    elif flat[1] < flat[0]:
      outFlat = -1
    return out, outFlat, n2, cnt if even else -cnt

  def __bar(self, ca, sign):
    """Return semi-graded sym or asym product for sign=-1 or 1."""
    out = CA(0)
    out.__entered0 = self.__entered0
    for grade1 in self.__g:
      l1 = sum(grade1.lens())
      for grade2 in ca.__g:
        l2 = sum(grade2.lens())
        out.__add(grade1.mergeBasis(grade2.value *0.5, grade2.bases()))
        sgn = -sign if (l1 +l2) %2 == 0 else sign
        out.__add(grade2.mergeBasis(grade1.value *0.5 *sgn, grade1.bases()))
    return out

  def __vectorSizes(self):
    """Return the CA vector sizes. Can't handle negative signatures."""
    dims = self.basis()
    for idx,val1 in enumerate(dims):
      val2 = CA.__maxBasis[idx]
      val2 = int(val2, CA.__HEX_BASIS +1)
      dims[idx] = max(val1, val2)
    if dims[0] < 3:
      dims[0] = 3
    dims[1] = 0  # No negative signatures
    return dims

  @staticmethod
  def _basisArgs(dim, part):
    """Used by BasisArgs and other calcs and matches Grade.order. Return
       digits list for the combinations of size part out of dim."""
    out = []
    if part > 0 and dim > 0:
      for form in Common.comb(dim, part, perms=True):
        out.append("".join(map(lambda x: "%X" %x, form)))
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

  def isVersor(self, nonHyperbolic=False):
    """isVersor(, nonHyperbolic)
       Return true if positive (if !nonHyperbolic) or mixed signature."""
    precision = Common._getPrecision()
    tmp,flat,n2,cnt = self.__versor()
    if nonHyperbolic and flat == 1:
      return abs(math.sqrt(n2 +self.w *self.w) -1.0) < precision
    return (flat == 0 or abs(cnt) == 1) and \
           abs(math.sqrt(n2 +self.w *self.w) -1.0) < precision

  def degrees(self, ang=None):
    """degrees(deg, [ang])
       Return or set scalar part in degrees."""
    if ang:
      Common._checkType(ang, (int, float), "degrees")
      self.w = math.radians(ang)
    return math.degrees(self.w)

  def scalar(self, w=None):
    """scalar([w])
       Return or set scalar part."""
    if w:
      Common._checkType(w, (int, float), "scalar")
      self.w = w
    return self.w

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

  def trim(self, precision=None):
    """trim([precision])
       Return copy with elements smaller than precision removed."""
    if precision is None:
      precision = Common._getPrecision()
    else:
      Common._checkType(precision, float, "trim")
    out = CA(0 if abs(self.w) < precision else self.w)
    out.__entered0 = self.__entered0
    for grade in self.__g:
      if abs(grade.value) >= precision:
        out.__g.append(self.Grade(grade.value, grade.bases()))
    return out

  def pure(self, dim=[], even=False, odd=False):
    """pure([dim,even,odd])
       Return the pure dim part or parts if dim is a list (non-scalar if
       empty). Else all even or odd grades above dim."""
    Common._checkType(dim, (int, tuple, list), "pure")
    Common._checkType(even, (bool), "pure")
    Common._checkType(odd, (bool), "pure")
    useDim = (not (even or odd) and isinstance(dim, int))
    maxDim = 0
    if not isinstance(dim, (list, tuple)):
      dim = [dim]
    for i in dim:
      Common._checkType(i, int, "pure")
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

  def vector(self):
    """vector()
       Return the pure vector part as a Matrix."""
    v = [0] *(sum(self.__vectorSizes()))
    for grade in self.__g:
      bases = grade.bases()
      if len(bases[0]) +len(bases[1]) > 1:
        break
      pos = 0
      for idx,base in enumerate(bases):
        if base:
          pos += int(base[0], self.__HEX_BASIS +1) -1
          pos += 0 if idx==0 else b[0]
      v[pos] = grade.value
    return Matrix(*v)

  def grades(self, maxSize=0):
    """grades([maxSize])
       Return a list of basis terms count at each grade with scalar first."""
    Common._checkType(maxSize, int, "grades")
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
    b = 0
    l = 1
    for grade in self.__g:
      eStr,iStr = grade.strs()
      l = (len(eStr) -1) if eStr else 0
      l += (len(iStr) -1) if iStr else 0
      if len(g) < l +1:
        g.extend([0] *(l -len(g) +1))
      if eStr and eStr[-1] > b:
        b = int(eStr[-1], self.__HEX_BASIS +1)
      if iStr and iStr[-1] > b:
        b = int(iStr[-1], self.__HEX_BASIS +1)
      if b > g[l]:
        g[l] = b
        b = 0
    return g

  def basis(self, *maxBasis):
    """basis([maxBasis,...])
       Return the signature or maximum dimension basis of basis elements.
       Optionally set the maxBasis for matrix output. The basis is of form
       ('1','F') or just the first value."""
    if maxBasis:
      if len(maxBasis) > len(self.__class__.__maxBasis):
        raise Exception("Invalid grade(): %s" %maxBasis)
      for idx in range(len(self.__class__.__maxBasis)):
        self.__class__.__maxBasis[idx] = '0'
      for idx,val in enumerate(maxBasis):
        if isinstance(val, int):
          val = hex(val).upper()[2:]
        if isinstance(val, Common._basestr) and len(val) == 1 and \
              (val.isdigit or val in self.__HEX_CHARS):
          self.__class__.__maxBasis[idx] = val
        else:
          raise Exception("Invalid grade(): %s" %val)
    dims = ["0", "0"]
    for grade in self.__g:
      eStr,iStr = grade.strs()
      dims[0] = max(dims[0], eStr[-1:])  # Convert max char to hex-digit
      dims[1] = max(dims[1], iStr[-1:])
    return [int(dims[0], self.__HEX_BASIS +1),
            int(dims[1], self.__HEX_BASIS +1)]

  def len(self):
    """len()
       Return the sqrt of the scalar sum of the product with it's conjugate."""
    n2 = self.w *self.w
    for grade in self.__g:
      n2 += grade.value *grade.value
    return math.sqrt(n2)

  def vectorLen(self):
    """vectorLen()
       Return the sqrt of the non-scalar product sum with the conjugate."""
    n2 = 0
    for base in self.__g:
      n2 += base.value *base.value
    return math.sqrt(n2)

  def conjugate(self, split=False):
    """conjugate([split])
       Return copy of self with imaginary (all if split) parts negated."""
    out = self.copy(self.w)
    out.__entered0 = self.__entered0
    if split:
      for grade in out.__g:
        grade.value = -grade.value
    else:
      for grade in out.__g:
        sgnVal = grade.copy(1)
        sgnVal = sgnVal.mergeBasis(1, grade.bases())
        grade.value *= sgnVal.value
    return out

  def inverse(self, noError=False):
    """inverse([noError])
       Return inverse attempt of self which is conj()/len() if len()!=0 and not
       known failures. Raise an error on failure or return 0 if noError."""
    out,flat,n2,cnt = self.__versor(inversed=True)
    l2 = float(n2 +self.w *self.w)
    isInvertable = (flat <= 0 or abs(cnt) == 1) and l2 >= Common._getPrecision()
    if not isInvertable:  # Could be anti-symmetric - check
      if not noError:
        raise Exception("Illegal versor for inverse()")
      tmp = self * self
      if not tmp.__g and tmp.w:
        out = self.copy(self.w /tmp.w)
        for grade in out.__g:
          grade.value /= float(tmp.w)
        return out
    out.w /= l2
    for grade in out.__g:
      grade.value /= l2
    return out

  def dot(self, ca):
    """dot(ca)
       Return dot product of pure(1) parts and squaring instead of using sym."""
    Common._checkType(ca, CA, "dot")
    out = idx1 = idx2 = 0
    x = self.__g[0]
    y = ca.__g[0]
    while idx1 < len(self.__g) and idx2 < len(ca.__g):
      x = self.__g[idx1]
      y = ca.__g[idx2]
      if sum(x.lens()) > 1 or sum(y.lens()) > 1:
        break
      order = x.order(y)
      if order < 0:
        idx1 += 1
      elif order > 0:
        idx2 += 1
      else:
        sgn = 1 if x.lens()[0] == 1 else -1
        out += x.value *y.value *sgn 
        idx1 += 1
        idx2 += 1
    return out

  def cross(self, ca):
    """cross(ca)
       Return cross product of pure(1) parts and using e321 * asym product."""
    Common._checkType(ca, CA, "cross")
    x = self.pure(1)
    y = ca.pure(1)
    out = CA(e321=1) *(x *y -y *x) *0.5 
    out.__entered0 = self.__entered0
    return out

  def sym(self, ca):
    """sym(ca)
       Return symmetric product of two CAs. The dot product is for vectors."""
    Common._checkType(ca, CA, "sym")
    out = (self *ca +ca *self) *0.5 
    out.__entered0 = self.__entered0
    return out

  def asym(self, ca):
    """asym(ca)
       Return anti-symmetric product of two CAs. The wedge product is the
       exterior pure of this product."""
    Common._checkType(ca, CA, "asym")
    out = (self *ca -ca *self) *0.5 
    out.__entered0 = self.__entered0
    return out
 
  def assoc(self, p, q):
    """assoc(p,q)
       Return the associator [self,p,q] = (self * p) *q - self *(p * q),"""
    out = (self * p) *q - self *(p * q)
    out.__entered0 = self.__entered0
    return out

  def symbar(self, ca):
    """symbar(ca)
       Return semi-graded symmetric product of two CAs."""
    Common._checkType(ca, CA, "symbar")
    return self.__bar(ca, -1)

  def asymbar(self, ca):
    """asymbar(ca)
       Return semi-graded anti-symmetric product of two CAs."""
    Common._checkType(ca, CA, "asymbar")
    return self.__bar(ca, 1)

  def projects(self, ca):
    """projects(ca)
       Return (parallel, perpendicular) parts of ca projected onto 2-form self.
       Can project multiple grades onto the plane."""
    Common._checkType(ca, CA, "projects")
    mix = self.grades()[2] if len(self.grades()) == 3 else 0
    if mix == 0 or self.grades() != [0,0,mix]:
      raise Exception("Can only apply projects to a 2-form")
    n1 = self.vectorLen()
    if n1 < Common._getPrecision():
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
       Reflect ca by self taking into account r-form parity."""
    Common._checkType(ca, CA, "reflect")
    parity = self.basisTerms()
    if not parity[0]: # Ignore scalars
      return ca
    if len(parity[0]) != 1:
      raise Exception("Illegal basis for reflect()")
    inv,flat,n2,cnt = self.__versor(inversed=True)
    return self *ca *inv *(1 if len(parity[0][0]) %2 else -1)

  def reflection(self, ref):
    """reflection(ref)
       Reflect self inplace by ref taking into account r-form parity."""
    Common._checkType(ref, CA, "reflection")
    parity = ref.basisTerms()
    if not parity[0]: # Ignore scalars
      return
    if len(parity[0]) != 1:
      raise Exception("Illegal basis for reflection")
    inv,flat,n2,cnt = ref.__versor(inversed=True)
    newSelf = ref *self *inv *(1 if len(parity[0][0]) %2 else -1)
    self.__g = newSelf.__g

  def rotate(self, ca):
    """rotate(q)
       Rotate ca by self converting to versor first. See rotation."""
    Common._checkType(ca, CA, "rotate")
    precision = Common._getPrecision()
    inv,flat,n2,cnt = self.__versor(inversed=True, both=True)
    l2 = float(n2 + self.w *self.w)
    if (flat != 0 and cnt != 1) or l2 < precision:
      raise Exception("Illegal versor for rotate")
    if n2 < precision:
      return ca
    if abs(l2 -1.0) < precision:
      l2 = 1.0
    return self *ca *inv /l2

  def rotation(self, rot):
    """rotation(rot)
       Rotate self inplace by rot converting rot to versor first, if necessary.
       Applying to versors rotates in the same sense as quaternions and frame.
       For CA vectors this is the same as rot.inverse()*self*rot."""
    Common._checkType(rot, CA, "rotation")
    precision = Common._getPrecision()
    inv,flat,n2,cnt = rot.__versor(inversed=True, both=True)
    l2 = float(n2 + rot.w *rot.w)
    if (flat != 0 and cnt != 1) or l2 < precision:
      raise Exception("Illegal versor for rotation")
    if n2 < precision:
      return self.copy()
    if abs(l2 -1.0) < precision:
      l2 = 1.0
    newSelf = rot *self *inv /l2
    self.w = newSelf.w
    self.__g = newSelf.__g

  def frame(self, noError=False):
    """frame()
       Return self=w+v as a frame=acos(w)*2 +v*len(w+v)/asin(w) for vector v.
       Ready for frameMatrix. Also handles hyperbolic vector. See versor."""
    precision = Common._getPrecision()
    tmp,flat,n2,cnt = self.__versor()
    l2 = n2 +self.w *self.w
    if (flat != 0 and cnt != 1) or abs(math.sqrt(l2) -1.0) >= precision:
      if not noError:
        raise Exception("Illegal versor for frame")
    if n2 < precision:
      return CA(1)
    if flat > 0:
      w = abs(self.w)
      if w < 1.0:
        raise Exception("Invalid hyperbolic frame angle")
      out = self.copy(math.acosh(w))
      if self.w < 0:
        out.w *= -1
    else:
      w = (self.w +1.0) %2.0 -1.0
      out = self.copy(math.acos(w) *2)
    n1 = math.sqrt(n2)
    if n1 >= precision:
      n0 = 1.0 /n1
      for base in out.__g:
        base.value *= n0
    return out
    
  def versor(self, nonHyperbolic=False, noError=False):
    """versor([noError])
       Return the generalised even parts as unit() and scalar as angle. Opposite
       of frame. See norm. Needs mixed or positive signature or single grade."""
    precision = Common._getPrecision()
    tmp,flat,n2,cnt = self.__versor()
    w1 = self.w
    if cnt < 0:
      w1 = self.__g[-1].value
      n2 -= w1 *w1
    l2 = n2 +w1 *w1
    if (flat > 0 and cnt != 1) or l2 < precision:
      if not noError:
        raise Exception("Illegal versor for versor")
    if n2 < precision:
      return CA(1)
    if flat > 0:
      sw = math.sinh(w1 /2.0)
      cw = math.cosh(w1 /2.0)
    else:
      sw,cw = Common._sincos(w1 /2.0)
    sw /= math.sqrt(n2)
    if cnt < 0:
      out = self.copy()
      for base in out.__g[:-1]:
        base.value *= sw
      out.__g[-1].value = cw
    else:
      out = self.copy(cw)
      for base in out.__g:
        base.value *= sw
    return out

  def unit(self):
    """unit()
       Return self with graded parts normalised to length one."""
    out = self.copy()
    n2 = 0
    for base in out.__g:
      n2 += base.value *base.value
    if n2 >= Common._getPrecision():
      n1 = math.sqrt(n2)
      for base in out.__g:
        base.value /= n1
    return out

  def distance(self, ca):
    """distance(ca)
       Return the Geodesic norm which is the half angle subtended by
       the great arc of the S3 sphere d(p,q)=|log(p.inverse())*q|. Both
       self & argument ca need to be non-hyperbolic versors."""
    Common._checkType(ca, CA, "distance")
    if self.isVersor(True) and ca.isVersor(True):
      return (self.inverse() *ca).log().len()
    raise Exception("Invalid non-hyperbolic, non-versor for distance")

  def norm(self):
    """norm()
       Normalise - reduces error accumulation. Versors have norm 1."""
    precision = Common._getPrecision()
    n = self.len()
    if n < precision:
      return CA(1.0)
    out = self.copy(self.w /n)
    for base in out.__g:
      base.value /= n
    return out

  def pow(self, exp):
    """pow(exp)
       For even q=w+v then a=|q|cos(a) & v=n|q|sin(a), n unit."""
    # Look for even, non-hyperbolic form
    Common._checkType(exp, (int, float), "pow")
    tmp,flat,n2,cnt = self.__versor()
    if flat <= 0:
      l1 = math.sqrt(n2 +self.w *self.w)
      w = pow(l1, exp)
      if l1 < Common._getPrecision():
        return CA(w)
      a = math.acos(self.w /l1)
      s,c = Common._sincos(a *exp)
      s *= w /math.sqrt(n2)
      out = CA(w *c)
      out.__entered0 = self.__entered0
      for grade in self.__g:
        eStr,iStr = grade.strs()
        out += CA(**{eStr +iStr: grade.value *s})
      return out
    elif isinstance(exp, int):
      out = CA(1.0)
      for cnt in range(exp):
        out *= self
      return out
    raise Exception("Invalid float exponent for non-hyperbolic, non-versor pow")
  __pow__ = pow

  def exp(self):
    """exp()
       For even q=w+v then exp(q)=exp(w)exp(v), exp(v)=cos|v|+v/|v| sin|v|."""
    # Look for even, non-hyperbolic form
    tmp,flat,n2,cnt = self.__versor()
    if out and n2 < Common._getPrecision():
      return CA(self.w)
    if out and flat <= 0:
      n1 = math.sqrt(n2)
      s,c = Common._sincos(n1)
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
    tmp,flat,n2,cnt = self.__versor()
    l1 = math.sqrt(self.w *self.w +n2)
    if out and n2 < Common._getPrecision():
      return CA(math.log(l1))
    if out and flat <= 0:
      s = math.acos(self.w /l1) /math.sqrt(n2)
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
    precision = Common._getPrecision()
    ee3 = 1 -Common._EARTH_ECCENT2
    x = [0] *3
    for grade in self.__g:
      eStr,iStr = grade.strs()
      if eStr and eStr in ["e1", "e2", "e3"] and not iStr:
        x[int(eStr[1], CA.__HEX_BASIS +1) -1] = grade.value
    p = math.sqrt(x[0] *x[0] +x[1] *x[1])
    lat = math.atan2(x[2], p *ee3) # First approx.
    while True:
      lat0 = lat
      sLat,cLat = Common._sincos(lat)
      N = Common.EARTH_MAJOR_M /math.sqrt(cLat *cLat +sLat *sLat *ee3)
      if p >= precision:
        h = p /cLat -N
        lat = math.atan(x[2] /p /(1 -Common._EARTH_ECCENT2 *N/(N +h)))
      elif lat >= 0.0:
        h = x[2] -Common.EARTH_MINOR_M
        lat = math.pi *0.5
      else:
        h = x[2] +Common.EARTH_MINOR_M
        lat = -math.pi *0.5
      if abs(lat -lat0) < precision:
        break
    return Matrix(math.degrees(lat),
                  math.degrees(math.atan2(x[1], x[0])), h)

  def euler(self, noError=False):
    """euler([noError])
       Versors can be converted to Euler Angles & back uniquely for default
       order. For n-D greater than 3 need to extract sine terms from the last
       basis at each rank and reduce by multiplying by the inverse for each rank
       until 3-D is reached. See Common.Euler.Matrix. Once 3-D is reached the
       quaternion.euler() calculation can be used. Euler parameters are of the
       form cos(W/2) +n sin(W/2), n pure unit versor. Set noError to return a
       zero Euler if self is not valid to be a versor or norm if possible.
       Also signature must be positive."""
    tmp,flat,n2,cnt = self.__versor()
    l2 = n2 +self.w *self.w
    if flat != 0 or cnt != 1:
      if not noError:
        raise Exception("Illegal versor for euler")
    if abs(l2 -1.0) >= Common._getPrecision():
      if not noError:
        raise Exception("Illegal versor norm for euler")
      tmp = tmp.norm()
    if n2 < Common._getPrecision():
      return Euler()
    dims = self.__vectorSizes()
    xyz = CA.VersorArgs(*dims, rotate=True)
    cnt = len(xyz)
    angles = [0] *cnt
    for rank in reversed(range(4, dims[0] +1)):
      base = CA(**{"e%X" %rank: 1})
      mul = tmp.rotate(base)
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
        s,c = Common._sincos(val *0.5)
        mul *=  CA(c, **{xyz[idx]: -s})
      tmp = mul *tmp
    args = [0] *3
    xyz = CA.VersorArgs(3)
    for grade in tmp.__g:
      eStr,iStr = grade.strs()
      if eStr in xyz:
        idx = xyz.index(eStr)
        args[idx] = -grade.value if idx != 1 else grade.value #rotate
    w, x, y, z = tmp.w, args[2], args[1], args[0] # rotate
    disc = w *y - x *z
    if abs(abs(disc) -0.5) < Common._getPrecision():
      sgn = 2.0 if disc < 0 else -2.0
      angles[0] = sgn *math.atan2(x, w)
      angles[1] = -math.pi /sgn
      angles[2] = 0.0
    else:
      angles[0] = math.atan2(2.0 * (z * y + w * x), 1.0 - 2.0 * (x * x + y * y))
      angles[1] = math.asin(2.0 * (y * w - z * x))
      angles[2] = math.atan2(2.0 * (z* w + x * y), 1.0 - 2.0 * (y * y + z * z))
    return Euler(*angles)

  def versorMatrix(self, noError=False):
    """versorMatrix([noError])
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
       noError to norm the versor if not normed."""
    return self.euler(noError).matrix()

  def frameMatrix(self):
    """frameMatrix()
       Rodriges for n-D. See https://math.stackexchange.com/questions/1288207/
       extrinsic-and-intrinsic-euler-angles-to-rotation-matrix-and-back for
       3-D. Converts self to versor then applies each even part."""
    return self.versor().euler().matrix()

  def basisTerms(self):
    """basisTerms()
       Return self as 3 lists = a list of e-basis indicies, values & i-basis."""
    out1,out2,out3 = [],[],[]
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

  def perm(self, cycle):
    perms = list(x+1 for x in range(7))
    for idx,val in enumerate(cycle):
      perms[val -1] = cycle[idx +1] if idx < len(cycle) -1 else cycle[0]
    out = CA(self.w)
    for term in self._basisTerms()[1]:
      newTerm = term[:]
      for idx,per in enumerate(perms):
        if idx +1 in term:
          newTerm[term.index(idx +1)] = per
      out += CA.Eval(newTerm)
    return out

  def swap(self, basisTerms, signTerms=[]):
    """swap(basisTerms,[signTerms])
       Morphism with a list of terms that contain even lists taken as pairs of
       basis dimension integers, strings or basis with (1,2) or e12 meaning
       map e13 -> -e23 and e23 -> e13 in self. Hence this is a rotation of 90
       degrees with the advantage that rounding errors do not need to be
       trimmed. Terms as lists of indicies are added without sign and a single
       term may contain indicies directly."""
    if isinstance(basisTerms, CA):
      basisTerms = basisTerms.basisTerms()
      if signTerms:
        raise Exception("Swap signTerms only valid for list basisTerms")
      if basisTerms[2] and basisTerms[2][0]:
        raise Exception("Swap basisTerms can't be imaginary")
      signTerms = basisTerms[1]
      basisTerms = basisTerms[0]
    else:
      Common._checkType(signTerms, (list, tuple), "swap")
      signTerms = signTerms +[1] *(len(basisTerms) -len(signTerms))
    Common._checkType(basisTerms, (list, tuple), "swap")
    if len(basisTerms) == 0:
      out = self.copy() 
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
            if isinstance(base, Common._basestr) and len(base) == 1:
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

  def morph(self, pairs):
    """morph(pairs)
       Morphism with a list of name pairs as ("e1","e2") to map only e1->e2."""
    out = CA(self.w)
    out.__entered0 = self.__entered0
    for grade in self.__g:
      out += CA(**Common._morph(grade.strs(), grade.value, pairs))
    return out

  def allSigns(self, half=False):
    """allSigns([half])
       Generate a list of all, half [boolean] or a single indexed term [half=
       int] of the signed combinations of self, (eg allSigns(e1)=[e1,-e1])."""
    stopCnt = -1
    if not isinstance(half, bool):
      stopCnt = half
      half = False
    Common._checkType(half, bool, "allSigns")
    dim = sum(self.grades()[1:])
    terms = self.copyTerms()
    halfStop = (half and dim %2 == 0)
    halfDim = (int(dim /2) if half else dim)
    halfComb = int(Common.comb(dim, halfDim) /2)
    for n in range(halfDim +1):
      for cnt,sgns in enumerate(Common.comb(dim, n, True)): # For n -sign combos
        if n == halfDim and halfStop and cnt == halfComb:
          break
        stopCnt -= 1
        if stopCnt == -1:
          break
        p0 = list((CA(**dict((x,))) for x in terms))
        for sgn in sgns:
          p0[sgn -1] *= -1
        yield sum(p0)
      if stopCnt == -1:
        break

  def allSignsIndicies(self):
    """allSignsIndicies()
       Return index and minus sign count of self in allSigns."""
    cnt,sgns = 0,[]
    for idx,term in enumerate(self.copyTerms()):
      if term[1] < 0:
        sgns.append(idx +1)
      cnt += 1
    offs = 0
    for dim in range(len(sgns)):
      offs += Common.comb(cnt, dim)
    for idx,allSgns in enumerate(Common.comb(cnt, len(sgns), True)):
      if allSgns == sgns:
        break
    return idx +offs, len(sgns)

  def spin(self, basis=[]):
    """spin([basis])
       Return the Common.Table triad list and Basis list if basis else
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
      Common._checkType(basis, (list, tuple), "spin")
      if len(basis) != maxBasis:
        raise Exception("Invalid basis length for spin")
    else:
      sBasis = CA.VersorArgs(maxBasis)
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
  def Basis(eDim, iDim=0, parity=0, maxGrade=0):
    """Basis(eDim, [iDim, parity,maxGrade])
       Yield (e,i) basis elements with value one. Parity=0:all,1:odd,2:even."""
    Common._checkType(eDim, int, "Basis")
    Common._checkType(iDim, int, "Basis")
    Common._checkType(parity, int, "Basis")
    if parity:
      if parity <0 or parity > 2:
        raise Exception("Invalid parity for Basis")
      for base in CA.BasisArgs(eDim, iDim, maxGrade):
        odd = 1 if 'e' in base else 0
        odd += 1 if 'i' in base else 0
        if (len(base) +odd +parity) %2 == 0:
          yield CA(**{base: 1})
    else:
      for base in CA.BasisArgs(eDim, iDim, maxGrade):
        yield CA(**{base: 1})

  @staticmethod
  def BasisArgs(eDim, iDim=0, maxGrade=0):
    """BasisArgs(eDim, [iDim,maxGrade])
       Yield (e,i) basis elements as a list of names in addition order."""
    Common._checkType(eDim, int, "BasisArgs")
    Common._checkType(iDim, int, "BasisArgs")
    if eDim > CA.__HEX_BASIS or iDim > CA.__HEX_BASIS:
      raise Exception("Too many basis arguments")
    if eDim < 0 or iDim < 0:
      raise Exception("Too few basis arguments")
    for n in range(1, eDim +iDim +1):
      if maxGrade > 0 and n > maxGrade:
        break
      for i in range(n +1):
        if eDim >= n-i and iDim >= i:
          oute = tuple(("e" +x for x in CA._basisArgs(eDim, n-i)))
          outi = tuple(("i" +x for x in CA._basisArgs(iDim, i)))
          for out in Common._mergeBasis(oute, outi):
            yield out

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
    Common._checkType(eDim, int, "VersorArgs")
    Common._checkType(iDim, int, "VersorArgs")
    Common._checkType(rotate, bool, "VersorArgs")
    if eDim +iDim < 2 or eDim < 0 or iDim < 0 or \
       eDim > CA.__HEX_BASIS or iDim > CA.__HEX_BASIS:
      raise Exception("Invalid VersorArgs argument size")
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
    for i in range(1, eDim):
      for j in range(i+1, iDim+1):
        out.append("e%Xi%X" %(i,j))
    return out

  @staticmethod
  def Versor(*args, **kwargs):
    """Versor([scalar, e1 multiplier, ...][basis=multiplier, ...])
       Return versor(2-D +...) where ... is higher dimensions in the
       form e32=x, e13=y, e21=z, .... Each dimension has (D 2)T=D(D-1)/2
       parameters and these are added as e4 to xyz, e5 to these +e45, etc.
       Use VersorArgs() to see this list. See Euler() for an angle version
       instead of parameters being n sin(W/2), n unit."""
    # See Wikipedia.org rotations in 4-dimensional Euclidean space
    if args:
      dim = int((math.sqrt(8*(len(args)-1) +1) +1) /2 +0.9) # l=comb(dim,2)
      if dim > CA.__HEX_BASIS:
        raise Exception("Invalid number of Versor euler angles")
      xyz = CA.VersorArgs(dim, rotate=True)
      for idx,val in enumerate(args[1:]):
        if xyz[idx] in kwargs:
          raise Exception("Invalid Versor basis duplication: %s" %xyz[idx])
        kwargs[xyz[idx]] = val
      args = args[:1]
    ca = CA(*args, **kwargs)
    return ca.versor()

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
    Common._checkType(order, (list, tuple), "Euler")
    Common._checkType(implicit, bool, "Euler")
    for key in kwargs:
      if key not in ("order", "implicit"):
        raise TypeError("Euler() got unexpected keyword argument %s" %key)
    if len(args) == 1 and isinstance(args[0], Euler):
      args = list(args[0])
    out = CA(1.0)
    implicitRot = CA(1.0)
    store = []
    dim = 2
    xyz = CA.VersorArgs(dim, rotate=True)
    if args:
      dim = int((math.sqrt(8*len(args) +1) +1) /2 +0.9) # l=comb(dim,2)
      if dim > CA.__HEX_BASIS:
        raise Exception("Invalid number of Euler angles")
      xyz = CA.VersorArgs(dim, rotate=True)
      for idx,val in enumerate(args):
        if xyz[idx] in kwargs:
          raise Exception("Invalid Euler basis duplication: %s" %xyz[idx])
    xyz_in = CA.VersorArgs(3)
    for bi,val in kwargs.items():
      if bi not in ("order", "implicit"):
        while bi not in xyz +xyz_in:
          dim += 1
          if dim > CA.__HEX_BASIS:
            raise Exception("Invalid Euler parameter: %s" %bi)
          xyz = CA.VersorArgs(dim, rotate=True)
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
      Common._checkType(key, (float, int), "Euler")
      key = int(key)
      if key in store or key <= 0 or key > len(args):
        raise Exception("Invalid order index for rotation: %s" %key)
      ang = args[key -1]
      Common._checkType(ang, (int, float), "Euler")
      s,c = Common._sincos(ang *0.5)
      rot = CA(c, **{xyz[key -1]: s})
      if implicit:
        tmpRot = rot.copy()
        rot.rotation(implicitRot)
        implicitRot *= tmpRot
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
    Common._checkType(lat, (int, float), "LatLon")
    Common._checkType(lng, (int, float), "LatLon")
    sLat,cLat = Common._sincos(math.radians(lat))
    sLng,cLng = Common._sincos(math.radians(lng))
    major = Common.EARTH_MAJOR_M
    minor = Common.EARTH_MINOR_M
    latParametric = math.atan2(minor *sLat, major *cLat)
    sLat,cLat = Common._sincos(latParametric)
    xMeridian = major *cLat
    return CA(0, e1=xMeridian *cLng, e2=xMeridian *sLng, e3=minor *sLat)

  @staticmethod
  def LatLonAlt(lat, lng, alt=0):
    """LatLonAlt(lat, lng, [alt])
       Return Earth Centred, Earth Fixed (ECEF) vector for geodetic WGS-84 
       lat(deg)/long(deg)/altitude(m). From fossen.biz/wiley/pdf/Ch2.pdf.
       EarthPolar/EarthMajor = sqrt(1-e*e), e=eccentricity."""
    Common._checkType(lat, (int, float), "LatLonAlt")
    Common._checkType(lng, (int, float), "LatLonAlt")
    Common._checkType(alt, (int, float), "LatLonAlt")
    sLat,cLat = Common._sincos(math.radians(lat))
    sLng,cLng = Common._sincos(math.radians(lng))
    ee3 = 1 -Common._EARTH_ECCENT2
    N = Common.EARTH_MAJOR_M /math.sqrt(cLat *cLat +sLat *sLat *ee3)
    return CA(0, e1=(N +alt) *cLat *cLng, e2=(N +alt) *cLat *sLng,
                   e3=(N *ee3 +alt) *sLat)
  @staticmethod
  def NED(lat, lng):
    """NED(lat, lng)
       Lat/long Earth Centred-Earth Fixed (ECEF) to North-East-Down (NED)
       frame. Return a versor to perform this rotation. The inverse changes
       from NED to ECEF."""
    Common._checkType(lat, (int, float), "NED")
    Common._checkType(lng, (int, float), "NED")
    sLat,cLat = Common._sincos(math.radians(-lat -90) *0.5)
    sLng,cLng = Common._sincos(math.radians(lng) *0.5)
    return CA(cLng *cLat, e32=-sLng *sLat, e13=cLng *sLat, e21=sLng *cLat)

  @staticmethod
  def FrameMatrix(mat):
    """FrameMatrix(mat)
       Return the CA for a 3-D frame matrix ie opposite of frameMatrix()
       for a unit vector except that angles outside 90 deg are disallowed.
       tr(mat) = 2cosW +1. If W=0 R=Id. If W=pi R=?."""
    Common._checkType(mat, (Matrix, Tensor), "FrameMatrix")
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
    xyz = CA.VersorArgs(3, rotate=True)
    kw = {} 
    for idx,val in enumerate(xyz):
      kw[val] = args[idx] /a
    return CA(W, **kw)

  @staticmethod
  def Eval(sets):
    """Eval(sets)
       Return the CA evaluated from sets of e-basis, str lists or dict pairs."""
    Common._checkType(sets, (list, tuple), "Eval")
    if not isinstance(sets[0], (list, tuple)):
      sets = [sets]
    terms = {}
    for item in sets:
      Common._checkType(item, (list, tuple), "Eval")
      if isinstance(item, Common._basestr):
        base = item[0] if item[0][0] in CA.__allChars else ("e" +item[0])
        terms[base] = 1
      elif not isinstance(item, (list, tuple)):
        raise Exception("Invalid basis for Eval: %s" %item)
      elif isinstance(item[0], Common._basestr):
        base = item[0] if item[0][:1] in CA.__allChars else ("e" +item[0])
        terms[base] = item[1]
      else:
        base = "e"
        sgn = 1
        for num in item:
          Common._checkType(num, int, "Eval")
          base += "%X" %abs(num)
          if num < 0:
            sgn *= -1
        terms[base] = sgn
    return CA(**terms)

  @staticmethod
  def Spin(triList):
    basis = []
    for x in triList:
      basis.append(CA.Eval([x[0:2]]))
      basis.append(CA.Eval([x[1:3]]))
      basis.append(CA.Eval([[x[2], x[0]]]))
    return basis

  @staticmethod
  def Q(*args):
    """Q([scalar, x, y, z])
       Map quaternion basis (w,i,j,k) to (w, e32, e13, e21) with up to 4
       arguments. If calc(Q) included then scalar may instead be a Q object."""
    if CA.__useQuat:   # If module calcQ included can use Q class
      if len(args) == 1 and isinstance(args[0], Q):
        q = args[0]
        args = [q.w, q.x, q.y, q.z]
    if len(args) > 4:
      raise Exception("Invalid Q parameters")
    xyz = CA.VersorArgs(3, rotate=True)
    kw = {} 
    for idx,val in enumerate(xyz):
      kw[val] = 0 if len(args) < 4 else args[idx +1]
    return CA(0 if len(args) < 1 else args[0], **kw)

  @staticmethod
  def IsCalc(calc):
    """Check if calcQ has been loaded."""
    if calc == "CA": return True
    return (calc == "Q" and CA.__useQuat)

  ###################################################
  ## Calc class help and basis processing methods  ##
  ###################################################
  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for CA."""
    calcHelp = """Clifford Algebra Calculator - Process 30-dimensional basis
          numbers (e0..F or i0..F) and multiples."""
    return (("CA", "Q", "R"), ("CA", "math"), "default.ca", calcHelp, "")

  @classmethod
  def _setCalcBasis(cls, calcs, dummy):
    """Load other calculator. If quaternions are loaded then convert
       i,j,k into Q() instead of e32,e13,e21. Default True."""
    if "Q" in calcs:
      cls.__useQuat = True
      for i in cls.__QUAT_CHARS:
        if i not in cls.__allChars:
          cls.__allChars.append(i)
    if cls.__useQuat:
      return "CA calculator has Quaternions enabled"
    return ""

  @classmethod
  def _validBasis(cls, value):
    """Used by Calc to recognise repeated basis forms e... and i...
       and quaternions basis i, j, k because this module is optional 
       and we don't want files & tests to break."""
    if len(value) == 1:
      return 1 if value in cls.__QUAT_CHARS else 0
    if value[0] not in cls.__allChars:
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
    """Convert the store array into CA(...) or Q(...) python code. Convert to
       Q(...) if basis is i, j, k and __useQuat else expand to bivectors. If
       isMults1/2 set then double up since CA to Q or MULTS are higher priority
       then SIGNS. The state is a ParseState from Calculator.processTokens()."""
    kw = {}
    line = ""
    isQuat = False
    quatKeys = (None, 'i', 'j', 'k')  # based on __QUAT_KEYS
    signTyp = ""
    firstCnt = 1 if state.isMults1 else -1
    lastCnt = len(state.store) -1 if state.isMults2 else -1
    for cnt,value in enumerate(state.store):
      val,key = value
      if key in cls.__QUAT_CHARS:
        isQuat = cls.__useQuat   # The same validBasis type
        if not isQuat:
          xyz = cls.VersorArgs(3, rotate=True)
          key = xyz[cls.__QUAT_CHARS.index(key)]

      # If basis already entered or single so double up or accum scalar
      isMult = (cnt in (firstCnt, lastCnt) and lastCnt != 0)
      if key in kw or isMult:
        if key is None and not isMult:  # Duplicate scalar
          val = "(%s%s)" %(kw[None], val)
        elif isMult and len(kw) == 1 and None in kw:
          line += signTyp +kw[None]
          signTyp = "+"
          kw[None] = "0"
        elif isQuat:
          line += signTyp +"Q(%s)" %",".join( \
                  ("%s" %kw[x] if x in kw else '0') for x in quatKeys)
          signTyp = "+"
          kw = {}
        else:
          scalar = ""
          if None in kw:
            scalar = "%s," %kw[None]
            del(kw[None])
          line += signTyp +"CA(%s%s)" %(scalar, ",".join(map(lambda x: \
                  ("%s=%s" %(x,kw[x])) if x else str(kw[x]), kw.keys())))
          signTyp = "+"
          kw = {}
        isQuat = False
      kw[key] = val

    # Dump the remainder
    if len(kw) == 1 and None in kw:
      line += signTyp +kw[None] +" "
    elif isQuat:
      line += signTyp +"Q(%s)" %",".join( \
              ("%s" %kw[x] if x in kw else '0') for x in quatKeys)
    else:
      scalar = ""
      if None in kw:
        scalar = "%s, " %kw[None]
        del(kw[None])
      line += signTyp +"CA(%s%s)" %(scalar, ",".join(map(lambda x: \
              ("%s=%s" %(x,kw[x])) if x else str(kw[x]), kw.keys())))
    if state.isNewLine:
      line += '\n'
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
  try:
    import importlib
  except:
    importlib = None          # No changing of calculators

  # CA Unit test cases for Calc with Tests[0] being init for each case
  Tests = [\
    """d30=radians(30); d60=radians(60); d45=radians(45); d90=radians(90)
       e=Euler(pi/6,pi/4,pi/2); c=e1+2e2+3e3; c.basis(3)""",
    """# Test 1 Rotate via frameMatrix == versor half angle rotation.
       Rx=d60+e32; rx=(d60 +e32).versor()
       test = Rx.frameMatrix() *c.vector(); store = (rx*c*rx.inverse()).vector()
       Calculator.log(store == test, store)""",
    """# Test 2 Rotate via frameMatrix == versor.rotate(half angle)].
       Rx=d60+e13; rx=CA.Versor(d60,0,1)
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
       if CA.IsCalc("Q"):
         test = CA.Q(Q.Euler(e, order=[1,2,3], implicit=True))
       else:
         test = CA.Euler(e, order=[3,2,1])
         Common.precision(1E-12)
       store = CA.Euler(e, order=[1,2,3], implicit=True)
       Calculator.log(store == test, store)""",
    """# Test 6 Versor squared == exp(2*log(e)).
       test = CA.Euler(e).pow(2); store = (CA.Euler(e).log() *2).exp()
       Calculator.log(store == test, store)""",
    """# Test 7 Rotate via frameMatrix == versor.versorMatrix(half angle).
       if CA.IsCalc("Q"):
         test = (d45+i+j+k).frameMatrix()
       else:
         test = (d45+i+j+k).frameMatrix()
       store = (d45+i+j+k).versor().versorMatrix()
       Calculator.log(store == test, store)""",
    """# Test 8 Rotate via versor.versorMatrix() == versor.euler().matrix().
       r = d45 +i +j +k; store = r.norm().euler().matrix()
       if CA.IsCalc("Q"):
         test = r.norm().versorMatrix()
       else:
         test = r.norm().versorMatrix()
       Calculator.log(store == test, store)""",
    """# Test 9 Euler Matrix is inverse of versorMatrix.
       test=Matrix(pi/6, pi/4, pi/2)
       store=Euler.Matrix(CA.Euler(*test).versorMatrix())
       Calculator.log(store == Euler(*test), store)""",
    """# Test 10 Geodetic distance = acos(p.w *d.w -p.dot(d)).
       if CA.IsCalc("Q"):
         p = Q.Euler(e); d=(d45+i+2j+3k).versor()
         test = math.acos(p.w *d.w -p.dot(d))
         p = CA.Q(p); d = CA.Q(d)
       else:
         p = CA.Euler(e); d=(d45+e321*c).versor()
         test = math.acos(p.w *d.w -p.pure(2).sym(d.pure(2)).scalar())
       store = p.distance(d)
       Calculator.log(abs(store - test) < 3E-5, store)""",
    """# Test 11 Length *2 == dot(self +self).
       store = (c *2).len(); test = math.sqrt((c +c).dot(c +c))
       Calculator.log(abs(store - test) <1E-15, store)""",
    """# Test 12 Versor *3 /3 == versor.norm
       Calculator.log(c/c.len() == c.norm(), c.norm())""",
    """# Test 13 Check Rodriges formula
       def para(a,r,w): return a *a.dot(r)
       def perp(a,r,w): return r *math.cos(w) +CA(e321=1)*a.asym(r) \\
               *math.sin(w) -a *a.dot(r) *math.cos(w)
       store = para(e1,e1+e2,d30)+perp(e1,e1+e2,d30)
       if CA.IsCalc("Q"):
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
       if CA.IsCalc("Q"):
         test = Q.Euler(pi/6, pi/4, pi/2).versorMatrix()
       else:
         test = CA.Euler(pi/6, pi/4, pi/2).versorMatrix()
       store = Euler(pi/6, pi/4, pi/2).matrix()
       Calculator.log(store == test, store)""",
    """# Test 16 Check lat-long conversion to ECEF xyz and back.
       lat=45; lng=45; store = Tensor(lat,lng); Common.precision(1E-8)
       if CA.IsCalc("Q"):
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
       if CA.IsCalc("Q"):
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
       model2NED = (egNED * model).norm()
       ECEF = (CA.NED(lat, lng) * model2NED).norm()
       store = ECEF.versorMatrix()
       test = Tensor(\
         ( 0.552107290714106247, 0.63168203529742073, -0.544201567273411735),\
         (-0.341090423374471263,-0.424463728320773503,-0.838741835383363443),\
         (-0.760811975866937606, 0.648697425324424537,-0.0188888261919459843))
       Calculator.log(store == test, store)""",
    """# Test 21 CA.Euler.euler 7-D is same as Euler.
       test = Matrix(list((x *0.01 for x in range(1,22))))
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