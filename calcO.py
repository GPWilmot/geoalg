#!/usr/bin/env python3
################################################################################
## File: calcO.py needs calcR.py and file is part of GeoAlg.
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
## CalcO is a commnd line calculator that converts basis numbers into
## Octonians, Sedonions, Quaternions, Versors and Euler Angle rotations.
## CA numbers and Quaternions can also be included and processed separately
## which allows Dixon Algebra to be defined using the Tensor class. 
##
## O multivectors are of the form p = w +p o1..F u1..F +....
## A quaternion maps as i->o1, j->o2, k->-o12. u basis has the same 
## multiplication table as o basis except the squares are positive unity.
## Quaternions are of the form q = w + _v_ where _v_ is a i,j,k vector
## s.t. _v_*_v_ >= 0 and w is a scalar. A pure quaternion has w==0.
## A unit quaternion has _n_*_n_ = -1. A versor has norm = 1 which means
## r = cos(a) + sin(a) * _n_ where _n_ in a unit. This is extended to
## Octonians for basis number 3 and Sedonians for numbers 4 and above.
## The calc(Q) command interprets i, j, k using Q class and the calc(CA)
## command interprets CA numbers via CA class. Need to avoid name conflicts
## in loaded files in the later case.
## Start with either calcO.py, python calcO.py or see ./calcR.py -h.
################################################################################
__version__ = "0.1"
import math
from calcCommon import *

################################################################################
class O():
  """Class to process Octonians and Unity Octionains consisting of 32-D numbers,
     half being positive definite (u1..uF) and half being standard imaginary
     Octonians (o1..oF). O(q,p) specifies q the Octonian dimension (o<hex>) and
     p specifies the dimension of Unitary Octonians. The later have the same
     multiplication table as Octonians but with squares being unity (see Tensor.
     Table(O.Basis(0,3)). The notation is the same as for CA so o1*o2=-o12 but
     the basis is ungraded so o12 is a single imaginary entity and o21 is
     disallowed. This allows easy extension to arbitray Sedonian Algebras (O(q)
     with q>4) and justifies the dimension terninology.

     O(1) is complex numbers, O(2) is quaternions, O(3) is Octonians. Just enter
     arbitrary elements and the highest basis dimensions are remembered for
     producing matricies. Package math methods only work on the scalar part
     eg sin(o.scalar()).
     """
  __HEX_BASIS   = 16                     # e and i basis size
  __HEX_CHARS   = ('A', 'B', 'C', 'D', 'E', 'F')
  __QUAT_CHARS  = ('i', 'j', 'k')        # Quaternion basis chars
  __CA_CHARS    = ('e', 'i')             # CA basis chars only
  __BASIS_CHARS = ('o', 'u')             # O basis chars only
  __allChars    = ['o', 'u']             # Include CA & Quaternions
  __maxBasis    = ['0', '0']             # Store the max dimensions
  __useQuat     = False                  # Q class is loaded
  __useCA       = False                  # CA class is loaded
  __basisXyz    = ("",)                  # Cache maximum basis order
  __basisDim    = 0                      # Cache maximum basis size
  __basisCache  = []                     # Cache partial multiplication table
  __wikiMulRule = False                  # Change Cayley-Dickson multiply rule
  dumpRepr      = False                  # Repr defaults to str

  class Product(list):
    """Cayley-Dickson expansion as recursive binary tree to arbitrary level.
       (a,b)*(c,d) = (a*c-d.conj*b, d*a +b*c.conj) [Wikipedia] or 
       (a,b)*(c,d) = (a*c-d*b.conj, a.conj*d +c*b) [J.C.Baez]."""
    def __init__(self, n, lhs, rhs=None):
      """Define lhs as binary tree (level n>=0) or multiply lhs * rhs (n<0)."""
      if n >= 0:
        if rhs or not (isinstance(lhs, list) and len(lhs) == pow(2,n)):
          raise Exception("Create Product with list of %d numbers" %pow(2,n))
        if n == 0:
          super(O.Product, self).__init__((0, lhs[0], 0))
        elif n == 1:
          super(O.Product, self).__init__((1, lhs[0], lhs[1]))
        else:
          half = int(pow(2, n)) //2
          p = O.Product(-n, lhs[:half], lhs[half:])
          super(O.Product, self).__init__((n, p[1], p[2]))
      elif n == -1:
        n = 1
        if isinstance(lhs, O.Product):
          n = lhs[0] +1
        super(O.Product, self).__init__((n, lhs, rhs))
      elif n == -2:
        pL = O.Product(-1, lhs[0], lhs[1])
        pR = O.Product(-1, rhs[0], rhs[1])
        super(O.Product, self).__init__((2, pL, pR))
      else:
        half = int(pow(2, -n -1)) //2
        pL = O.Product(n +1, lhs[:half], lhs[half:])
        pR = O.Product(n +1, rhs[:half], rhs[half:])
        super(O.Product, self).__init__((-n, pL, pR))
    def __str__(self):
      """Overload string output by dumping tree recursively."""
      return "P(%s,%s:%d)" %(self[1], self[2], self[0])
    __repr__ = __str__
    def __mul__(self, p):
      """Product n => (a,b)(c,d) = (ac-db*, a*d+cb) for conjugate level n-1.
         This is baezRule - for wikiRule see _mulWiki."""
      n = self[0]
      if not isinstance(p, O.Product) or n != p[0]:
        raise Exception("Invalid Product to multiply: %s * %s" %(self, p))
      if n == 0:
        return self[1] * p[1]
      # (p,q)*(r,s) = (pr -sq*, p*s +rq) [baezRule]."""
      if n == 1:
        return O.Product(-1, self[1] *p[1] -p[2] *self[2],
                             self[1] *p[2] +p[1] *self[2])
      return O.Product(-1, self[1] *p[1] -p[2] *self[2].__conj(n-1),
                           self[1].__conj(n-1) *p[2] +p[1] *self[2])
    def __add__(self, p):
      """Return new Product adding tree recursively."""
      return O.Product(-1, self[1] +p[1], self[2] +p[2])
    def __sub__(self, p):
      """Return new Product subtracting tree recursively."""
      return O.Product(-1, self[1] -p[1], self[2] -p[2])
    def __neg__(self):
      """Return new Product negating tree recursively."""
      return O.Product(-1, -self[1], -self[2])
    def __conj(self, n):
      """Conjugate level n => (a,b)*n = (a*(n-1), -b) recursive."""
      if self[0] == n:
        if n > 1:
          return O.Product(-1, self[1].__conj(n -1), -self[2])
        return O.Product(-1, self[1], -self[2])
      return self
    def _mulWiki(self, p):
      """Product n => (a,b)(c,d) = (ac-d*b, da+bc*) for conjugate level n-1.
         This is wikiRule - for baezRule see __mul__."""
      n = self[0]
      if not isinstance(p, O.Product) or n != p[0]:
        raise Exception("Invalid Product to multiply: %s * %s" %(self, p))
      if n == 0:
        return self[1]._mulWiki(p[1])
      # (p,q)*(r,s) = (pr -s*q, sp +qr*) [wikiRule]
      if n == 1:
        return O.Product(-1, self[1] *p[1] -p[2] *self[2],
                             p[2] *self[1] +self[2] *p[1])
      return O.Product(-1,
                     self[1]._mulWiki(p[1]) -p[2].__conj(n-1)._mulWiki(self[2]),
                     p[2]._mulWiki(self[1]) +self[2]._mulWiki(p[1].__conj(n-1)))
    def walk(self, array, idx=0):
      """Walk through tree filling array with index values."""
      if isinstance(self[1], O.Product):
        idx = self[1].walk(array, idx)
        idx = self[2].walk(array, idx)
      else:
        if self[1]:
          array[idx] = self[1]
        if self[2]:
          array[idx +1] = self[2]
        idx += 2
      return idx

  class Grade:
    """Each O has a list of o & u basis elements ordered by Product index.
       Each basis is an index into existing _basisArgs()."""
    def __init__(self, value, bases):
      """Element with scalar and p, o & u bases as Product index & BasisArgs."""
      self.value = value
      self.__pBase = bases[0]
      self.__oBase = bases[1]
      self.__uBase = bases[2]
      self.__unitaryErr = "Can't mix octonian signatures with same index"
    def bases(self):
      return (self.__pBase, self.__oBase, self.__uBase)
    def lens(self):
      return (len(self.__oBase), len(self.__uBase))
    def strs(self, oCh='o', uCh='u'):
      return ((oCh +self.__oBase) if self.__oBase else "",
              (uCh +self.__uBase) if self.__uBase else "")
    def __str__(self):
      return "%s[%s,%s]" %(self.value, self.__oBase, self.__uBase)
    __repr__ = __str__
    def __mergeStr(self, lhs, rhs):
      out = ""
      idx1 = 0
      idx2 = 0
      while True:
        if idx1 < len(lhs):
          if idx2 < len(rhs):
            if lhs[idx1] < rhs[idx2]:
              out += lhs[idx1]
              idx1 += 1
            elif lhs[idx1] > rhs[idx2]:
              out += rhs[idx2]
              idx2 += 1
            else:
              idx1 += 1
              idx2 += 1
          else:
            out += lhs[idx1]
            idx1 += 1
        elif idx2 < len(rhs):
          out += rhs[idx2]
          idx2 += 1
        else:
          break
      return out
    def copy(self):
      return O.Grade(self.value, (self.__pBase, self.__oBase, self.__uBase))

    def isEq(self, cf, precision):
      """Return true if the grades are equal within precision."""
      return abs(self.value -cf.value) < precision \
             and self.__oBase == cf.__oBase and self.__uBase == cf.__uBase

    def order(self, cf):
      """Find the placement of a single O term in self taking into account
         the base signature and sign change under swapping."""
      if self.__pBase < cf.__pBase:
        return -1
      if self.__pBase > cf.__pBase:
        return 1
      if self.__oBase < cf.__oBase:
        return -1
      if self.__oBase > cf.__oBase:
        return 1
      if self.__uBase < cf.__uBase:
        return -1
      if self.__uBase > cf.__uBase:
        return 1
      return 0

    def mergeBasis(self, value, rhs):
      """Multiply graded basis self by rhs as one row due to the definition of
         the product. This is done row by row of the table using successive
         integer values to separate index positions. Rows are cached and the
         order doesn't change as rows and columns increase is size, This works
         for the sparse multiplication used by the graded lists."""
      value *= self.value
      lhs = self.bases()
      loStr = roStr = ""
      xyz = None
      bases = [0, "", ""]     # Base for lhs o and u, resp
      rBase = rhs[0] # Iterate rhs o and u
      lBase = lhs[0]
      hasU = lhs[2] or rhs[2]
      if rBase:
        if lBase:
          val1 = value
          row = O._basisCache(lBase)
          if not row or len(row) <= rBase:
            xyz, maxDim, wikiMul = O._basisArgs()
            lp = [0] *len(xyz)
            rp = list((x for x in range(len(xyz))))
            lp[lBase] = 1
            walk = [0] *len(xyz)
            row = [0] *len(xyz)
            if wikiMul:
              prod = O.Product(maxDim, lp)._mulWiki(O.Product(maxDim, rp))
            else:
              prod = O.Product(maxDim, lp) * O.Product(maxDim, rp)
            prod.walk(walk)
            for idx,val in enumerate(walk):
              if idx > 0:
                row[abs(val)] = idx if val >= 0 else -idx
            O._basisCache(lBase, row)
          idx = row[rBase]
          bases[0] = abs(idx)
          if idx <= 0:
            value = -value
          bases[1] = self.__mergeStr(lhs[1], rhs[1])

          if hasU:   # Split multiplication
            bases[2] = self.__mergeStr(lhs[2], rhs[2])
            for lCh in lhs[2]:
              if rhs[2].find(lCh) >= 0:
                value = -value
              if rhs[1].find(lCh) >= 0:
                raise Exception(self.__unitaryErr)
            for rCh in rhs[2]:
              if lhs[1].find(rCh) >= 0:
                raise Exception(self.__unitaryErr)
        else:
          bases = rhs
          if hasU:
            for rCh in rhs[2]:
              if lhs[1].find(rCh) >= 0:
                raise Exception(self.__unitaryErr)
      else:
        bases = lhs
      return O.Grade(value, bases)

  ##############################################################################
  ## Class overwritten functionality methods
  ##############################################################################
  def __init__(self, *args, **kwargs):
    """O([scalar, o1 multiplier, ...][basis=multiplier, ...])
       The scalar is followed by values for each octonian in BasisArgs order.
       Basis elements can also be entered in dictionary form as o<hex>=<value>.
       Higher and unitary grades can also be entered eg: o12u1=1.0. Repeated
       bases are not allowed and hex digits must be increasing. See Basis and
       BasisArgs for a list of basis numbers and names."""
    self.w = 0.0 if len(args) == 0 else args[0] # Scalar
    Common._checkType(self.w, (int, float), "O")
    self.__g = []                               # Array of ordered Grades
    self.__currentAdd = -1                      # Previous add index
    if len(args) > 1:
      dim = int(math.log(len(args))/math.log(2) +1)       # l=pow(2,dim)
      if dim > self.__HEX_BASIS +1:
        raise Exception("Too many basis elements")
      xyz = O._basisArgs(dim)[0]   # Setup global dimension
      for idx,val in enumerate(args[1:]):
        Common._checkType(val, (int, float), "O")
        if val:
          self.__g.append(O.Grade(val, [idx +1, xyz[idx +1], ""]))
        if xyz[idx][-1:] > O.__maxBasis[0]:
          O.__maxBasis[0] = xyz[idx][-1]
    for key,value in kwargs.items():
      Common._checkType(value, (int, float), "O")
      if value:
        lGrade = O._init(key, value)
        self.__add(lGrade)

  @staticmethod
  def _init(key, value):
    """Return the Grade for basis string key and value +/-1."""
    typ = None
    rBases = [0, "", ""]
    lBase = ""
    rBase = ""
    typ = None
    lastChar = '0'
    for char in key:
      offset = int(typ == O.__BASIS_CHARS[1]) # o==0, u==1
      if typ and char.isdigit():
        if char <= lastChar:
          raise Exception("Invalid basis: %s" %key)
        rBase += char
        lBase += char
        lastChar = char
      elif typ and char in O.__HEX_CHARS:
        if char <= lastChar:
          raise Exception("Invalid basis: %s" %key)
        rBase += char
        lBase += char
        lastChar = char
      elif char in O.__BASIS_CHARS:
        rBases[offset +1] += rBase
        rBase = ""
        typ = char
      else:
        raise Exception("Invalid basis: %s" %key)
    if not rBase:
      raise Exception("Invalid basis: %s" %key)
    if lastChar > O.__maxBasis[offset]:
      O.__maxBasis[offset] = lastChar
    xyz = O._basisArgs(lastChar)[0]
    rBases[0] = xyz.index(lBase)
    rBases[offset +1] += rBase
    return O.Grade(value, rBases)

  def __str__(self):
    """Overload string output. Printing taking resolution into account."""
    out = ""
    sign = ""
    for base in [None] +self.__g:
      if base:
        val = base.value
        oOut,uOut = base.strs()
      else:
        val = self.w
        oOut = uOut = ""
      out += Common._resolutionDump(sign, val, oOut +uOut)
      if out:
        sign = " +"
    return out if out else "0"
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if Common._isVerbose() and O.dumpRepr:
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __hash__(self):
    """Allow dictionary access for basis objects."""
    return hash(str(self))

  def __eq__(self, cf):
    """Return True if 2 Os are equal within precision."""
    precision = Common._getPrecision()
    if isinstance(cf, (int, float)):
      return not self.__g  and abs(self.w -cf) < precision
    elif not isinstance(cf, O):
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

  def __add__(self, q):
    """Add 2 Os or a scalar from w."""
    if isinstance(q, O):
      out = self.copy(self.w +q.w)
      for grade in q.__g:
        out.__add(grade)
    elif isinstance(q, Tensor):
      out = q.__add__(self)
    else:
      Common._checkType(q, (int, float), "add")
      out = self.copy(self.w +q)
    return out
  __radd__ = __add__

  def __sub__(self, q):
    """Subtract 2 Os or a scalar from w."""
    if isinstance(q, O):
      lhs = self.__copy()
      rhs = q.__copy()
      for key,val in rhs.items():
        if key in lhs:
          lhs[key] -= val
        else:
          lhs[key] = -val
      return self.copy(self.w -q.w, **lhs)
    if isinstance(q, Tensor):
      return q.__add__(-self)
    Common._checkType(q, (int, float), "sub")
    out = self.copy(self.w -q)
    return out
  def __rsub__(self, q):
    """Subtract Q from scalar with Q output."""
    return self.__neg__().__add__(q)

  def __neg__(self):
    """Unitary - operator for O."""
    out = self.copy(-self.w)
    for grade in out.__g:
      grade.value = -grade.value
    return out
  def __pos__(self):
    """Unitary + operator for O."""
    return self
  def __abs__(self):
    """Unitary abs operator for O."""
    out = self.copy(abs(self.w))
    for grade in out.__g:
      grade.value = abs(grade.value)
    return out
  abs = __abs__

  def __mul__(self, q):
    """Multiplication of 2 Os or self by scalar."""
    if isinstance(q, O):
      out = O(self.w *q.w)
      if self.w:
        for grade2 in q.__g:
          grade = self.Grade(self.w, (0, "", ""))
          out.__add(grade.mergeBasis(grade2.value, grade2.bases()))
      if q.w:
        for grade1 in self.__g:
          out.__add(grade1.mergeBasis(q.w, (0, "", "")))
      for grade1 in self.__g:
        for grade2 in q.__g:
          out.__add(grade1.mergeBasis(grade2.value, grade2.bases()))
    elif isinstance(q, Tensor):
      out = q.__rmul__(self)
    else:
      Common._checkType(q, (int, float), "mul")
      out = O(self.w *q)
      for grade in self.__g:
        out.__g.append(self.Grade(grade.value *q, grade.bases()))
    return out
  __rmul__ = __mul__

  def __bool__(self):
    return self != 0
  __nonzero__ = __bool__

  def __div__(self, q):
    """Attempted division for 2 versors or self by scalar."""
    if isinstance(q, O):
      return self.__mul__(q.inverse())
    Common._checkType(q, (int, float), "div")
    if abs(q) < Common._getPrecision():
      raise Exception("Illegal divide by zero")
    if sys.version_info.major == 2 and isinstance(q, int): # Python v2 to v3
      q = float(q)
    out = O(self.w /q)
    for grade in self.__g:
      out.__g.append(self.Grade(grade.value /q, grade.bases()))
    return out 
  __truediv__ = __div__
  __floordiv__ = __div__

  def __rdiv__(self, q):
    """Division for number, q, divided by an O."""
    return self.inverse().__mul__(q)
  __rtruediv__ = __rdiv__
  __rfloordiv__ = __rdiv__

  def __cf(self, cf, oper):
    """Return inside/outside graded comparisons for operator."""
    if isinstance(cf, (int, float)):
      return oper(self.w, cf)
    elif not isinstance(cf, O):
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
        if not oper(cfBase.value, 0.0):
          return False
        cfIdx += 1
      else:
        if not oper(bas.value, cfBase.value):
          return False
        idx += 1
        cfIdx += 1
    return True

  def __add(self, grade):
    """Add a single O term to self placing it in the correct order."""
    if sum(grade.lens()) == 0:
      self.w += grade.value
      return
    pos = 0
    for idx,base in enumerate(self.__g[:]):
      order = base.order(grade)
      if order == 0:
        self.__g[pos].value += grade.value
        if not self.__g[idx].value:
          del(self.__g[idx])
        return
      elif order > 0:
        break
      pos = idx +1
    if grade.value:
      self.__g.insert(pos, grade)

  def __copy(self):
    """Used by copy() to turn the basis into a kwargs dictionary."""
    v = {}
    for base in self.__g:
      oOut,uOut = base.strs()
      v["%s%s" %(oOut,uOut)] = base.value
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
      cnt += 1
      dim = int(sum(grade.lens()) %2 == 0)
      if even == -1:
        even = dim
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

  def __vectorSizes(self):
    """Return the Octonian vector sizes. Can't handle negative signatures."""
    dims = self.basis()
    for idx,val1 in enumerate(dims):
      val2 = O.__maxBasis[idx]
      val2 = int(val2, O.__HEX_BASIS +1)
      dims[idx] = max(val1, val2)
    return (dims[0], 0)

  @staticmethod
  def _basisArgs(dim=0, offset=None):
    """Used by Grade, BasisArgs and other calcs and matches Grade.order.
       Returns basis digits list for max dim >= dim and current max dim.
       If offset then return incremental digits for dim part with offset."""
    if isinstance(dim, Common._basestr):
      dim = int(dim, O.__HEX_BASIS +1)
    if dim > O.__basisDim:
      out = [""]
      for ii in range(1, dim +1):
        form = "%X" %ii
        for val in out[:]:
          out.append(val +form)
      O.__basisXyz = out
      O.__basisDim = dim
    else:
      out = O.__basisXyz
    if offset is None:
      return out, O.__basisDim, O.__wikiMulRule
    if dim <= 0:
      return []
    pos0 = int(pow(2, dim -1))
    pos1 = int(pow(2, dim))
    out = [""]
    for ii in range(1, dim +1):
      form = "%X" %(ii +offset)
      for val in out[:]:
        out.append(val +form)
    return out[pos0:pos1]

  @staticmethod
  def _basisCache(col, row=None):
    """Used by Grade to store the multiplication table row by row."""
    if row:
      lenCol = len(O.__basisCache)
      if col >= lenCol or len(row) >= lenCol:
        O.__basisCache.extend([None] *(len(O.__basisXyz) -lenCol))
      O.__basisCache[col] = row
    if col >= len(O.__basisCache):
      return None
    return O.__basisCache[col]

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
    return (flat == 0 or cnt == 1) and \
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
    out = O(*args, **kw)
    return out

  def trim(self, precision=None):
    """trim([precision])
       Remove elements smaller than precision."""
    if precision is None:
      precision = Common._getPrecision()
    else:
      Common._checkType(precision, float, "trim")
    out = O(0 if abs(self.w) < precision else self.w)
    for grade in self.__g:
      if abs(grade.value) >= precision:
        out.__g.append(self.Grade(grade.value, grade.bases()))
    return out

  def pure(self):
    """pure()
       Return the pure imaginary part of self."""
    out = O()
    for grade in self.__g:
      out.__g.append(self.Grade(grade.value, grade.bases()))
    return out

  def vector(self):
    """vector()
       Return the coefficients as a Matrix."""
    dims = self.__vectorSizes()
    xyz = O.BasisArgs(*dims)
    v = [0] *len(xyz)
    for grade in self.__g:
      for base in grade.strs():
        if len(base) > 0:
          pos = xyz.index(base)
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
    b = '0'
    for grade in self.__g:
      bases = grade.strs()
      l = (len(bases[0]) -1) if bases[0] else 0
      l += (len(bases[1]) -1) if bases[1] else 0
      if len(g) < l +1:
        g.extend([0] *(l -len(g) +1))
      basis = bases[0]
      if basis and basis[-1] > b:
        b = basis[-1]
      basis = bases[1]
      if basis and basis[-1] > b:
        b = basis[-1]
      b0 = int(b, self.__HEX_BASIS +1)
      if b0 > g[l]:
        g[l] = b0
        b = '0'
    return g

  def basis(self, *maxBasis):
    """basis([maxBasis,...])
       Return the signature or maximum dimension basis of basis elements.
       Optionally set the maxBasis for matrix output. The basis is of form
       ('1','F') or integer and remaining values are set to zero."""
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
      bases = grade.strs()
      dims[0] = max(dims[0], bases[0][-1:])
      dims[1] = max(dims[1], bases[1][-1:])
    dims[0] = int(dims[0], self.__HEX_BASIS +1)
    dims[1] = int(dims[1], self.__HEX_BASIS +1)
    return dims

  def len(self):
    """len()
       Return the sqrt of the sum of the multiplication with the
       complex conjugate."""
    n2 = self.w*self.w
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
    out = self.copy(self.w /l2)
    if split:
      for grade in out.__g:
        grade.value = -grade.value
    else:
      for grade in out.__g:
        sgnVal = grade.copy()
        sgnVal = sgnVal.mergeBasis(grade.value, grade.bases())
        grade.value *= sgnVal.value
    return out

  def inverse(self, noError=False):
    """inverse([noError])
       Return inverse of self which is conj()/len() if len()!=0 and is a versor.
       Raise an error on failure or return 0 if noError."""
    out,flat,n2,cnt = self.__versor(inversed=True)
    l2 = float(n2 +self.w *self.w)
    isInvertable = (flat <= 0 or cnt == 1) and l2 >= Common._getPrecision()
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

  def dot(self, q):
    """dotq)
       Return dot product of pure part and squaring instead of using sym."""
    Common._checkType(q, O, "dot")
    out = idx1 = idx2 = 0
    x = self.__g[0]
    y = q.__g[0]
    while idx1 < len(self.__g) and idx2 < len(q.__g):
      x = self.__g[idx1]
      y = q.__g[idx2]
      order = x.order(y)
      if order < 0:
        idx1 += 1
      elif order > 0:
        idx2 += 1
      else:
        sgn = 1 if x.bases()[1] > 0 else -1
        out += x.value *y.value *sgn 
        idx1 += 1
        idx2 += 1
    return out

  def cross(self, q):
    """cross(q)
       Return cross product of pure part and using asym product."""
    Common._checkType(q, O, "cross")
    x = self.pure()
    y = q.pure()
    out = (x *y -y *x) *0.5 
    return out

  def sym(self, q):
    """sym(q)
       Return symmetric product of two Os. The dot product is for vectors."""
    Common._checkType(q, O, "sym")
    out = (self *q +q *self) *0.5 
    return out

  def asym(self, q):
    """asym(q)
       Return anti-symmetric product of two Os. The wedge product is the
       exterior part of this product."""
    Common._checkType(q, O, "asym")
    return (self *q -q *self) *0.5 
 
  def assoc(self, p, q):
    """assoc(p,q)
       Return the associator [self,p,q] = (self * p) *q - self *(p * q),"""
    out = (self * p) *q - self *(p * q)
    return out

  def moufang(self, p, q, number=0):
    """moufang(p,q,[number])
       Return differences sum of all four Moufang tests for power-associate or
       just one if number is set (0=all)."""
    Common._checkType(p, (O), "moufang")
    Common._checkType(q, (O), "moufang")
    Common._checkType(number, (int), "moufang")
    if number == 1:   out = q*(self *(q*p)) -((q*self) *q) *p
    elif number == 2: out = self *(q* (p*q)) -((self*q) *p) *q
    elif number == 3: out = (q*self) *(p*q) -(q *(self*p)) *q
    elif number == 4: out = (q*self) *(p*q) -q *((self*p) *q)
    elif number != 0:
      raise Exception("Invalid vaue for number in moufang")
    out = q*(self *(q*p)) -((q*self) *q) *p \
        + self *(q* (p*q)) -((self*q) *p) *q \
        + (q*self) *(p*q) -(q *(self*p)) *q \
        + (q*self) *(p*q) -q *((self*p) *q)
    return out

  def projects(self, q):
    """projects(q)
       Return (parallel, perpendicular) parts of vector q projected onto self
       interpreted as a plane a*b with parts (in,outside) the plane. If q is
       a*b, a != b then return parts (perpendicular, parallel) to plane of a &
       b. a.cross(b) is not needed as scalar part is ignored."""
    Common._checkType(q, O, "projects")
    n1 = self.vectorLen()
    if n1 < Common._getPrecision():
      raise Exception("Invalid length for projects")
    mul = self.pure()
    vect = q.pure()
    mul = mul *vect *mul /float(n1 *n1)
    return (vect +mul)/2.0, (vect -mul)/2.0
  
  def rotate(self, q):
    """rotate(q)
       Rotate q by self. See rotation."""
    Common._checkType(q, O, "rotate")
    precision = Common._getPrecision()
    inv,flat,n2,cnt = self.__versor(inversed=True, both=True)
    l2 = float(n2 + self.w *self.w)
    if (flat != 0 and cnt != 1) or l2 < precision:
      raise Exception("Illegal versor for rotate()")
    if n2 < precision:
      return q
    if abs(l2 -1.0) < precision:
      l2 = 1.0
    return inv *q *self /l2

  def rotation(self, rot):
    """rotation(rot)
       Rotate self inplace by rot, if necessary. Applying to versors rotates
       in the same sense as quaternions and frame. For O vectors this is the
       same as rot.inverse()*self*rot."""
    Common._checkType(rot, O, "rotation")
    precision = Common._getPrecision()
    inv,flat,n2,cnt = rot.__versor(inversed=True, both=True)
    l2 = float(n2 + rot.w *rot.w)
    if (flat != 0 and cnt != 1) or l2 < precision:
      raise Exception("Illegal versor for rotation()")
    if n2 < precision:
      return self.copy()
    if abs(l2 -1.0) < precision:
      l2 = 1.0
    newSelf = inv *self *rot /l2
    self.w = newSelf.w
    self.__g = newSelf.__g

  def frame(self):
    """frame()
       Return self=w+v as a frame=acos(w)*2 +v*len(w+v)/asin(w) taking vector
       part v. Ready for frameMatrix. Also handles hyperbolic versor. See
       versor."""
    precision = Common._getPrecision()
    tmp,flat,n2,cnt = self.__versor()
    l2 = n2 +self.w *self.w
    if (flat != 0 and cnt != 1) or abs(math.sqrt(l2) -1.0) >= precision:
      if not noError:
        raise Exception("Illegal versor for frame()")
    if n2 < precision:
      return O(1)
    if flat > 0:
      w = abs(self.w)
      if w < 1.0:
        raise Exception("Invalid hyperbolic frame angle")
      out = self.copy(math.acosh(w))
      if self.w < 0:
        out.w *= -1
    else:
      w = (self.w +1.0) %2.0 -1.0
      out =self.copy( math.acos(w) *2)
    n1 = math.sqrt(n2)
    if n1 >= precision:
      n0 = 1/n1
      for base in out.__g:
        base.value *= n0
    return out

  def versor(self, noError=False):
    """versor()
       Return a versor of length 1 assuming w is the angle(rad) ready for
       rotation. Opposite of frame. See norm. Handles both signatures."""
    precision = Common._getPrecision()
    tmp,flat,n2,cnt = self.__versor()
    w1 = self.w
    if cnt < 0:
      w1 = self.__g[-1].value
      n2 -= w1 *w1
    l2 = n2 +w1 *w1
    if (flat > 0 and cnt != 1) or l2 < precision:
      if not noError:
        raise Exception("Illegal versor for versor()")
    if n2 < precision:
      return O(1)
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
       Return vector & scalar part with the vector as length one."""
    precision = Common._getPrecision()
    out = self.copy(self.w)
    n2 = 0
    for base in out.__g:
      n2 += base.value *base.value
    if n2 >= precision:
      n1 = math.sqrt(n2)
      for base in out.__g:
        base.value /= n1
    return out

  def distance(self, q):
    """distance(qa)
       Return the Geodesic norm which is the half angle subtended by
       the great arc of the S3 sphere d(p,q)=|log(p.inverse())*q|."""
    Common._checkType(q, O, "distance")
    return (self.inverse() *q).log().len()

  def norm(self):
    """norm()
       Normalise - reduces error accumulation. Versors have norm 1."""
    precision = Common._getPrecision()
    n = self.len()
    if n < precision:
      return O(1.0)
    out = self.copy(self.w /n)
    for base in out.__g:
      base.value /= n
    return out

  def pow(self, exp):
    """pow(exp)
       For even q=w+v then a=|q|cos(a) & v=n|q|sin(a), n unit."""
    Common._checkType(exp, (int, float), "pow")
    if isinstance(exp, int):
      out = O(1.0)
      for cnt in range(exp):
        out *= self
      return out
    tmp,flat,n2,cnt = self.__versor()
    if flat <= 0:
      l1 = math.sqrt(n2 +self.w *self.w)
      w = pow(l1, exp)
      if l1 < Common._getPrecision():
        return O(w)
      a = math.acos(self.w /l1)
      s,c = Common._sincos(a *exp)
      s *= w /math.sqrt(n2)
      out = O(w *c)
      for grade in self.__g:
        oStr,uStr = grade.strs()
        out += O(**{oStr +uStr: grade.value *s})
      return out
    raise Exception("Invalid float exponent for non-hyperbolic, non-versor pow")
  __pow__ = pow

  def exp(self):
    """exp()
       For even q=w+v then exp(q)=exp(w)exp(v), exp(v)=cos|v|+v/|v| sin|v|."""
    tmp,flat,n2,cnt = self.__versor()
    if n2 < Common._getPrecision():
      return O(self.w)
    if flat <= 0:
      n1 = math.sqrt(n2)
      s,c = Common._sincos(n1)
      exp = pow(math.e, self.w)
      s *= exp /n1
      out = O(exp *c)
      for grade in self.__g:
        oStr,uStr = grade.strs()
        out += O(**{oStr +uStr: grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for exp")

  def log(self):
    """log()
       The functional inverse of the quaternion exp()."""
    tmp,flat,n2,cnt = self.__versor()
    l1 = math.sqrt(self.w *self.w +n2)
    if n2 < Common._getPrecision():
      return O(math.log(l1))
    if flat <= 0:
      s = math.acos(self.w /l1) /math.sqrt(n2)
      out = O(math.log(l1))
      for grade in self.__g:
        oStr,uStr = grade.strs()
        out += O(**{oStr +uStr: grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for log")

  def euler(self, noError=False):
    """euler([noError])
       Quaternion versors can be converted to Euler Angles & back uniquely for
       normal basis order. Error occurs for n-D greater than 3 or positive
       signature.  Euler parameters are of the form cos(W/2) +n sin(W/2),
       n pure unit versor. Set noError to return a zero Euler if self is not
       valid to be a versor or norm if possible."""
    if self.basis()[0] > 2 or self.basis()[1] > 0:
      raise Exception("Illegal versor signature or size for euler")
    tmp,flat,n2,cnt = self.__versor()
    l2 = n2 +self.w *self.w
    if not (flat == 0 or cnt == 1):
      raise Exception("Illegal versor for euler")
    if abs(l2 -1.0) >= Common._getPrecision():
      if not noError:
        raise Exception("Illegal versor norm for euler")
      tmp = tmp.norm()
    if n2 < Common._getPrecision():
      return Euler()
    args = [0] *3
    xyz = O.BasisArgs(2)
    angles = [0] *len(xyz)
    for grade in tmp.__g:
      idx = grade.bases()[0] -1
      if idx < 3:
        args[idx] = grade.value
    w, x, y, z = tmp.w, args[0], args[1], args[2]
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
       This is same as frameMatrix but for a versor ie half the angle.
       Converts self to euler than to matrix assuming normal basis order.
       Only defined for quaternion part. Set noError to norm the versor
       if not normed."""
    out = self.euler(noError)
    dim = int(Common.comb(len(out), 2))
    dims = self.__vectorSizes()
    dim = int(pow(2, sum(dims)) -1)
    out.extend([0] *(dim - len(out)))
    return Euler(*out).matrix()

  def frameMatrix(self):
    """frameMatrix()
       Rodriges for 3-D. See https://math.stackexchange.com/questions/1288207/
       extrinsic-and-intrinsic-euler-angles-to-rotation-matrix-and-back.
       Converts self to versor then to euler then matrix."""
    out = self.versor().euler()
    dim = int(Common.comb(len(out), 2))
    dims = self.__vectorSizes()
    dim = int(pow(2, sum(dims)) -1)
    out.extend([0] *(dim - len(out)))
    return Euler(*out).matrix()

  def morph(self, pairs):
    """morph(pairs)
       Morphism with a list of pairs of names with o1,o2 meaning map o1->o2."""
    out = O(self.w)
    for grade in self.__g:
      out += O(**Common._morph(grade.strs(), grade.value, pairs))
    return out

  ##############################################################################
  ## Other creators and source inverters
  ##############################################################################
  @staticmethod
  def CayleyDicksonRule(wiki=True):
    """cayleyDicksonRule([wiki])
       Change Cayley-Dickson multiplication from Baez to Wikipedia (or back if
       wiki is False). (a,b)*(c,d) = (a*c-d.conj*b, d*a +b*c.conj) [Wikipedia] or 
       (a,b)*(c,d) = (a*c-d*b.conj, a.conj*d +c*b) [J.C.Baez]."""
    Common._checkType(wiki, bool, "cayleyDicksonRule")
    O.__basisCache  = []
    O.__wikiMulRule = wiki

  @staticmethod
  def Basis(oDim, uDim=0):
    """Basis(pDim, [nDim])
       Return (o,u) basis elements with value one."""
    Common._checkType(oDim, int, "Basis")
    Common._checkType(uDim, int, "Basis")
    return tuple((O(**{x: 1}) for x in O.BasisArgs(oDim, uDim)))

  @staticmethod
  def BasisArgs(oDim, uDim=0):
    """BasisArgs(oDim, [uDim])
       Return (o,u) basis elements as a list of names in addition order.""" # XXX TBD
    Common._checkType(oDim, int, "BasisArgs")
    Common._checkType(uDim, int, "BasisArgs")
    if oDim +uDim > O.__HEX_BASIS:
      raise Exception("Too many basis arguments")
    if oDim < 0 or uDim < 0:
      raise Exception("Too few basis arguments")
    out = []
    for n in range(1, oDim +uDim +1):
      for i in range(n +1):
        if oDim >= n-i and uDim >= i:
          outo = tuple(map(lambda x: "o" +x, O._basisArgs(n-i, 0)))
          outu = tuple(map(lambda x: "u" +x, O._basisArgs(i, oDim)))
          out.extend(Common._mergeBasis(outo, outu))
    return out

  @staticmethod
  def VersorArgs(oDim, uDim=0, rotate=False):
    """VersorArgs(oDim, [uDim])
       Same as BasisArgs except o12 is negated."""
    Common._checkType(oDim, int, "VersorArgs")
    Common._checkType(uDim, int, "VersorArgs")
    Common._checkType(rotate, bool, "VersorArgs")
    out = O.BasisArgs(oDim, uDim)
    if rotate and oDim > 1:
      out[2] = "-" +out[2]
    return out

  @staticmethod
  def Versor(*args, **kwargs):
    """Versor([scalar, o1 multiplier, ...][basis=multiplier, ...])
       Return versor(2-D +...) where ... is higher dimensions in the
       form o1=x, o2=y, o12=z, .... Each dimension has (D 2)T=D(D-1)/2
       parameters and these are added as o3, o13, etc.
       Use BasisArgs() to see this list. See Euler() for an angle version
       instead of parameters being n sin(W/2), n unit."""
    # See Wikipedia.org rotations in 4-dimensional Euclidean space
    if len(kwargs) > Common.comb(O.__HEX_BASIS, 2):
      raise Exception("Invalid number of Versor euler bivectors")
    if args:
      dim = int((math.sqrt(8*(len(args)-1) +1) +1) /2 +0.9) # l=comb(dim,2)
      if dim > O.__HEX_BASIS +1:
        raise Exception("Too many basis elements")
      xyz = O.BasisArgs(dim)
      for idx,val in enumerate(args[1:]):
        if xyz[idx] in kwargs:
          raise Exception("Invalid Versor basis duplication: %s" %xyz[idx])
        kwargs[xyz[idx]] = val
      args = args[:1]
    q = O(*args, **kwargs)
    return q.versor()

  @staticmethod
  def Euler(*args, **kwargs): #order=[], implicit=False):
    """Euler([angles, ...][o1=multiplier, ...][order, implicit])
       Euler angles in higher dimensions have (D 2)T=D(D-1)/2 parameters.
       SO(4) has 6 and can be represented by two Oernions. Here they are
       changed to a versor using explicit rotation & this is returned.
       So in 3-D q' = (cx+sx e23) *(cy+sy e13) *(cz+sz e12).
       kwargs may contains "order" and "implicit".  The args arguments
       are entered as radian angles and rotations applied in the given order
       as shown using BasisArgs(). This order can be changed using the order
       array which must be as long as the list of angles. The default is 
       [1,2,...] and must have unique numbers. If implicit is set True then
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
    out = O(1.0)
    implicitRot = O(1.0)
    store = []
    dim = int((math.sqrt(8*len(args) +1) +1) /2 +0.9) # l=comb(dim,2)
    xyz = O.BasisArgs(dim)
    for bi,val in kwargs.items():
      if bi not in ("order", "implicit"):
        while bi not in xyz:
          dim += 1
          if dim > O.__HEX_BASIS:
            raise Exception("Invalid Euler parameter: %s" %bi)
          xyz = CA.BasisArgs(dim)
        if bi in xyz:
          idx = xyz.index(bi) 
        args.extend([0] *(idx -len(args) +1))
        args[idx] = val 
    if not order:
      order = range(1, len(args) +1)
    elif len(order) < len(args):
      raise Exception("Invalid order size")
    for idx,key in enumerate(order):
      Common._checkType(key, (float, int), "Euler")
      key = int(key)
      if key in store or key > len(args):
        raise Exception("Invalid order index for rotation: %s" %key)
      ang = args[key -1]
      Common._checkType(ang, (int, float), "Euler")
      s,c = Common._sincos(ang *0.5)
      rot = O(c, **{xyz[key -1]: s})
      if implicit:
        tmpRot = rot.copy()
        rot.rotation(implicitRot)
        implicitRot = tmpRot * implicitRot
      else:
        store.append(key)
      out *= rot
    return out

  @staticmethod
  def Q(*args):
    """Q([scalar, x, y, z])
       Map quaternion basis (w,i,j,k) to (w, o1, o2, -o12) with up to 4
       arguments. If calc(Q) included then w may instead be a Q object."""
    if O.__useQuat:   # If module calcQ included can use Euler class
      if len(args) == 1 and isinstance(args[0], Q):
        q = args[0]
        args = []
        for val in (q.w, q.x, q.y, q.z):
          args.append(val)
    xyz = O.VersorArgs(2, rotate=True)
    if len(args) > 4:
      raise Exception("Invalid O parameters")
    kwargs = {}
    for idx,val in enumerate(args[1:]):
      kwargs[xyz[idx]] = val if idx < 3 else -val
    return O(args[0], **kwargs)

  @staticmethod
  def IsCalc(calc):
    """Check if calcQ or calcCA has been loaded."""
    if calc == "O": return True
    if calc == "CA" and O.__useCA:
      return True
    return (calc == "Q" and O.__useQuat)

  ###################################################
  ## Calc class help and basis processing methods  ##
  ###################################################
  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for O."""
    calcHelp = """Octonian/Sedonian Calculator - Process 30-dimensional basis
          numbers (o1..F or u1..F) and multiples."""
    return (("O", "CA", "Q", "R"), ("O", "math"), "default.oct", calcHelp, "")

  @classmethod
  def _setCalcBasis(cls, calcs, dummy):
    """Load other calculator. If quaternions are loaded then convert
       i,j,k into Q() instead of o1,o2,-o12. Default True."""
    loaded = ""
    if "CA" in calcs:
      for i in cls.__CA_CHARS:
        if i not in cls.__allChars:
          cls.__allChars.append(i)
      cls.__useCA = True
    if cls.__useCA:
      loaded = "CA"
    if "Q" in calcs:
      for i in cls.__QUAT_CHARS:
        if i not in cls.__allChars:
          cls.__allChars.append(i)
      cls.__useQuat = True
    if cls.__useQuat:
      loaded += " and " if cls.__useCA else ""
      loaded += "Quaternions"
    if loaded:
      return "Octonion calculator has %s enabled" %loaded
    return ""

  @classmethod
  def _validBasis(cls, value, full=False):
    """Used by Calc to recognise full basis forms o... and u... or i, j, k
       or e... and i... if CA is loaded."""
    if len(value) == 1:
      return 1 if value in cls.__QUAT_CHARS else 0
    if value[0] not in cls.__allChars:
      return 0
    isBasis = True
    isCA = value[0] in cls.__CA_CHARS
    for ch in value:
      if isBasis and ch in cls.__allChars:
        if isCA != (ch in cls.__CA_CHARS) or ch in cls.__QUAT_CHARS[1:]:
          return 0
        isBasis = False
      elif ch.isdigit() or ch in cls.__HEX_CHARS:
        isBasis = True
      else:
        return 0
    return 2 if isCA else 3

  @classmethod
  def _processStore(cls, state):
    """Convert the store array into O(...) or Q(...) python code. Convert to
       Q(...) if basis is i, j, k and __useQuat or CA(...) for e/i basis and
       __useCA else expand to bivectors. If isMults1/2 set then double up
       since O to Q or MULTS are higher priority then SIGNS. The state is a
       ParseState from Calculator.processTokens()."""
    kw = {}
    line = ""
    isQuat = isCA = False
    quatKeys = (None, 'i', 'j', 'k')  # based on __QUAT_KEYS
    signTyp = ""
    firstCnt = 1 if state.isMults1 else -1
    lastCnt = len(state.store) -1 if state.isMults2 else -1
    for cnt,value in enumerate(state.store):
      val,key = value
      if key:
        if key in cls.__QUAT_CHARS:
          isQuat = O.__useQuat     # The same validBasis type
          if not isQuat:
            xyz = cls.BasisArgs(2)
            if key == 'k':
              val = val[1:] if val[:1] == "-" else "-" +val
            key = xyz[cls.__QUAT_CHARS.index(key)]
        elif key[0] in cls.__CA_CHARS:
          isCA = O.__useCA

      # If basis already entered or single so double up
      isMult = (cnt in (firstCnt, lastCnt) and lastCnt != 0)
      if key in kw or isMult:
        if key is None and not isMult:  # Duplicate scalar
          val = "(%s%s)" %(kw[None], val)
        elif isMult and len(kw) == 1 and None in kw:
          line += kw[None]
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
            scalar = "%s, " %kw[None]
            del(kw[None])
          if isCA:
            line += signTyp +"CA(%s%s)" %(scalar, ",".join(map(lambda x: \
                    ("%s=%s" %(x,kw[x])) if x else str(kw[x]), kw.keys())))
          else:
            line += signTyp +"O(%s%s)" %(scalar, ",".join(map(lambda x: \
                    ("%s=%s" %(x,kw[x])) if x else str(kw[x]), kw.keys())))
          signTyp = "+"
          kw = {}
        isQuat = False
        isCA = False
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
      if isCA:
        line += signTyp +"CA(%s%s)" %(scalar, ",".join(map(lambda x: \
                ("%s=%s" %(x,kw[x])) if x else str(kw[x]), kw.keys())))
      else:
        line += signTyp +"O(%s%s)" %(scalar, ",".join(map(lambda x: \
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

  # O Unit test cases for Calc with Tests[0] being init for each case
  Tests = [\
    """d30=radians(30); d60=radians(60); d45=radians(45); d90=radians(90)
       e=Euler(pi/6,pi/4,pi/2); c=o1+2o2+3o12; c.basis(2)""",
    """# Test 1 Rotate via frameMatrix == versor half angle rotation.
       Rx=d60+o1; rx=(d60 +o1).versor()
       test = Rx.frameMatrix() *c.vector(); store = (rx.inverse()*c*rx).vector()
       Calculator.log(store == test, store)""",
    """# Test 2 Rotate via frameMatrix == versor.rotate(half angle)].
       Rx=d60+o2; rx=O.Versor(d60,0,1)
       test = Rx.frameMatrix() *c.vector(); store = (rx.rotate(c)).vector()
       Calculator.log(store == test, store)""",
    """# Test 3 Rotate versor rotate == rotation of copy.
       Rx=d60+o12; rx=math.cos(d30) +o12*math.sin(d30)
       test = Rx.frameMatrix() *c.vector(); store = (rx.inverse()*c*rx).vector()
       Calculator.log(store == test, store)""",
    """# Test 4 Quat Euler == O Euler.
       test = O.Euler(pi/6,pi/4,pi/2)
       store = O.Euler(e,order=[1,2,3],implicit=False)
       Calculator.log( store == test, store)""",
    """# Test 5 Euler implicit rotation == other order, Rzyx==Rxy'z''.
       if O.IsCalc("Q"):
         test = O.Q(Q.Euler(e, order=[1,2,3], implicit=True))
       else:
         test = O.Euler(e, order=[3,2,1])
         Common.precision(1E-12)
       store = O.Euler(e, order=[1,2,3], implicit=True)
       Calculator.log(store == test, store)""",
    """# Test 6 Versor squared == exp(2*log(e)).
       test = O.Euler(e).pow(2); store = (O.Euler(e).log() *2).exp()
       Calculator.log(store == test, store)""",
    """# Test 7 Rotate via frameMatrix == versor.versorMatrix(half angle).
       if O.IsCalc("Q"):
         test = (d45+i+j+k).frameMatrix()
       else:
         test = (d45+i+j+k).frameMatrix()
       store = (d45+i+j+k).versor().versorMatrix()
       Calculator.log(store == test, store)""",
    """# Test 8 Rotate via versor.versorMatrix() == versor.euler().matrix().
       r = d45 +i +j +k; store = r.norm().euler().matrix()
       if O.IsCalc("Q"):
         test = r.norm().versorMatrix()
       else:
         test = r.norm().versorMatrix().reshape((3,3))
       Calculator.log(store == test, store)""",
    """# Test 9 Euler Matrix is inverse of versorMatrix.
       test=Matrix(pi/6, pi/4, pi/2)
       store=Euler.Matrix(O.Euler(*test).versorMatrix())
       Calculator.log(store == Euler(*test), store)""",
    """# Test 10 Geodetic distance = acos(p.w *d.w -p.dot(d)).
       if O.IsCalc("Q"):
         p = Q.Euler(e); d=(d45+i+2j+3k).versor()
         test = math.acos(p.w *d.w -p.dot(d))
         p = O.Q(p); d = O.Q(d)
       else:
         p = O.Euler(e); d=(d45+c).versor()
         test = math.acos(p.w *d.w -p.pure().sym(d.pure()).scalar())
       store = p.distance(d)
       Calculator.log(abs(store - test) < 3E-5, store)""",
    """# Test 11 Length *2 == dot(self +self).
       store = (c *2).len()
       test = math.sqrt(-(c +c).pure().sym((c +c).pure()).scalar())
       Calculator.log(abs(store - test) <1E-15, store)""",
    """# Test 12 Versor *3 /3 == versor.norm
       Calculator.log(c/c.len() == c.norm(), c.norm())""",
    """# Test 13 Check Rodriges formula
       def para(a,r,w): return -a *a.sym(r)
       def perp(a,r,w): return r *math.cos(w) -a.asym(r) \\
               *math.sin(w) +a *a.sym(r) *math.cos(w)
       store = para(o1,o1+o2,d30)+perp(o1,o1+o2,d30)
       if O.IsCalc("Q"):
         test = O.Q((d30+i).versor().rotate(i+j))
       else:
         test = (d30+o1).versor().rotate(o1+o2)
       Calculator.log(store == test, store)""",
    """# Test 14 Compare Tensor projection and O.projects.
       def Ptest(a, b, x):
         G,P,N = Tensor.Rotations(a.unit().vector(), b.unit().vector())
         p = (a * b).projects(x); x0 = P *x.vector()
         return [p[0].vector(), p[1].vector()] == [x0, x.vector()-x0]
       d2 = Ptest(O(0,1), O(0,0,1), O(0,1,2))
       d3 = Ptest(O(0,1,0,0), O(0,0,1,2), O(0,1,2,3))
       Calculator.log(d2 and d3, (d2, d3))""",
    """# Test 15 Euler Matrix is inverse of versorMatrix.
       if O.IsCalc("Q"):
         test = Q.Euler(pi/6, pi/4, pi/2).versorMatrix()
       else:
         test = O.Euler(pi/6, pi/4, pi/2).versorMatrix()
       store = Euler(pi/6, pi/4, pi/2).matrix()
       Calculator.log(store == test, store)""",
       ]

  calc = Calculator(O, Tests)
  calc.processInput(sys.argv)
###############################################################################
