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
## Octonions, Sedenions, Quaternions, Versors and Euler Angle rotations.
## CA numbers and Quaternions can also be included and processed separately
## which allows Dixon Algebra to be defined using the Tensor class. 
##
## O multivectors are of the form p = w +p o1..F u1..F +....
## A quaternion maps as i->o1, j->o2, k->-o12. u basis has the same 
## multiplication table as o basis except the squares are positive unity
## and define the split algebra.
## Quaternions are of the form q = w + _v_ where _v_ is a i,j,k vector
## s.t. _v_*_v_ >= 0 and w is a scalar. A pure quaternion has w==0.
## A unit quaternion has _n_*_n_ = -1. A versor has norm = 1 which means
## r = cos(a) + sin(a) * _n_ where _n_ in a unit. This is extended to
## Octonians for basis number 3 and Sedenians for numbers 4 and above.
## The calc(Q) command interprets i, j, k using Q class and the calc(CA)
## command interprets CA numbers via CA class. Need to avoid name conflicts
## in loaded files in the later case.
## Start with either calcO.py, python calcO.py or see ./calcR.py -h.
################################################################################
__version__ = "0.2"
import math
from calcCommon import *

################################################################################
class O():
  """Class to process octonions and unity elements consisting of 32-D numbers,
     half being positive definite (u1..uF) and half being standard imaginary
     octonions (o1..oF). O(q,p) specifies q the octonian dimension (o<hex>) and
     p specifies the dimension of octonions with unity base elements (u1..uF).

     The notation is the same as for CA so o1*o2=-o12 but the basis is ungraded
     so o12 is a single imaginary entity and o21 or o1o2 becomes o12. u1o2u3o4
     becomes o24u13. This allows easy extension to arbitray Sedenian Algebras
     (O(q) with q>4) and split octonian and sedenions. The o and u basis hex
     numbers can be entered in any order but must be unique.

     O(1) is complex numbers, O(2) is quaternions, O(3) is Octonions. Just enter
     arbitrary elements and the highest basis dimensions are remembered for
     producing matricies. O(1,1) are Lorentz numbers and O(2,1), O(1,2) and 
     O(0,3) are split octonions. The O.Basis(p,q) function creates the basis 
     with unity basis elements last. These can be created manually and u1, u2,
     u3, u12, u23, u13, u123 provide the unity basis for all seven
     representations of the split octonion algebras. The multiplication tables
     for these can be displayed using Tensor.Table(O.Basis(p,q)) and mapping
     found with the seach method. Note that package math methods only work on
     the scalar part eg sin(a)==sin(a.scalar()).
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
  __baezMulRule = False                  # Change Cayley-Dickson multiply rule
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
      """Product n => (a,b)(c,d) = (ac-d*b, da+bc*) for conjugate level n-1.
         This is wikiRule - for baezRule see _mulBaez."""
      n = self[0]
      if not isinstance(p, O.Product) or n != p[0]:
        raise Exception("Invalid Product to multiply: %s * %s" %(self, p))
      if n == 0:
        return self[1] * p[1]
      if n == 1:
      # (p,q)*(r,s) = (pr -s*q, sp +qr*) [wikiRule]
        return O.Product(-1, self[1] *p[1] -p[2] *self[2],
                             p[2] *self[1] +self[2] *p[1])
      return O.Product(-1, self[1] *p[1] -p[2].__conj(n-1) *self[2],
                           p[2] *self[1] +self[2] *p[1].__conj(n-1))
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
    def _mulBaez(self, p):
      """Product n => (a,b)(c,d) = (ac-db*, a*d+cb) for conjugate level n-1.
         This is baezRule - for wikiRule see __mul__."""
      n = self[0]
      if not isinstance(p, O.Product) or n != p[0]:
        raise Exception("Invalid Product to multiply: %s * %s" %(self, p))
      if n == 0:
        return self[1]._mulBaez(p[1])
      # (p,q)*(r,s) = (pr -sq*, p*s +rq) [baezRule]."""
      if n == 1:
        return O.Product(-1, self[1] *p[1] -p[2] *self[2],
                             self[1] *p[2] +p[1] *self[2])
      return O.Product(-1,
                     self[1]._mulBaez(p[1]) -p[2]._mulBaez(self[2].__conj(n-1)),
                     self[1].__conj(n-1)._mulBaez(p[2]) +p[1]._mulBaez(self[2]))
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
    """Each Grade is a list of Product index, o & u basis element parts. Product
       index is taken from _basisArray()."""
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
      cnt = 0
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
              cnt += 1
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
      return (out,cnt)

    def copy(self, value=None):
      return O.Grade(self.value if value is None else value,
                    (self.__pBase, self.__oBase[:], self.__uBase[:]))

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
      xyz = None
      bases = [0, "", ""]     # Base for lhs o and u, resp
      rBase = rhs[0] # Iterate rhs o and u
      lBase = lhs[0]
      hasU = lhs[2] or rhs[2]
      row = O._basisCache(lBase)
      if not row or len(row) <= max(lBase, rBase):
        xyz, maxDim, baezMul = O._basisArray()
        lp = [0] *len(xyz)
        rp = list((x for x in range(len(xyz))))
        lp[lBase] = 1
        walk = [0] *len(xyz)
        row = [0] *len(xyz)
        if baezMul:
          prod = O.Product(maxDim, lp)._mulBaez(O.Product(maxDim, rp))
        else:
          prod = O.Product(maxDim, lp) * O.Product(maxDim, rp)
        prod.walk(walk)
        for idx,val in enumerate(walk):
          if idx > 0:
            row[abs(val)] = idx if val >= 0 else -idx
        O._basisCache(lBase, row)
      if rBase:
        if lBase:
          idx = row[rBase]
          bases[0] = abs(idx)
          if idx <= 0:
            value = -value
          bases[1] = self.__mergeStr(lhs[1], rhs[1])[0]

          if hasU:   # Split multiplication
            bases[2],cnt = self.__mergeStr(lhs[2], rhs[2])
            if cnt %2 == 1:
              value = -value
            for lCh in lhs[2]:
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
      xyz = O._basisArray(dim)[0]   # Setup global dimension
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
    """Return the Grade for basis string key. Separate o & u parts."""
    typ = None
    bases = [0, "", ""]
    base = ""
    pBase = ""
    typ = None
    lastChar = '0'
    for char in key:
      offset = int(typ == O.__BASIS_CHARS[1]) # o==0, u==1
      if char in pBase:
        raise Exception("Invalid basis duplication: %s" %key)
      if typ and char.isdigit():
        base += char
        pBase += char
      elif typ and char in O.__HEX_CHARS:
        base += char
        pBase += char
      elif char in O.__BASIS_CHARS:
        bases[offset +1] += base
        base = ""
        typ = char
      else:
        raise Exception("Invalid basis: %s" %key)
    if not pBase:
      raise Exception("Invalid basis: %s" %key)
    bases[offset +1] += base
    bases[1] = "".join(sorted(bases[1]))
    bases[2] = "".join(sorted(bases[2]))
    pBase = "".join(sorted(pBase))
    lastChar = pBase[-1]
    if lastChar > O.__maxBasis[offset]:
      O.__maxBasis[offset] = lastChar
    xyz = O._basisArray(lastChar)[0]
    bases[0] = xyz.index(pBase)
    return O.Grade(value, bases)

  def __float__(self):
    return float(self.w)
  def __int__(self):
    return trunc(self.w)
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
      out = self.dup(self.w +q.w)
      for grade in q.__g:
        out.__add(grade)
    elif isinstance(q, Tensor):
      out = q.__add__(self)
    else:
      Common._checkType(q, (int, float), "add")
      out = self.dup(self.w +q)
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
    out = self.dup(self.w -q)
    return out
  def __rsub__(self, q):
    """Subtract Q from scalar with Q output."""
    return self.__neg__().__add__(q)

  def __neg__(self):
    """Unitary - operator for O."""
    out = self.dup(-self.w)
    for grade in out.__g:
      grade.value = -grade.value
    return out
  def __pos__(self):
    """Unitary + operator for O."""
    return self
  def __abs__(self):
    """Unitary abs operator for O."""
    out = self.dup(abs(self.w))
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
      if not self.__g:
        return oper(self.w, cf)
      for g in self.__g:
        if oper(g.value, cf):
          return True
      return False
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

  def _invertible0(self):
    """Return (Signed conjugate, invertSgn, sum of basis squares, signed sum of
       basis squares). Actual conjugate negate the pure part while signed one
       doesn't negate the unity basis or mixed parts. For pure self inv=1/-1/0
       if pure unity/mixed count >/</== pure imaginary count. If inv!=0 then
       inverse is inv*self otherwise no inverse. With scalar then inv=-1 is
       invertible via conjugate (changed to inv=-2), as is inv=0 (so this is
       changed to inv=2) but inv=+1 with scalar is not invertible (changed to
       inv=0)."""
    out = None
    out = O(self.w) 
    p2,n2 = 0,0
    p1,n1 = 0,0
    for grade in self.__g:
      value = grade.value
      sgnVal = grade.copy()
      sgnVal = sgnVal.mergeBasis(value, grade.bases())
      if sgnVal.value < 0:
        value = -value
        n1 += 1
        n2 += grade.value *grade.value
      else:
        p1 += 1
        p2 += grade.value *grade.value
      out.__g.append(self.Grade(value, grade.bases()))
    inv = p1-n1
    if self.w:
      if inv < 1:
        inv = 2
      elif inv > 1:
        inv = -2
      else:
        inv = 0
    elif inv < 0:
      inv = -1
    elif inv > 0:
      inv = 1
    return inv,p2-n2

  def __invertible(self):
    """Return (Signed conjugate, invCode, sum of basis squares, signed sum of
       basis squares). Actual conjugate negate the pure part while signed one
       doesn't negate the unity basis or mixed parts. Return invCode is 0 for
       not invertible otherwise inverse is scaled self for +/-1 and conjugate
       if +/-2; + if more positive signature and - for more negative. Pure
       self is self inverse if p1-n1 != 0 and with scalar then conjugate=
       scalar-pure parts is the inverse if p1-n1 != 1."""
    out = None
    out = O(self.w) 
    p2,n2 = 0,0
    p1,n1 = 0,0
    for grade in self.__g:
      value = grade.value
      sgnVal = grade.copy()
      sgnVal = sgnVal.mergeBasis(value, grade.bases())
      if sgnVal.value < 0:
        value = -value
        n1 += 1
        n2 += grade.value *grade.value
      else:
        p1 += 1
        p2 += grade.value *grade.value
      out.__g.append(self.Grade(value, grade.bases()))
    invCode = p1-n1
    if self.w:
      if invCode != 1:
        invCode = 2 if invCode > 1 else -2
      else:
        invCode = 0
    elif invCode != 0:
      invCode = 1 if invCode > 0 else -1
    return out, invCode, p2+n2, p2-n2

  def __vectorSizes(self):
    """Return the Octonian vector sizes. Can't handle negative signatures."""
    dims = self.basis()
    for idx,val1 in enumerate(dims):
      val2 = int(O.__maxBasis[idx], O.__HEX_BASIS +1)
      dims[idx] = max(val1, val2)
    if dims[0] < 2:
      dims[0] = 2
    return dims

  @staticmethod
  def _basisArray(dim=0):
    """Used by Grade, BasisArgs and other calcs and matches Grade.order.
       Returns basis digits list for current max dim = oDim + uDim,
       current max (increasing if dim > max dim) and multiplication rule."""
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
    return out, O.__basisDim, O.__baezMulRule

  @staticmethod
  def _basisCache(idx, row=None):
    """Used by Grade to store the multiplication table row by row."""
    if row:
      lenIdx = len(O.__basisCache)
      if idx >= lenIdx or len(row) >= lenIdx:
        O.__basisCache.extend([None] *(len(O.__basisXyz) -lenIdx))
      O.__basisCache[idx] = row
    if idx >= len(O.__basisCache):
      return None
    return O.__basisCache[idx]

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

  def isVersor(self):
    """isVersor()
       Return true if negative signature inverse."""
    precision = Common._getPrecision()
    tmp,invCode,p2,n2 = self.__invertible()
    return invCode < 0 and \
           abs(math.sqrt(p2 +self.w *self.w) -1.0) < precision

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

  def dup(self, w=None):
    """dup([scalar])
       Fast copy with optional scalar only."""
    out = O()
    if w:
      Common._checkType(w, (int, float), "dup")
      out.w = w
    out.__g = self._copyGrades()
    return out

  def copy(self, *args, **kwargs):
    """copy([scalar, e1 multiplier, ...][basis=multiplier, ...])
       Return clone with optional new basis values."""
    kw = self.__copy()
    kw.update(kwargs)
    if len(args) == 0:
      args = [self.w]
    out = O(*args, **kw)
    return out

  def copyTerms(self):
    """copyTerms()
       Return terms as a list of pairs of (term, factor). Cf O(**dict(...))."""
    v = [("", self.w)] if self.w else []
    for grade in self.__g:
      oStr,uStr = grade.strs()
      v.append(("%s%s" %(oStr, uStr), grade.value))
    return v

  def basisTerms(self):
    """basisTerms()
       Return self as 3 lists = a list of o-basis indicies, values & u-basis."""
    out1,out2,out3 = [],[],[]
    for grade in self.__g:
      pBasis,oBase,uBase = grade.bases()
      basis = []
      for ch in oBase:
        basis.append(int(ch, self.__HEX_BASIS +1))
      out1.append(basis)
      out2.append(grade.value)
      basis = []
      for ch in uBase:
        basis.append(int(ch, self.__HEX_BASIS +1))
      out3.append(basis)
    return out1,out2,out3

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
    return self.dup(0)

  def vector(self):
    """vector()
       Return the coefficients as a 1-D Matrix with psuedo-scalar."""
    dims = self.__vectorSizes()
    vec = [0] *(sum(dims))
    for grade in self.__g:
      bases = grade.bases()
      lens = len(bases[1]) +len(bases[2])
      if lens == 1:
        if bases[1]:
          idx = int(bases[1], O.__HEX_BASIS +1)
        else:
          idx = int(bases[2], O.__HEX_BASIS +1) +dims[0]
        vec[idx -1] = grade.value
      elif lens == sum(dims):
        vec[-1] = grade.value
    return Matrix(*vec)

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

  def norm(self):
    """norm()
       Return the square of the sum of the signed squares."""
    out,invCode,p2,n2 = self.__invertible()
    return p2

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
    out,invCode,p2,n2 = self.__invertible()
    return out

  def inverse(self, noError=False):
    """inverse([noError])
       Return inverse of self which is conj()/len() if len()!=0 and is a versor.
       Raise an error on failure or return 0 if noError."""
    out,invCode,p2,n2 = self.__invertible()
    l2 = float((self.w *self.w -n2) if abs(invCode) == 2 else n2)
    isInvertable = invCode and abs(l2) >= Common._getPrecision()
    if not isInvertable:  # Could be anti-symmetric - check
      if not noError:
        raise Exception("Illegal versor for inverse()")
      return O(0)
    out = self.copy(self.w /l2)
    if abs(invCode) == 2:
      l2 = -l2
    for grade in out.__g:
      grade.value /= l2
    return out

  def cross(self, q):
    """cross(q)
       Return half asym product of pure parts."""
    Common._checkType(q, O, "cross")
    x = O()
    x.__g = self.__g    # Shallow copies
    y = O()
    y.__g = q.__g
    out = (x *y -y *x) *0.5 
    return out

  def sym(self, q):
    """sym(q)
       Return symmetric product of two Os. The pure part is always zero."""
    Common._checkType(q, O, "sym")
    out = (self *q +q *self)
    return out

  def asym(self, q):
    """asym(q)
       Return antisymmetric product of two Os. Cross product is pure part."""
    Common._checkType(q, O, "asym")
    return (self *q -q *self)
 
  def associator(self, p, q, nonAlternate=False, nonPower=False):
    """assoc[iator](p,q, [nonAlternate,nonPower])
       Return the associator [self,p,q] = (self * p) *q - self *(p * q) or
       the first non-alternate or non-power if non-associative where alternate
       if assoc(x,x,y)==0 and power if assoc(x,x,x)==0 for any input x. Of
       course, nonPower is always empty as x*x is a scalar."""
    Common._checkType(p, O, "associator")
    Common._checkType(q, O, "associator")
    Common._checkType(nonAlternate, bool, "associator")
    Common._checkType(nonPower, bool, "associator")
    if nonAlternate and nonPower:
      raise Exception("Invalid number of options in assoc")
    accum = []
    out = (self * p) *q - self *(p * q)
    if out and nonAlternate:
      for y in ((self, self, p), (p, self, self), (p, p, q),
                (q, p, p), (self, self, q), (q, self, self)):
        out = (y[0] *y[1]) *y[2] -y[0] *(y[1] *y[2])
        if out:
          break
    elif out and nonPower:
      for y in ((self, self, self), (p, p, p), (q, q, q)):
        out = (y[0] *y[1]) *y[2] -y[0] *(y[1] *y[2])
        if out:
          break
    return out
  assoc = associator

  def moufang(self, p, q, number=0):
    """moufang(p,q,[number])
       Return differences sum of all four Moufang tests for power-associate or
       just one if number is set (0=all)."""
    Common._checkType(p, O, "moufang")
    Common._checkType(q, O, "moufang")
    Common._checkType(number, int, "moufang")
    if number == 1:   out = q*(self *(q*p)) -((q*self) *q) *p
    elif number == 2: out = self *(q* (p*q)) -((self*q) *p) *q
    elif number == 3: out = (q*self) *(p*q) -(q *(self*p)) *q
    elif number == 4: out = (q*self) *(p*q) -q *((self*p) *q)
    elif number != 0:
      raise Exception("Invalid vaue for number in moufang")
    else:
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
    out,invCode,p2,n2 = self.__invertible()
    l2 = float(p2 + self.w *self.w)
    if invCode > 0 or l2 < precision:
      raise Exception("Illegal versor for rotate()")
    if p2 < precision:
      return q
    if abs(l2 -1.0) < precision:
      l2 = 1.0
    out = self /l2
    if abs(invCode) == 2:
      out = out.w +out.w -out
    return self *q *out /l2

  def rotation(self, rot):
    """rotation(rot)
       Rotate self inplace by rot, if necessary. Applying to versors rotates
       in the same sense as quaternions and frame. For O vectors this is the
       same as rot.inverse()*self*rot. Multiple rotations are TBD."""
    Common._checkType(rot, O, "rotation")
    precision = Common._getPrecision()
    out,invCode,p2,n2 = self.__invertible()
    l2 = float(n2 + rot.w *rot.w)
    if invCode > 0 or l2 < precision:
      raise Exception("Illegal versor for rotation()")
    if p2 < precision:
      return self.copy()
    if abs(l2 -1.0) < precision:
      l2 = 1.0
    out = self /l2
    if abs(invCode) == 2:
      out = out.w +out.w -out
    newSelf = rot *self *out /l2
    self.w = newSelf.w
    self.__g = newSelf.__g

  def frame(self):
    """frame()
       Return self=w+v as a frame=acos(w)*2 +v*len(w+v)/asin(w) taking vector
       part v. Ready for frameMatrix. Also handles hyperbolic versor. See
       versor."""
    precision = Common._getPrecision()
    out,invCode,p2,n2 = self.__invertible()
    l2 = p2 +self.w *self.w
    if invCode != 0 or abs(math.sqrt(l2) -1.0) >= precision:
      if not noError:
        raise Exception("Illegal versor for frame()")
    if p2 < precision:
      return O(1)
    if invCode > 0:
      w = abs(self.w)
      if w < 1.0:
        raise Exception("Invalid hyperbolic frame angle")
      out = self.copy(math.acosh(w))
      if self.w < 0:
        out.w = -out.w
    else:
      w = (self.w +1.0) %2.0 -1.0
      out =self.copy( math.acos(w) *2)
    p1 = mat.sqrt(p2)
    if p1 >= precision:
      if abs(invCode) == 2:
        p1 = -p1
      p0 = 1/p1
      for base in out.__g:
        base.value *= p0
    return out

  def versor(self, noError=False):
    """versor()
       Return a versor of length 1 assuming w is the angle(rad) ready for
       rotation. Opposite of frame. See normalise. Handles both signatures.
       Greater than 4-form versors are TBD."""
    precision = Common._getPrecision()
    out,invCode,p2,n2 = self.__invertible()
    w1 = self.w
    if invCode > 0:
      w1 = self.__g[-1].value
      p2 -= w1 *w1
    l2 = p2 +w1 *w1
    if invCode == 0 or l2 < precision:
      if not noError:
        raise Exception("Illegal versor for versor()")
    if p2 < precision:
      return O(1)
    if invCode > 0:
      sw = math.sinh(w1 /2.0)
      cw = math.cosh(w1 /2.0)
    else:
      sw,cw = Common._sincos(w1 /2.0)
    sw /= math.sqrt(p2)
    if invCode > 0:
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

  def normalise(self):
    """normalise()
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
    out,invCode,p2,n2 = self.__invertible()
    if invCode < 0:
      l1 = math.sqrt(p2 +self.w *self.w)
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
    out,invCode,p2,n2 = self.__invertible()
    if p2 < Common._getPrecision():
      return O(self.w)
    if invCode < 0:
      p1 = math.sqrt(p2)
      s,c = Common._sincos(p1)
      exp = pow(math.e, self.w)
      s *= exp /p1
      out = O(exp *c)
      for grade in self.__g:
        oStr,uStr = grade.strs()
        out += O(**{oStr +uStr: grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for exp")

  def log(self):
    """log()
       The functional inverse of the quaternion exp()."""
    out,invCode,p2,n2 = self.__invertible()
    l1 = math.sqrt(self.w *self.w +p2)
    if p2 < Common._getPrecision():
      return O(math.log(l1))
    if invCode < 0:
      s = math.acos(self.w /l1) /math.sqrt(p2)
      out = O(math.log(l1))
      for grade in self.__g:
        oStr,uStr = grade.strs()
        out += O(**{oStr +uStr: grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for log")

  def euler(self, noError=False):
    """euler([noError])
       Quaternion versors can be converted to Euler Angles & back uniquely for
       normal basis order. Error occurs for positive signature. Euler parameters
       are of the form cos(W/2) +m sin(W/2), m pure unit versor. Set noError to
       return a zero Euler if self is not valid to be a versor or norm. n>3 is
       under development and rotations should be 7-D, 15-D, etc. TBD."""
    out,invCode,p2,n2 = self.__invertible()
    l2 = p2 +self.w *self.w
    if invCode > -1:
      raise Exception("Illegal versor for euler")
    if invCode == -2:
      tmp = self.w +self.w -self  # Conjugate
    else:
      tmp = self.copy()
    if abs(l2 -1.0) >= Common._getPrecision():
      if not noError:
        raise Exception("Illegal versor norm for euler")
      tmp = tmp.normalise()
    if p2 < Common._getPrecision():
      return Euler()
    dims = self.basis()
    vecSiz = max(sum(dims), 3)
    args = [0] *max(vecSiz, int(Common.comb(sum(dims), 2)))
    xyz = list(x for x in O.BasisArgs(*dims) if len(x) == 3) # grade 2
    angles = [0] *vecSiz
    for grade in tmp.__g:
      base = "".join(grade.strs())
      if len(base) == 3:  # grade 2 only
        args[xyz.index(base)] = grade.value
    w, x, y, z = tmp.w, args[2], args[1], args[0]
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
    out.extend([0] *(sum(dims) - len(out)))
    return Euler(*out).matrix()

  def frameMatrix(self):
    """frameMatrix()
       Rodriges for 3-D. See https://math.stackexchange.com/questions/1288207/
       extrinsic-and-intrinsic-euler-angles-to-rotation-matrix-and-back.
       Converts self to versor then to euler then matrix."""
    out = self.versor().euler()
    dim = int(Common.comb(len(out), 2))
    dims = self.__vectorSizes()
    out.extend([0] *(sum(dims) - len(out)))
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
  def CayleyDicksonRule(baez=None):
    """cayleyDicksonRule([baez])
       Change Cayley-Dickson multiplication from Wikipedia to Baez (or back if
       baez is False). (a,b)*(c,d) = (a*c-d.conj*b, d*a +b*c.conj) [Wikipedia] or 
       (a,b)*(c,d) = (a*c-d*b.conj, a.conj*d +c*b) [J.C.Baez]."""
    if baez is not None:
      Common._checkType(baez, bool, "cayleyDicksonRule")
      O.__basisCache  = []
      O.__baezMulRule = baez
    return "baez" if O.__baezMulRule else "wiki"

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
       Return (o,u) basis elements as a list of names in addition order."""
    Common._checkType(oDim, int, "BasisArgs")
    Common._checkType(uDim, int, "BasisArgs")
    if oDim +uDim > O.__HEX_BASIS:
      raise Exception("Too many basis arguments")
    if oDim < 0 or uDim < 0:
      raise Exception("Too few basis arguments")
    if oDim +uDim == 0:
      return []
    arr = O._basisArray(oDim +uDim)[0]
    out = []
    oMax = "%X" %oDim
    for base in arr[1:int(pow(2, oDim +uDim))]:
      if base:
        if base[0] > oMax:
          out.append("u" +base)
        elif base[-1] <= oMax:
          out.append("o" +base)
        else:
          for idx in range(len(base)):
            if base[idx] > oMax:
              out.append("o%su%s" %(base[:idx], base[idx:]))
              break
    return out

  @staticmethod
  def VersorArgs(oDim, uDim=0, rotate=False):
    """VersorArgs(oDim, [uDim])
       Same as BasisArgs except o12 is negated. TBD"""
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
       Every O is a 7-D versor if scaled. Return versor() of inputs."""
    q = O(*args, **kwargs)
    return q.versor()

  @staticmethod
  def Euler(*args, **kwargs): #order=[], implicit=False):
    """Euler([angles, ...][o1=multiplier, ...][order, implicit])
       Euler angles in higher dimensions have (D 2)T=D(D-1)/2 parameters.
       SO(4) has 6 and can be represented by two octonions. Here they are
       changed to a versor using explicit rotation & this is returned.
       So in 3-D q' = (cx+sx e23) *(cy+sy e13) *(cz+sz e12).
       kwargs may contains "order" and "implicit".  The args arguments
       are entered as radian angles and rotations applied in the given order
       as shown using BasisArgs(). This order can be changed using the order
       array which must be as long as the list of angles. The default is 
       [1,2,...] and must have unique numbers. If implicit is set True then
       repeats. So Euler(x,y,z,order=[3,1,3]) is R=Z(x)X'(y)Z''(z). If the
       quat module is included then args can be a Euler object. n>3 is under
       development TBD."""
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
          xyz = O.BasisArgs(dim)
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
  def ZeroDivisors(pDim, nDim=0, dump=False):
    """ZeroDivisors(dim,[nDim=0,dump=False])
       Return all zero divisors (a+b)(c+d) as [[b,c,d,d1...]...] a=bcd at level
       pDim. Dump logs progress and checks memory and aborts if too small."""
    Common._checkType(pDim, int, "ZeroDivisors")
    Common._checkType(nDim, int, "ZeroDivisors")
    Common._checkType(dump, bool, "ZeroDivisors")
    out = []
    rng = O.Basis(pDim, nDim)
    lr = len(rng)
    cnt = 0
    Common.procTime()
    for b in range(lr):
      if dump and b %10 == 0:
        sys.stdout.write("%s (%ds) %d: total=%d %dMB\n" %(Common.date(True),
                       int(Common.procTime()), b, len(out), Common.freeMemMB()))
        if Common.freeMemMB() < Common._memLimitMB:
          sys.stdout.write("ABORT: Memory limit reached\n")
          break
      for c in range(b +1, lr):
        bb,cc = rng[b], rng[c]
        buf = [bb,cc]
        for d in range(c +1, lr):
          dd = rng[d]
          aa = bb *cc *dd
          if not aa.isScalar():
            if (aa+bb)*(cc+dd)==0:
              buf.append(dd)
              cnt += 1
        if len(buf) > 2:
          out.append(tuple(buf))
    if dump:
      sys.stdout.write("%s (%ds) total=%d\n" %(Common.date(True),
                       int(Common.procTime()), cnt))
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
    calcHelp = """Octonian/Sedenian Calculator - Process 30-dimensional basis
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
    signTyp = state.signVal
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
          val = "+(%s%s)" %(kw[None], val)
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
      signTyp = kw[None][0] if signTyp or kw[None][0] == "-" else ""
      line += signTyp +kw[None][1:]
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
  exp = Common.exp
  log = Common.log
  try:
    import importlib
  except:
    importlib = None          # No changing of calculators

  # O Unit test cases for Calc with Tests[0] being init for each case
  # Can only test 2-D rotations until euler stuff is updated. TBD
  Tests = [\
    """d30=radians(30); d60=radians(60); d45=radians(45); d90=radians(90)
       e=Euler(pi/6,pi/4,pi/2); c=2o2+3o3; c.basis(3)""",
    """# Test 1 Rotate via frameMatrix == versor half angle rotation.
       Rx=d60-o23; rx=(d60 -o23).versor()
       test = Rx.frameMatrix() *c.vector(); store = (rx*c*rx.inverse()).vector()
       Calculator.log(store == test, store)""",
    """# Test 2 Rotate via frameMatrix == versor.rotate(half angle)].
       Rx=d60+o23; rx=O.Versor(d60,o23=1)
       test = Rx.frameMatrix() *c.vector(); store = (rx.rotate(c)).vector()
       Calculator.log(store == test, store)""",
    """# Test 3 Rotate versor rotate == rotation of copy.
       Rx=d60+o23; rx=math.cos(d30) +o23*math.sin(d30)
       test = Rx.frameMatrix() *c.vector(); store = (rx*c*rx.inverse()).vector()
       Calculator.log(store == test, store)""",
    """# Test 4 Quat Euler == O Euler.
       test = O.Euler(pi/6,pi/4,pi/2)
       store = O.Euler(e,order=[1,2,3],implicit=False)
       Calculator.log( store == test, store)""",
    """# Test 5 Euler implicit rotation == other order, Rzyx==Rxy'z'' FAILS TBD.
       e=Euler(pi/6)
       if O.IsCalc("Q"):
         test = O.Q(Q.Euler(e, order=[1,2,3], implicit=True))
       else:
         test = O.Euler(e, order=[3,2,1])
         Common.precision(1E-12)
       store = O.Euler(e, order=[1,2,3], implicit=True)
       Calculator.log(store != test, store)""",
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
       r = d45 +i +j +k; store = r.normalise().euler().matrix()
       if O.IsCalc("Q"):
         test = r.normalise().versorMatrix()
       else:
         test = r.normalise().versorMatrix().reshape((3,3))
       Calculator.log(store == test, store)""",
       ]

  calc = Calculator(O, Tests)
  calc.processInput(sys.argv)
###############################################################################
