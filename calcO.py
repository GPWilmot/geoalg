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
## A quaternion maps as i->o1, j->o2, k->o12. u basis has the same 
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
__version__ = "0.7"
import math
from calcLib import *

################################################################################
class O():
  """Class to process octonions and unity elements consisting of 32-D numbers,
     half being positive definite (u1..uF) and half being standard imaginary
     octonions (o1..oF). O(q,p) specifies q the octonian dimension (o<hex>) and
     p specifies the dimension of octonions with unity base elements (u1..uF).

     The notation is the same as for CA so o1*o2=o12 but the basis is ungraded
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
  __HEX_BASIS   = 15                     # e and i basis size
  __HEX_CHARS   = ('A', 'B', 'C', 'D', 'E', 'F')
  __CA_CHARS    = ('e', 'i')             # CA basis chars only
  __BASIS_CHARS = ('o', 'u')             # O basis chars only
  __allChars    = ['o', 'u']             # Include CA
  __basisList   = ['', '']               # Store the dimension characters
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
        return O.Product(0,[self[1] * p[1]], 0)
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
       index is taken from _BasisArray()."""
    def __init__(self, value, bases):
      """Element with scalar and p, o & u bases as Product index & BasisArgs."""
      self._init(value, bases)
    def _init(self, value, bases):
      self.value = value
      self.__pBase = bases[0]
      self.__oBase = bases[1]
      self.__uBase = bases[2]
    def __str__(self):
      return "%s[%s,%s,%s]" %(self.value, self.__pBase,
                             self.__oBase, self.__uBase)
    __repr__ = __str__
    def __mergeStr(self, dupStr):
      sStr = sorted(dupStr)
      out = ""
      idx = 0
      sgn = 0
      while idx < len(sStr):
        ch = sStr[idx]
        if idx < len(sStr) -1:
          if ch == sStr[idx +1]:
            idx += 2
            sgn += 1
          else:
            out += ch
            idx += 1
        else:
          out += ch
          idx += 1
      return out, (sgn %2 == 1)

    def new(self, value):
      inherit = self.__new__(O.Grade)
      inherit._init(value, (0, "", ""))
      return inherit
    def bases(self):
      return (self.__pBase, self.__oBase, self.__uBase)
    def lens(self):
      """Octonians are ungraded so length is 1 if o &/or u set."""
      both = self.__oBase and self.__uBase
      return (1 if self.__oBase or both else 0,
              1 if self.__uBase and not both else 0)
    def strs(self, oCh='o', uCh='u'):    # O.__BASIS_CHARS[0/1]
      return ((oCh +self.__oBase) if self.__oBase else "",
              (uCh +self.__uBase) if self.__uBase else "")
    def str(self, oCh='o', uCh='u'):    # O.__BASIS_CHARS[0/1]
      return (oCh +self.__oBase) if self.__oBase else "" \
             + (uCh +self.__uBase) if self.__uBase else ""

    def copy(self, value=None):
      inherit = super().__new__(self.__class__)
      inherit._init(self.value if value is None else value,
                    (self.__pBase, self.__oBase[:], self.__uBase[:]))
      return inherit

    def isEq(self, cf, precision):
      """Return true if the grades are equal within precision."""
      return abs(self.value -cf.value) <= precision \
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
         the product. This is done row by row of the table so as to not rebuild
         the whole cache at the start. Rows are cached and the order doesn't
         change as rows and columns increase in size. This works for the sparse
         multiplication used by the graded lists."""
      value *= self.value
      lhs = self.bases()
      xyz = None
      bases = [0, "", ""]     # Base for lhs p, o and u, resp
      rBase = rhs[0] # Iterate rhs o and u
      lBase = lhs[0]
      row = O._basisCache(lBase)
      if not row or len(row) <= max(lBase, rBase):
        xyz, maxDim, baezMul = O._BasisArray()
        lp = [0] *len(xyz)
        rp = list(range(len(xyz)))
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
          bases[1] = self.__mergeStr(lhs[1] +rhs[1])[0]
          bases[2],sgn = self.__mergeStr(lhs[2] +rhs[2])
          if sgn:
            value = -value
        else:
          bases = rhs
      else:
        bases = lhs
      inherit = super().__new__(self.__class__)
      inherit._init(value, bases)
      return inherit

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
    self.w = args[0] if args else 0             # Scalar
    self.__g = []                               # Array of ordered Grades
    self.__currentAdd = -1                      # Previous add index
    Lib._checkType(self.w, (int, float), "O")
    if len(args) > 1:
      dim = int(math.log(len(args))/math.log(2) +1)       # l=pow(2,dim)
      if dim > self.__HEX_BASIS:
        raise Exception("Too many basis elements")
      xyz = O._BasisArray(dim)[0]   # Setup global dimension
      for idx,val in enumerate(args[1:]):
        Lib._checkType(val, (int, float), "O")
        if val:
          self.__g.append(O.Grade(val, [idx +1, xyz[idx +1], ""]))
    for key,value in kwargs.items():
      Lib._checkType(value, (int, float), "O")
      if not key:
        self.w += value
      elif value:
        self.__add(O._init(key, value, O.__BASIS_CHARS))

  def _initAdd(self, grade):
    """Used internally to add grades to this O."""
    self.__add(grade)

  @staticmethod
  def _init(key, value, baseChars):
    """Return the Grade for basis string key. Separate o & u parts."""
    lGrade = O.Grade(1, (0, "", ""))  # Base for o and u, resp
    grades = [lGrade, lGrade.copy()]
    rBases = [0, "", ""]
    typ = None
    baseCh = False
    lastChar = ''
    for char in key:
      if (char.isdigit() or char in O.__HEX_CHARS) and char > lastChar:
        lastChar = char
    xyz = O._BasisArray(lastChar)[0]
    lastChar,oneByOne  = '', False
    cntBases = 0
    for char in key:
      offset = int(typ == baseChars[1]) # o==0, u==1
      oneByOne = (oneByOne or char <= lastChar)
      if typ and char.isdigit():
        if char in O.__basisList[1 -offset]:
          raise Exception("Dimension already used by %s%s" \
                           %(baseChars[1 -offset], char))
        if oneByOne:
          rBases[0] = xyz.index("".join(sorted(rBases[offset +1])))
          grades[offset] = grades[offset].mergeBasis(1, rBases)
          rBases = [0, "", ""]
        lastChar = char
        if char not in O.__basisList[offset]:
          O.__basisList[offset] += char
        rBases[offset +1] += char
        baseCh = False
      elif typ and char in O.__HEX_CHARS:
        if char in O.__basisList[1 -offset]:
          raise Exception("Invalid basis: %s%s" %(typ, char))
        if oneByOne:
          rBases[0] = xyz.index(''.join(sorted(rBases[offset +1])))
          grades[offset] = grades[offset].mergeBasis(1, rBases)
          rBases = [0, "", ""]
        lastChar = char
        if char not in O.__basisList[offset]:
          O.__basisList[offset] += char
        rBases[offset +1] += char
        baseCh = False
      elif char in baseChars and not baseCh:
        if cntBases == 3 or char == typ:
          raise Exception("Invalid basis duplication: %s" %char)
        if rBases[offset +1]:
          rBases[0] = xyz.index(''.join(sorted(rBases[offset +1])))
          grades[offset] = grades[offset].mergeBasis(1, rBases)
          rBases = [0, "", ""]
        cntBases += 1
        typ = char
        baseCh = True
        lastChar,oneByOne  = '', False
      else:
        raise Exception("Invalid basis: %s" %key)
    if typ and baseCh:
      raise Exception("Invalid last basis: %s" %key)
    rBases[0] = xyz.index("".join(sorted(rBases[1] +rBases[2])))
    grades[offset] = grades[offset].mergeBasis(value, rBases)
    if cntBases == 1:
      return grades[offset]
    rBases[1] = grades[0].bases()[1]
    rBases[2] = grades[1].bases()[2]
    if typ == baseChars[0]:
      grades[0].value *= -1
    rBases[0] = xyz.index("".join(sorted(rBases[1] +rBases[2])))
    grades[0]._init(grades[0].value *grades[1].value, rBases)
    return grades[0]

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
        val = grade.value
        gOut = "".join(grade.strs())
        if gOut[:1] == "-":
          val = -grade.value
          gOut = gOut[1:]
      else:
        val = self.w
        gOut = ""
      out += Lib._resolutionDump(sign, val, gOut)
      if out:
        sign = " +"
    return out if out else "0"
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if Lib._isVerbose() and O.dumpRepr:
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __hash__(self):
    """Allow dictionary access for basis objects."""
    return hash(str(self))

  def __eq__(self, cf):
    """Return True if 2 Os are equal within precision."""
    precision = Lib._getPrecision()
    if isinstance(cf, (int, float)):
      return not self.__g  and abs(self.w -cf) <= precision
    elif not isinstance(cf, O):
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
      Lib._checkType(q, (int, float), "add")
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
    Lib._checkType(q, (int, float), "sub")
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
      out = self.__class__(self.w *q.w)
      if self.w:
        for grade2 in q.__g:
          grade = grade2.new(self.w)
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
      Lib._checkType(q, (int, float), "mul")
      out = self.__class__(self.w *q)
      if q:
        for grade in self.__g:
          out.__g.append(grade.copy(grade.value *q))
    return out
  __rmul__ = __mul__

  def __bool__(self):
    return self != 0
  __nonzero__ = __bool__

  def __div__(self, q):
    """Attempted division for 2 versors or self by scalar."""
    if isinstance(q, O):
      return self.__mul__(q.inverse())
    Lib._checkType(q, (int, float), "div")
    if abs(q) < Lib._getPrecision():
      raise Exception("Illegal divide by zero")
    if sys.version_info.major == 2 and isinstance(q, int): # Python v2 to v3
      q = float(q)
    out = self.__class__(self.w /q)
    for grade in self.__g:
      out.__g.append(grade.copy(grade.value /q))
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
    elif not isinstance(cf, O):
      raise Exception("Invalid comparison for O: %s" %type(cf))
    cfIdx = 0
    idx = 0
    order = 0
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
      v[base.str()] = base.value
    return v

  def __invertible(self, conj=True):
    """Return (conjugate, simple, hyperbolic, sum of basis squares).
       This is correct for simple forms but may fail otherwise.
       Flat = [number of imaginary terms, number of hyperbolic terms].
       Diff = Flat[0] == Flat[1] + 1 if scalar != 0.
       Simple = (Diff != Commutes) and 2 or less grades with scalar.
       Even = not appropriate as octonions are not graded
       Commutes = +ve/-ve terms commute.
       Hyperbolic = has x*x>0 terms but no imaginary terms."""
    sgnOut = self.__class__(0)
    out = self.__class__(self.w)
    p2 = 0
    lastDim = (0, 0)
    cnt = 0        # Count of total basis terms
    flat = [0, 0]  # Count of Imaginaries, Hyperbolics with different basis dims
    for grade in self.__g:
      dim = grade.lens()
      cnt += 1
      if dim != lastDim:
        lastDim = dim
        sgnVal = grade.copy(1)
        sgnVal = sgnVal.mergeBasis(1, grade.bases())
        flat[int(sgnVal.value > 0)] += 1
      value = grade.value
      p2 += value *value
      if conj:
        if sgnVal.value < 0: # Conjugate if len < 0
          value *= -1
          sgnOut.__g.append(grade.copy(value))
    scalar = (1 if self.w else 0)
    simple = False
    if conj:
      if cnt +scalar == 1:
        simple = True
      elif flat[0] == 1:
        if flat[1] == 0 and scalar:
          simple = True
        elif flat[1] + scalar == 1:
          simple = (flat[0] == flat[1])
    return out +sgnOut, simple, (flat[1] > 0 and flat[0] == 0), p2

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

  def _vectorSizes(self, local=False):
    """Return the Octonian vector sizes. Can't handle negative signatures."""
    dims = self.basis(local)
    if dims[0] == 1:
      dims[0] = 2
    xyz = self.__class__._BasisArgs(*dims)
    return len(xyz), xyz

  def _basisType(self):
    """Return index position +1 of paired sum over all grades or 0 for mixed."""
    if not self.__g:
      return 1   # For scalar only
    cnts = [0] *(len(self.__g[0].lens()) //2)
    for grade in self.__g:
      lens = grade.lens()
      for idx in range(len(lens) //2):
        cnts[idx] += sum(lens[idx*2: idx*2 +2])
    for idx in range(len(lens) //2):
      cnts[idx] = int(bool(cnts[idx]))
    if sum(cnts) == 1:
      return cnts.index(1) +1
    return 0

  @staticmethod
  def _BasisConvert(q, part=0, cls=None):
    """Return basis[part] as a dictionary if class is None else return first
       part using empty class as the output with basis[part] set."""
    if not isinstance(q, O):
      raise Exception("Conversion needs octonion class input")
    if not q.__g:
      return {} if cls is None else q.w  # For scalar only
    if cls is None:
      if part < 0 or part >= len(q.__g[0].lens()) //2:
        raise Exception("Invalid part size in conversion")
      out = {}
      idx = (part -1) *3 +2
      for grade in q.__g:
        if part == 0:
          bases = grade.strs()[0] +grade.strs()[1]
          if bases:
            out[bases] = grade.value
        else:
          bases = grade.bases()[idx:idx+3]
          if bases[0]:
            out[(O.__BASIS_CHARS[0] +bases[1]) if bases[1] else "" \
              +(O.__BASIS_CHARS[1] +bases[2]) if bases[2] else ""] = grade.value
      return out
    if q.__class__ != O:
      raise Exception("Conversion needs octonion input")
    if not isinstance(cls, O):
      raise Exception("Conversion needs octonion class")
    if part < 1 or part >= len(cls.Grade(0, [None,[]]).new(0).lens()) //2:
      raise Exception("Invalid part size in conversion")
    cls.w = q.w
    for grade in q.__g:
      cls.__g.append(cls.Grade(0, [None, []]).new(grade.value, part, grade.bases()))
    return cls

  @staticmethod
  def _BasisArray(dim=0, exact=False):
    """Used by Grade and BasisArgs and matches Grade.order.
       Returns basis digits list for current max dim = oDim + uDim,
       current max (increasing if dim > max dim) and multiplication rule."""
    if isinstance(dim, Lib._basestr):
      dim = int(dim, O.__HEX_BASIS +1)
    if dim > O.__basisDim or (exact and dim < O.__basisDim):
      out = [""]
      for ii in range(1, dim +1):
        form = "%X" %ii
        for val in out[:]:
          out.append(val +form)
      if dim > O.__basisDim:
        O.__basisXyz = out
        O.__basisDim = dim
    else:
      out = O.__basisXyz
    return out, O.__basisDim, O.__baezMulRule

  @staticmethod
  def _BasisArgs(oDim, uDim, och="o", uch="u"):
    """Used by BasisArgs and externally to return the basis strs."""
    dims = O.__basisList
    pDim = max(oDim, len(dims[0])) +max(uDim, len(dims[1])) 
    xyz = O._BasisArray(pDim, True)[0]
    typs = [och] *pDim                    # Setup basisList o/u types
    for idx in dims[1]:
      typs[int(idx, O.__HEX_BASIS +1) -1] = uch
    if oDim > len(dims[0]) or uDim > len(dims[1]):
      missing = []                        # Add needed oDim/uDim positions
      for pos in range(1, pDim +1):
        if "%X" %pos not in dims[0] +dims[1]:
          missing.append(pos -1)
      pos = 0
      for idx in range(oDim -len(dims[0])):
        typs[missing[pos]] = och
        pos += 1
      for idx in range(uDim -len(dims[1])):
        typs[missing[pos]] = uch
        pos += 1
    xyzMap = list("%X" %pos for pos in range(1, pDim +1))
    if oDim +uDim < pDim:                # Remove excess basis elements
      xyz = O._BasisArray(oDim +uDim, True)[0]
      xyzMap = []
      typsNew = []
      oPos = uPos = 0
      for idx,ch in enumerate(typs):
        if ch == och and oPos < oDim:
          typsNew.append(och)
          xyzMap.append("%X" %(idx +1))
          oPos += 1
        elif ch == uch and uPos < uDim:
          typsNew.append(uch)
          xyzMap.append("%X" %(idx +1))
          uPos += 1
      typs = typsNew
    out = []
    for base in xyz[1:]:
      tmp = ""
      typ = ""
      for ch in base:
        idx = int(ch, O.__HEX_BASIS +1) -1
        if typ != typs[idx]:
          typ = typs[idx]
          tmp += typ +xyzMap[idx]
        else:
          tmp += xyzMap[idx]
      out.append(tmp)
    return out

  @staticmethod
  def _VersorArgs(oDim, uDim, och="o", uch="u"):
    """Used by VersorArgs and externally to return the versor strs."""
    return O._BasisArgs(oDim, uDim, och, uch)

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

  def isVersor(self, hyperbolic=False):
    """isVersor([hyperbolic])
       Return true if negative signature inverse."""
    precision = Lib._getPrecision()
    conj,simple,isHyperbolic,p2 = self.__invertible()
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
    if scalar is None:
      out = self.__class__(self.w)
    else:
      Lib._checkType(scalar, (int, float), "dup")
      out = self.__class__(scalar)
    for grade in self.__g:
      out.__g.append(grade.copy())
    return out

  def copy(self, *args, **kwargs):
    """copy([scalar, e1 multiplier, ...][basis=multiplier, ...])
       Return clone with optional new basis values."""
    kw = self.__copy()
    kw.update(kwargs)
    if len(args) == 0:
      args = [self.w]
    out = self.__class__(*args, **kw)
    return out

  def copyTerms(self):
    """copyTerms()
       Return terms as a list of pairs of (term, factor). Cf O(**dict(...))."""
    v = [("", self.w)] if self.w else []
    for grade in self.__g:
      v.append((grade.str(), grade.value))
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
  
  def cycles(self, c):
    """cycles(y):
       Return ((b,c),(b,bc),(c,bc)), b=self, bc=abs(b*c), Fast Lib.cycles()."""
    b,c,bc = sorted((self, c, abs(self *c)))
    return ((b, c), (b, bc), (c, bc))

  def cyclesType(self, c, d):
    """cyclesType(c, d)
       Return 3 chars of nonAssocType() for the self.cycles(c) with d."""
    Lib._checkType(c, O, "cyclesType")
    Lib._checkType(d, O, "cyclesType")
    return "".join(x[0].nonAssocType(x[1], d) for x in self.cycles(c))

  def trim(self, precision=None):
    """trim([precision])
       Remove elements smaller than precision."""
    if precision is None:
      precision = Lib._getPrecision()
    else:
      Lib._checkType(precision, float, "trim")
    out = self.__class__(0 if abs(self.w) < precision else self.w)
    for grade in self.__g:
      if abs(grade.value) >= precision:
        out.__g.append(grade.copy())
    return out

  def pure(self):
    """pure()
       Return the pure imaginary or unity part of self."""
    return self.dup(0)

  def vector(self, local=False, size=None):
    """vector([local,size])
       Return the coefficients as a 1-D Matrix optionally reshaped."""
    dim,xyz = self._vectorSizes(local)
    vec = [0] *dim
    for grade in self.__g:
      pos = xyz.index(grade.str())
      if pos < dim:
        vec[pos] = grade.value
    if size:
      return Matrix(*vec).reshape(size)
    return Matrix(*vec)

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
      bases = grade.strs()
      l = 0
      for base in bases:
        l += (len(base) -1) if base else 0
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

  def basis(self, local=False):
    """basis([local])
       Return the local signature or max. dimension basis of basis elements."""
    dims = self.__basisList
    if local:
      dims = [''] *local if isinstance(local, int) else ['', '']
      for grade in self.__g:
        for idx,bases in enumerate(grade.strs()):
          if idx < local:
            dims[idx] = max(dims[idx][-1:], bases[-1:])
    out = []       # Convert max. char to hex-digit
    for val in dims:
      out.append(int(sorted(val)[-1], self.__HEX_BASIS +1) if val else 0)
    return out

  def len(self):
    """len()
       Return the scalar square sum of the product with it's conjugate."""
    n2 = self.w*self.w
    for grade in self.__g:
      sgnVal = grade.copy(1)
      sgnVal = sgnVal.mergeBasis(1, grade.bases())
      n2 += grade.value *grade.value *sgnVal.value
    if n2 < 0:
      return -math.sqrt(-n2)
    return math.sqrt(n2)

  def pureLen(self):
    """pureLen()
       Return the signed len of the pure part only."""
    n2 = 0
    for grade in self.__g:
      sgnVal = grade.copy(1)
      sgnVal = sgnVal.mergeBasis(1, grade.bases())
      n2 += grade.value *grade.value *sgnVal.value
    if n2 < 0:
      return -math.sqrt(-n2)
    return math.sqrt(n2)

  def conjugate(self, split=False):
    """conj[ugate]([split])
       Return copy of self with pure parts negated (imaginary only if split)."""
    out = self.dup()
    if split:
      for grade in out.__g:
        sgnVal = grade.copy(1)
        sgnVal = sgnVal.mergeBasis(1, grade.bases())
        grade.value *= sgnVal.value
    else:
      for grade in out.__g:
        grade.value = -grade.value
    return out
  conj = conjugate

  def norm(self):
    """norm()
       Return the scalar sqrt of the product with it's conjugate."""
    p2 = self.w *self.w
    for grade in self.__g:
      p2 += grade.value *grade.value
    return math.sqrt(p2)

  def inverse(self, noError=False):
    """inverse([noError])
       Return inverse of self which is conj()/len() if len()!=0 and is a versor.
       Simple forms with only two grades can be checked immediately. Simple
       odd and mixed forms are also immediate. But with more than two grades
       an inverse is possible so try self*self & self*conj(). If this produces a
       single term then divide by this term which captures examples 3e1234567 
       +e123 +e145 +e167 +e246 +e257 +e347 +e356 and (e0+e1+e2+e12)^2 = 2e012
       (ie inverse -e0+e01-e02-e12). If not invertible then raise an exception
       unless not noError in which case 0 is returned.
       NB: Row11,0(paper)*e1234567 = 3 idempotents+2 is invertible but factors
       aren't eg (1+e1236)(1+e1467)(1+e3567)."""
    out,simple,isHyperbolic,p2 = self.__invertible()
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

  def cross(self, q):
    """cross(q)
       Return half asym product of pure parts."""
    Lib._checkType(q, O, "cross")
    x = self.__class__()
    x.__g = self.__g    # Shallow copies
    y = self.__class__()
    y.__g = q.__g
    out = (x *y -y *x) *0.5 
    return out

  def sym(self, q):
    """sym(q)
       Return symmetric product of two Os. The pure part is always zero."""
    Lib._checkType(q, O, "sym")
    return (self *q +q *self)

  def asym(self, q):
    """asym(q)
       Return antisymmetric product of two Os. Cross product is pure part."""
    Lib._checkType(q, O, "asym")
    return (self *q -q *self)
 
  def associator(self, p, q, alternate=False):
    """assoc[iator](p,q, [alternate])
       Return the associator [self,p,q] = (self * p) *q - self *(p * q) or
       the first alternate where alternate is [x,y,z]!=0 for any pair equal
       and x,y,z in {self,p,q}. Any scalar gives zero."""
    Lib._checkType(p, O, "associator")
    Lib._checkType(q, O, "associator")
    Lib._checkType(alternate, bool, "associator")
    accum = []
    out = self.__assoc(p, q)
    if out and alternate:
      none = True
      for y in ((p, self, q), (p, q, self), (self, q, p)):
        if out != -(y[0] *y[1]) *y[2] -y[0] *(y[1] *y[2]):
          none = False
          break
      if none:
        out = 0
    return out
  assoc = associator
  def __assoc(self, p, q):
    return (self * p) *q  -self *(p * q)

  def tripleAssociator(self, c, d):
    """tripleAssoc[iator](c,d)
       Return [b,d,c]+[b,c,d]+[c,b,d]. 0 if associative else non-associative."""
    Lib._checkType(c, O, "tripleAssociator")
    Lib._checkType(d, O, "tripleAssociator")
    return self.__assoc(d,c) +self.__assoc(c,d) + c.__assoc(self,d)
  tripleAssoc = tripleAssociator

  def moufang(self, p, q, number=0):
    """moufang(p,q,[number=0])
       Return differences of the four Moufang tests or sum of all if number=0,
         1: q*(s*(q*p)) -((q*s)*q)*p, 2: s*(q*(p*q)) -((s*q)*p)*q,
         3: (q*s)*(p*q) -(q*(s*p))*q, 4: (q*s)*(p*q) -q*((s*p)*q)."""
    Lib._checkType(p, O, "moufang")
    Lib._checkType(q, O, "moufang")
    Lib._checkType(number, int, "moufang")
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

  def malcev(self, p, q):
    """malcev(p,q) x=s y=p z=q
       Return (sp)(sq)-((sp)q)s -((pq)s)s -((qs)s)p, s=self."""
    Lib._checkType(p, O, "malcev")
    Lib._checkType(q, O, "malcev")
    return self *p *(self *q) -self *p *q *self -p *q *self *self -q *self *self *p

  def abcAssociator(self, c, d, abc=0):
    """abcAssoc[iator](c,d,[abc=0])
       Return [b,d,c] or [b,c,d] or [c,b,d] for abc=1-3 and all associative
       (ie all==0) for abc=0 where b=self. This is the same associativity as
       [b,a,c], [a,b,c], or [a,c,b] where a=b*c*d."""
    Lib._checkType(c, O, "abcAssociator")
    Lib._checkType(d, O, "abcAssociator")
    Lib._checkType(abc, int, "abcAssociator")
    anyAssoc = (abc == 0 or abs(abc) == 4)
    if abc < 0 or abc > 3:
      raise Exception("Invalid abc parameter for abcAssociator")
    if abc <= 1:
      ass = self.__assoc(d, c)
      if ass or abc == 1:
        return ass
    if abc in (0, 2):
      ass = self.__assoc(c, d)
      if ass or abc == 2:
        return ass
    if abc in (0, 3):
      ass = c.__assoc(self, d)
      if ass or abc == 3:
        return ass
    return 0
  abcAssoc = abcAssociator

  def nonAssocType(self, c, d):
    """nonAssocType(c, d)
       Return character from "ABCXs." for triad with ABC: abc non-associativity,
       X: completely non-associative, s:scalar, .: repeated elements. Since 
       associtivity is paired (ab,bc,ac) then non-associativive is (C,A,B),
       respectively, or all (X), trivial (.) or quaternion-like (s)."""
    Lib._checkType(c, O, "nonAssocType")
    Lib._checkType(d, O, "nonAssocType")
    a = self *c *d
    if self == c or d in (self,c):
      out = "."
    elif a.isScalar():
      out = "s"
    else:
      out = ""
      if self.__assoc(a, c) != 0:
        out += "A"
      if a.__assoc(self, c) != 0:
        out += "B"
      if a.__assoc(c, self) != 0:
        out += "C"
      if len(out) == 3:
        out = "X"
    return out

  def nonAssocMode(self, c, d, mode=0):
    """nonAssocMode(c, d, [mode=0])
       Return b',c',d' for (b,c,d),(b,c,a),(b,c,db),(b,c,a') where a=b*c*d, db=d*b
       and a'=b*c*db which are Prim, Dual, Extended or Both for mode=0,1,2,3. Or
       return a',b',c',d' ordered for zero divisor uniquness for mode+=4."""
    Lib._checkType(c, O, "nonAssocMode")
    Lib._checkType(d, O, "nonAssocMode")
    Lib._checkType(mode, int, "nonAssocMode")
    abcMode = mode if mode < 4 else mode -4
    a = self *c *d
    if abcMode == 0:
      out = (self,c,d)
    elif abcMode == 1:
      out = (self,c,a)
    else:
      db = abs(d *self)
      aa = abs(self *c *db)
      if abcMode == 2:
        out = (self,c,db)
      elif abcMode == 3:
        out = (self,c,aa)
      else:
        raise Exception("Invalid value for mode in nonAssocMode")
    if mode > 3:
      b,c,d = out
      a = b *c *d
      if abs(a) > abs(b): a,b = b,a
      if abs(c) > abs(d): c,d = d,c
      if d < 0: c,d = -c,-d
      if b < 0: a,b = -a,-b
      if abs(a) > abs(c):
        a,b,c,d = c,d,a,b
      if c < 0: a = -a; c = -c
      out =  (a,b,c,d)
    return out

  def nonAssocModeType(self, c, d):
    """nonAssocModeType(c, d)
       Return nonAssocType for (a+b)(c+d),(-d+b)(c+a),(a'+b)(c+db),(-db+b)(c+a')
       where b=self, a=b*c*d, db=d*b and a'=a*b*db as "p?d?e?b?". This is Primary,
       Dual, Extended and Both which is extended dual. See nonAssocMode()."""
    Lib._checkType(c, O, "nonAssocModeType")
    Lib._checkType(d, O, "nonAssocModeType")
    a = self *c *d
    db = abs(d *self)
    aa = self *c *db
    zeroDivMap = (("p", a, self, c, d),   ("d", -d, self, c, a),
                  ("e", aa, self, c, db), ("b", -db, self, c, aa))
    return "".join(map(lambda x: ((x[0] +x[2].nonAssocType(x[3], x[4])) \
                   if (x[1]+x[2])*(x[3]+x[4]) == 0 else ""), zeroDivMap))

  def projects(self, q):
    """projects(q)
       Return (parallel, perpendicular) parts of vector q projected onto self
       interpreted as a plane a*b with parts (in,outside) the plane. If q is
       a*b, a != b then return parts (perpendicular, parallel) to plane of a &
       b. a.cross(b) is not needed as scalar part is ignored."""
    Lib._checkType(q, O, "projects")
    n1 = abs(self.pureLen())
    if n1 < Lib._getPrecision():
      raise Exception("Invalid length for projects")
    mul = self.pure()
    vect = q.pure()
    mul = mul *vect *mul /float(n1 *n1)
    return (vect +mul)/2.0, (vect -mul)/2.0
  
  def rotate(self, q):
    """rotate(q)
       Rotate q by self. See rotation."""
    Lib._checkType(q, O, "rotate")
    precision = Lib._getPrecision()
    conj,simple,isHyperbolic,p2 = self.__invertible()
    l2 = float(p2 + self.w *self.w)
    if l2 <= precision or not simple:
      conj = self.__versible(conj)
      if conj == 0:
        raise Exception("Illegal versor for rotate")
    if p2 <= precision:
      return q.dup()
    if abs(math.sqrt(l2) -1.0) <= precision:
      l2 = 1.0
    return self *q *conj /l2

  def rotation(self, rot):
    """rotation(rot)
       Rotate self inplace by rot, if necessary. Applying to versors rotates
       in the same sense as quaternions and frame. For O vectors this is the
       same as rot.inverse()*self*rot. Multiple rotations are TBD."""
    Lib._checkType(rot, O, "rotation")
    precision = Lib._getPrecision()
    conj,simple,isHyperbolic,p2 = rot.__invertible()
    l2 = float(p2 + rot.w *rot.w)
    if l2 <= precision or not simple:
      conj = self.__versible(conj)
      if conj == 0:
        raise Exception("Illegal versor for rotation()")
    if p2 <= precision:
      return
    if abs(math.sqrt(l2) -1.0) <= precision:
      l2 = 1.0
    newSelf = rot *self *conj /l2
    self.w = newSelf.w
    self.__g = newSelf.__g

  def frame(self, hyperbolic=False):
    """frame([hyperbolic])
       Return self=w+v as a frame=acos(w)*2 +v*len(w+v)/asin(w) for vector v.
       Ready for frameMatrix. Also handles hyperbolic versor. See versor.
       Set hyperbolic to try an hyperbolic angles."""
    precision = Lib._getPrecision()
    conj,simple,isHyperbolic,p2 = self.__invertible()
    l2 = p2 +self.w *self.w
    if abs(math.sqrt(l2) -1.0) > precision:
      raise Exception("Illegal versor norm for frame")
    if isHyperbolic and not hyperbolic:
      raise Exception("Illegal hyperbolic versor for frame")
    if not simple:
      if self.__versible(conj) == 0:
        raise Exception("Illegal versor for frame")
    if p2 < precision:
      return self.__class__(1)
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
       rotation. Opposite of frame. See normalise. Hyperbolic versors use
       cosh and sinh expansions if hyperbolic is set."""
    precision = Lib._getPrecision()
    tmp,simple,isHyperbolic,p2 = self.__invertible()
    l2 = p2 +self.w *self.w
    if math.sqrt(l2) <= precision or not simple:
      raise Exception("Illegal versor for versor")
    if isHyperbolic and not hyperbolic:
      raise Exception("Illegal hyperbolic versor for versor")
    if p2 < precision:
      return self.__class__(1)
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
       Return vector & scalar part with the vector as length one."""
    out = self.dup()
    n2 = 0
    for base in out.__g:
      n2 += base.value *base.value
    if n2 > Lib._getPrecision():
      n1 = math.sqrt(n2)
      for base in out.__g:
        base.value /= n1
    return out

  def distance(self, q):
    """distance(qa)
       Return the Geodesic norm which is the half angle subtended by
       the great arc of the S3 sphere d(p,q)=|log(p.inverse())*q|."""
    Lib._checkType(q, O, "distance")
    if self.isVersor(True) and q.isVersor(True):
      return abs((self.inverse() *q).log().len())
    raise Exception("Invalid non-hyperbolic, non-versor for distance")

  def normalise(self):
    """normalise()
       Normalise - reduces error accumulation. Versors have norm 1."""
    n = self.norm()
    if n <= Lib._getPrecision():
      return self.__class__(1.0)
    out = self.dup(self.w /n)
    for base in out.__g:
      base.value /= n
    return out

  def pow(self, exp):
    """pow(exp)
       For even q=w+v then a=|q|cos(a) & v=n|q|sin(a), n unit."""
    Lib._checkType(exp, (int, float), "pow")
    if isinstance(exp, int):
      out = self.__class__(1.0)
      for cnt in range(exp):
        out *= self
      return out
    tmp,simple,isHyperbolic,p2 = self.__invertible()
    if simple and not isHyperbolic:
      l1 = math.sqrt(p2 +self.w *self.w)
      w = pow(l1, exp)
      if l1 <= Lib._getPrecision():
        return self.__class__(w)
      a = math.acos(self.w /l1)
      s,c = Lib._sincos(a *exp)
      s *= w /math.sqrt(p2)
      out = self.__class__(w *c)
      for grade in self.__g:
        oStr,uStr = grade.strs()
        out += self.__class__(**{oStr +uStr: grade.value *s})
      return out
    raise Exception("Invalid float exponent for non-hyperbolic, non-versor pow")
  __pow__ = pow

  def exp(self):
    """exp()
       For even q=w+v then exp(q)=exp(w)exp(v), exp(v)=cos|v|+v/|v| sin|v|."""
    tmp,simple,isHyperbolic,p2 = self.__invertible()
    if p2 <= Lib._getPrecision():
      return self.__class__(self.w)
    if not isHyperbolic:
      n1 = math.sqrt(p2)
      s,c = Lib._sincos(n1)
      exp = pow(math.e, self.w)
      s *= exp /n1
      out = self.__class__(exp *c)
      for grade in self.__g:
        out += self.__class__(**{grade.str(): grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for exp")

  def log(self):
    """log()
       The functional inverse of the quaternion exp()."""
    tmp,simple,isHyperbolic,p2 = self.__invertible()
    l1 = math.sqrt(p2 +self.w *self.w)
    if p2 <= Lib._getPrecision():
      return self.__class__(math.log(l1))
    if not isHyperbolic:
      s = math.acos(self.w /l1) /math.sqrt(p2)
      out = self.__class__(math.log(l1))
      for grade in self.__g:
        out += self.__class__(**{grade.str(): grade.value *s})
      return out
    raise Exception("Invalid non-hyperbolic, non-versor for log")

  def euler(self, hyperbolic=False):
    """euler([hyperbolic])
       Quaternion versors can be converted to Euler Angles & back uniquely for
       normal basis order. Error occurs for positive signature. Euler parameters
       are of the form cos(W/2) +m sin(W/2), m pure unit versor. Only defined
       for o1, o2, o12 quaternion part. Set hyperbolic to try hyperbolic
       angles."""
    precision = Lib._getPrecision()
    conj,simple,isHyperbolic,p2 = self.__invertible()
    l2 = p2 +self.w *self.w
    if not simple:
      raise Exception("Illegal versor for euler")
    if abs(math.sqrt(l2) -1.0) > precision:
      raise Exception("Illegal versor norm for euler")
    if isHyperbolic and not hyperbolic:
      raise Exception("Illegal hyperbolic versor for versor")
    if p2 <= precision:
      return Euler()
    dim,xyz = self._vectorSizes()
    angles = [0] *dim
    args = [0] *dim
    for grade in conj.__g:
      args[xyz.index(grade.str())] = grade.value
    w, x, y, z = conj.w, args[0], args[1], args[2]
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
       This is same as frameMatrix but for a versor ie half the angle.
       Converts self to euler than to matrix assuming normal basis order.
       Only defined for o1, o2, o12 quaternion part. Set hyperbolic to try
       hyperbolic angles."""
    out = self.euler(hyperbolic)
    dim,xyz = self._vectorSizes()
    return Euler(*out).matrix().reshape(dim)

  def frameMatrix(self, hyperbolic=False):
    """frameMatrix([hyperbolic])
       Rodriges for 3-D. See https://math.stackexchange.com/questions/1288207/
       extrinsic-and-intrinsic-euler-angles-to-rotation-matrix-and-back.
       Only defined for o12, o23, o13 quaternion part. Converts self to
       versor then to euler then matrix. Set hyperbolic to try hyperbolic
       angles."""
    out = self.versor().euler(hyperbolic)
    dim,xyz = self._vectorSizes()
    return Euler(*out).matrix().reshape(dim)

  def morph(self, pairs):
    """morph(pairs)
       Morphism with a list of pairs of names with o1,o2 meaning map o1->o2."""
    out = self.__class__(self.w)
    for grade in self.__g:
      out += self.__class__(**Lib._morph(grade.str(), grade.value, pairs))
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
      Lib._checkType(baez, bool, "cayleyDicksonRule")
      O.__basisCache  = []
      O.__baezMulRule = baez
    return "baez" if O.__baezMulRule else "wiki"

  @staticmethod
  def Basis(oDim, uDim=0):
    """Basis(pDim, [nDim])
       Return (o,u) basis elements with value one."""
    Lib._checkType(oDim, int, "Basis")
    Lib._checkType(uDim, int, "Basis")
    if oDim < 0 or uDim < 0 or oDim > O.__HEX_BASIS or uDim > O.__HEX_BASIS:
      raise Exception("Invalid Basis argument size")
    return tuple((O(**{x: 1}) for x in O._BasisArgs(oDim, uDim)))

  @staticmethod
  def BasisArgs(oDim, uDim=0):
    """BasisArgs(oDim, [uDim])
       Return (o,u) basis elements as a list of names in addition order."""
    Lib._checkType(oDim, int, "BasisArgs")
    Lib._checkType(uDim, int, "BasisArgs")
    if oDim < 0 or uDim < 0 or oDim > O.__HEX_BASIS or uDim > O.__HEX_BASIS:
      raise Exception("Invalid Basis argument size")
    return O._BasisArgs(oDim, uDim)

  @staticmethod
  def VersorArgs(oDim, uDim=0):
    """VersorArgs(oDim, [uDim])
       Just same as BasisArgs for octonions."""
    Lib._checkType(oDim, int, "VersorArgs")
    Lib._checkType(uDim, int, "VersorArgs")
    if oDim < 0 or uDim < 0 or oDim > O.__HEX_BASIS or uDim > O.__HEX_BASIS:
      raise Exception("Invalid VersorArgs argument size")
    return O._VersorArgs(oDim, uDim)

  @staticmethod
  def Versor(*args, **kwargs):
    """Versor([scalar, o1 multiplier, ...][basis=multiplier, ...])
       Every O is a 7-D versor if scaled. Return versor() of inputs."""
    return O.Euler(*args, **kwargs)

  @staticmethod
  def Euler(*args, **kwargs): #order=[], implicit=False):
    """Euler([angles, ...][o1=multiplier, ...][order,implicit])
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
    Lib._checkType(order, (list, tuple), "Euler")
    Lib._checkType(implicit, bool, "Euler")
    if len(args) == 1 and isinstance(args[0], Euler):
      args = list(args[0])
    out = O(1.0)
    implicitRot = O(1.0)
    store = []
    dim = int((math.sqrt(8*len(args) +1) +1) /2 +0.9) # l=comb(dim,2)
    xyz = O._VersorArgs(dim,0)
    args = list(args)
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
      Lib._checkType(key, (float, int), "Euler")
      key = int(key)
      if key in store or key > len(args):
        raise Exception("Invalid order index for rotation: %s" %key)
      ang = args[key -1]
      Lib._checkType(ang, (int, float), "Euler")
      s,c = Lib._sincos(ang *0.5)
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
  def AssocTriads(basis, nonAssoc=False, alternate=False, dump=False,
                  cntOnly=False):
    """AssocTriads(basis,[nonAssoc,alternate,dump,cntOnly])
       Return unique O.assoc(...) traids or not. See Lib.triadDump."""
    Lib._checkType(nonAssoc, bool, "AssocTriads")
    Lib._checkType(alternate, bool, "AssocTriads")
    tmp = Lib.triadPairs(O.__AssocTriads, basis, "AssocTriads", dump, alternate, cntOnly)
    if not nonAssoc: return tmp
    return Lib.triadPairs(Lib._allTriads, basis, "AssocTriads", dump, tmp)
  @staticmethod
  def __AssocTriads(out, basis, lr, a, b, params):
    cnt = 0
    buf = []
    aa,bb,alternate = params
    for c in range(b +1, lr):
      cc = basis[c]
      if not aa.assoc(bb, cc, alternate):
        buf.append(c)
        cnt += 1
    if out:
      out[a *lr +b] = buf
    return cnt
 
  @staticmethod
  def MoufangTriads(basis, number=0, nonMoufang=False,dump=False,cntOnly=False):
    """MoufangTriads(basis,[number,nonMoufang,dump,cntOnly])
       Return unique O.moufang() or non traids. See Lib.triadDump."""
    Lib._checkType(number, int, "MoufangTriads")
    Lib._checkType(nonMoufang, bool, "MoufangTriads")
    if cntOnly and nonMoufang:
      raise Exception("MoufangTriads cntOnly and nonMoufang is invalid")
    tmp = Lib.triadPairs(O.__MoufangTriads, basis, "MoufangTriads", dump,
                         number, cntOnly)
    if not nonMoufang: return tmp
    return Lib.triadPairs(Lib.inverseTriads, basis, "MoufangTriads", dump, tmp, cntOnly)
  @staticmethod
  def __MoufangTriads(out, basis, lr, a, b, params):
    cnt = 0
    buf = []
    aa,bb,number = params
    for c in range(b +1, lr):
      cc = basis[c]
      if not aa.moufang(bb, cc, number):
        buf.append(c)
        cnt += 1
    if out:
      out[a *lr +b] = buf
    return cnt

  @staticmethod
  def AbcTriads(basis, abc=0, nonAssoc=False, dump=False, cntOnly=False):
    """AbcTriads(basis,[abc,nonAssoc,dump,cntOnly])
       Return unique abcAssoc() traids or !=0. See Lib.triadDump."""
    Lib._checkType(abc, int, "AbcTriads")
    Lib._checkType(nonAssoc, bool, "AbcTriads")
    if cntOnly and nonMoufang:
      raise Exception("AbcTriads cntOnly and abc is invalid")
    tmp = Lib.triadPairs(O.__AbcTriads, basis, "AbcTriads", dump, abc, cntOnly)
    if not nonAssoc: return tmp
    return Lib.triadPairs(Lib.inverseTriads, basis, "AbcTriads", dump, tmp, cntOnly)
  @staticmethod
  def __AbcTriads(out,basis,lr,a,b, params):
    cnt = 0
    buf = []
    aa,bb,abc = params
    for c in range(b +1, lr):
      if not aa.abcAssociator(bb, basis[c], abc):
        buf.append(c)
        cnt += 1
    if out:
      out[a *lr +b] = buf
    return cnt

  @staticmethod
  def ZeroDivisors(basis, dump=False, cntonly=False):
    """zeroDivisors(basis,[dump,cntonly])
       Return zero divisors (abc+a)(b+c)=0 as triadPairs. See Lib.triadDump."""
    Lib._checkType(basis, (list, tuple), "zeroDivisors")
    return Lib.triadPairs(O.__ZeroDivisors, basis, "zeroDivisors", dump,
                             None, cntonly)
  @staticmethod
  def __ZeroDivisors(out, basis, lr, b, c, params):
    """Could use c-associativity but (a+b)*(c+d)==0 is quicker."""
    cnt = 0
    br = b *lr
    cr = c *lr
    bufout = []
    bb,cc,dummy = params
    for d in range(lr):
      dr = d *lr
      dd = basis[d]
      aa = bb *cc *dd
      aaa = abs(aa)
      if not aa.isScalar():
        a = basis.index(aaa)
        if a not in (b,c) and ((aa +bb) *(cc +dd) == 0 or \
           (aa +bb) *(cc -dd)==0):  # For split signature
          addit = True
          bd = br +d if b < d else dr +b
          if a in out[bd] or c in out[bd]:
            addit = False
          else:
            ar = a *lr
            ac = ar +c if a < c else cr +a
            if a in out[ac] or d in out[ac]:
              addit = False
            else:
              ad = ar +d if a < d else dr +a
              if b in out[ad] or c in out[ad]:
                addit = False
          if addit:
            bufout.append(d)
            cnt += 1
    if out and bufout:
      out[br +c] = tuple(bufout)
    return cnt

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
            if base and base[0] not in O.__BASIS_CHARS:
              base = "o" +base
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
        for term,base in enumerate(("o","","u")):
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
    return O(scalar, **out)

  @staticmethod
  def Q(*args):
    """Q([scalar, x, y, z])
       Map quaternion basis (w,i,j,k) to (w, o1, o2, o12) with up to 4
       arguments. If calc(Q) included then w may instead be a Q object."""
    if Lib.isCalc("Q"):     # If module calcQ included can use Q class
      if len(args) == 1 and isinstance(args[0], Q):
        q = args[0]
        args = []
        for val in (q.w, q.x, q.y, q.z):
          args.append(val)
    xyz = O.VersorArgs(2)
    if len(args) > 4:
      raise Exception("Invalid O parameters")
    kwargs = {}
    for idx,val in enumerate(args[1:]):
      kwargs[xyz[idx]] = val if idx < 3 else -val
    return O(args[0], **kwargs)

  @staticmethod
  def SetQuaternions():
    if len(O.__basisList[0]) >= 2:
      x1,x2 = O.__basisList[0][0], O.__basisList[0][1]
      return ("i,j,k=O(o%s=1),O(o%s=1),O(o%s=1)" %(x1, x2, x1+x2),
              "o%s,o%s,o%s" %(x1, x2, x1 +x2))
    elif len(O.__basisList[1]) >= 3:
      x1,x2,x3 = O.__basisList[1][0], O.__basisList[1][1], O.__basisList[1][2]
      return ("i,j,k=O(u%s=1),O(u%s=1),O(u%s=1)" %(x1 +x2, x2 +x3, x3+x1),
              "u%s,u%s,u%s" %(x1 +x2, x2 +x3, x3 +x1))
    else:
      return ("", "")

  ###################################################
  ## Calc class help and basis processing methods  ##
  ###################################################
  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for O."""
    cHelp = """Octonion/Sedenion/Ultra Calculator - Process 30-dimensional basis
          numbers (o1..F or u1..F) and multiples. The form oijk, for example,
          means left expansion, oi*oj*ok = ((oi*oj)*ok, and is internalised as
          a graded form with a subset of ordered indices from o123456789ABCDEF."""
    ijk = O.SetQuaternions()[0]
    return (("O", "CA", "Q", "R"), ("O", "math"), ijk, "default.oct", cHelp,"")

  @classmethod
  def _setCalcBasis(cls):
    """Load this other calculator. Quaternions are redefined."""
    if Lib.isCalc("CA"):
      for i in cls.__CA_CHARS:
        if i not in cls.__allChars:
          cls.__allChars.append(i)
      cls.__useCA = True
    return O.SetQuaternions()[1]

  @classmethod
  def _validBasis(cls, value, full=False):
    """Used by Calc to recognise full basis forms o... and u...
       or e... and i... if CA is loaded."""
    if len(value) == 1:
      return 0
    if value[0] not in cls.__allChars:
      return 0
    isBasis = True
    isCA = value[0] in cls.__CA_CHARS
    for ch in value:
      if isBasis and ch in cls.__allChars:
        if isCA != (ch in cls.__CA_CHARS):
          return 0
        isBasis = False
      elif ch.isdigit() or ch in cls.__HEX_CHARS:
        isBasis = True
      else:
        return 0
    return 2 if isCA else 3

  @classmethod
  def _processStore(cls, state):
    """Convert the store array into O(...) python code. Convert to CA(...)
       for e/i basis if __useCA. If isMults1/2 set then double up since
       MULTS are higher priority then SIGNS. The state is a
       ParseState from Calculator.processTokens()."""
    kw = {}
    line = ""
    isCA = False
    signTyp = state.signVal
    firstCnt = 1 if state.isMults1 else -1
    lastCnt = len(state.store) -1 if state.isMults2 else -1
    for cnt,value in enumerate(state.store):
      val,key = value
      isCA = False
      if key and key[0] in cls.__CA_CHARS:
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
  exp = Lib.exp
  log = Lib.log
  try:
    import importlib
  except:
    importlib = None          # No changing of calculators

  # O Unit test cases for Calc with Tests[0] being init for each case
  # Can only test 2-D rotations until euler stuff is updated. TBD
  Tests = [\
    """d30=radians(30); d60=radians(60); d45=radians(45); d90=radians(90)
       e=Euler(pi/6,pi/4,pi/2); c=o1+2o2+3o12""",
    """# Test 1 Rotate via frameMatrix == versor half angle rotation.
       Rx=d60-o12; rx=(d60 -o12).versor()
       test = Rx.frameMatrix() *c.vector(); store = (rx.inverse()*c*rx).vector()
       Calculator.log(store == test, store)""",
    """# Test 2 Rotate via frameMatrix == versor.rotate(half angle)].
       Rx=d60-o12; rx=O.Versor(o12=d60)
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
    """# Test 5 Euler implicit rotation == other order, Rzyx==Rxy'z'' QFAIL TBD.
       e=Euler(pi/6)
       if Lib.isCalc("Q"):
         test = O.Q(Q.Euler(e, order=[1,2,3], implicit=True))
       else:
         test = O.Euler(e, order=[3,2,1])
         Lib.precision(1E-12)
       store = O.Euler(e, order=[1,2,3], implicit=True)
       Calculator.log(store == test, store)""",
    """# Test 6 Versor squared == exp(2*log(e)).
       test = O.Euler(e).pow(2); store = (O.Euler(e).log() *2).exp()
       Calculator.log(store == test, store)""",
    """# Test 7 Rotate via frameMatrix == versor.versorMatrix(half angle).
       if Lib.isCalc("Q"):
         test = (d45+i+j+k).frameMatrix()
       else:
         test = (d45+o1+o2+o12).frameMatrix()
       store = (d45+o1+o2+o12).versor().versorMatrix()
       Calculator.log(store == test, store)""",
    """# Test 8 Rotate via versor.versorMatrix() == versor.euler().matrix().
       r = d45 +o1 +o2 +o12; store = r.normalise().euler().matrix()
       if Lib.isCalc("Q"):
         test = (d45+i+j+k).normalise().versorMatrix()
       else:
         test = r.normalise().versorMatrix()
       Calculator.log(store == test, store)""",
       ]

  calc = Calculator(O, Tests)
  calc.processInput(sys.argv)
###############################################################################
