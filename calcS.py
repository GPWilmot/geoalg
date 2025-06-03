#!/usr/bin/env python
################################################################################
## File: calcS.py needs calcR.py and is part of GeoAlg.
## Copyright (c) 2022, 2023 G.P.Wilmot
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
## CalcS is a commnd line calculator that converts names into symbols as 
## strings.
##
## Simple symbolic expandion calculator. Used to derive n-D rotation matricies
## for calcCA and octonians (sedenions, etc ) for calcO. This module defines
## the P class for building Cayley-Dickson pyramids and the S class for
## checking matrix rotations. See default.sym.
## Start with either calcS.py, python calcS.py or see ./calcR.py -h.
################################################################################
__version__ = "0.2"
import sys, math
import keyword
from calcLib import *

################################################################################
class P(list):
  """Class to develop Cayley-Dickson product."""
  _posSigs = []  # Change these indicies into Positive Signature
  def __init__(self, n, lhs, rhs=0):
    """Define lhs as binary tree (level n>=0) or multiply lhs * rhs (n<0)."""
    if n >= 0:
      if isinstance(lhs, P) or rhs:
        raise Exception("Create P with single name: %s" %lhs)
      if n == 0:
        super(P, self).__init__([0, lhs, 0])
      elif n == 1:
        super(P, self).__init__([1, S(lhs).L(1), S(lhs).R(1)])
      else:
        p = P(-n, S(lhs).L(n), S(lhs).R(n))
        super(P, self).__init__([n, p[1], p[2]])
    elif n == -1:
      n = 1
      if isinstance(lhs, P):
        n = lhs[0] +1
      super(P, self).__init__([n, lhs, rhs])
    else:
      n = -n
      pL = P(1 -n, S(lhs).L(n-1), S(lhs).R(n-1))
      pR = P(1 -n, S(rhs).L(n-1), S(rhs).R(n-1))
      super(P, self).__init__([n, pL, pR])

  def __str__(self):
    """Overload string output. Printing using array if defined."""
    lhs = str(self[1])
    rhs = str(self[2])
    if lhs.find('+') > 0 or lhs[1:].find('-') > 0:
      lhs = "(%s)" %lhs
    if rhs.find('+') > 0 or rhs[1:].find('-') > 0:
      rhs = "(%s)" %rhs
    if lhs == "0":
      if rhs == "0":
        out = lhs
      elif rhs == "1":
        out = "o%d" %self[0]
      elif rhs == "-1":
        out = "-o%d" %self[0]
      else:
        out = "%s*o%d" %(rhs, self[0])
    elif rhs == "0":
      out = lhs
    else:
      if rhs == "1":
        rhs = "+"
      elif rhs == "-1":
        rhs = "-"
      elif isinstance(rhs, Lib._basestr):
        if rhs[0] != "-":
          rhs = "+" +rhs
      elif rhs >= 0:
        rhs = "+" +str(rhs)
      out = "%s%s*o%d" %(lhs, rhs, self[0])
    return out
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    return "P(%d: %s, %s)" %(self[0],repr(self[1]), repr(self[2]))

  def __mul__(self, p):
    """Product n => (a,b)(c,d) = (ac-d*b, da+bc*) for conjugate level n-1."""
    Lib._checkType(p, P, "mul")
    #print(repr(self), "*", repr(p))
    n = self[0]
    if n != p[0]:
      raise Exception("Invalid P to multiply: %s * %s" %(self, p))
    if n == 0:
      return self[1] * p[1]
    if n == 1:
      if n in P._posSigs:
        return P(-1, self[1] *p[1] +p[2] *self[2], self[1]*p[2] +p[1] *self[2])
      return P(-1, self[1] *p[1] -p[2] *self[2], p[2] *self[1] +self[2] *p[1])
    if n in P._posSigs:
      return P(-1, self[1] *p[1] +p[2].__conj(n-1) *self[2],
                   p[2] *self[1] +self[2] *p[1].__conj(n-1))
    return P(-1, self[1] *p[1] -p[2].__conj(n-1) *self[2],
                 p[2] *self[1] +self[2] *p[1].__conj(n-1))
  def __add__(self, p):
    """Return new Product adding tree recursively."""
    return P(-1, self[1] +p[1], self[2] +p[2])
  def __sub__(self, p):
    """Return new Product subtracting tree recursively."""
    return P(-1, self[1] -p[1], self[2] -p[2])
  def __neg__(self):
    """Return new Product negating tree recursively."""
    return P(-1, -self[1], -self[2])
  def __conj(self, n):
    """Conjugate level n => (a,b)*n = (a*(n-1), -b) recursive."""
    if self[0] != n:
      raise Exception("Invalid P to conjugate at %d: %s" %(n, self))
    if n > 1:
      return P(-1, self[1].__conj(n -1), -self[2])
    return P(-1, self[1], -self[2])

  @staticmethod
  def Table(dim):
    """Table(dim)
       Generate Octonian/Senonian multiplication table for dimension dim."""
    n = int(pow(2,dim))
    out = Matrix.Diag([0] *(n -1))
    x = [0] *n
    y = [0] *n
    for idx1 in range(1, n):
      x[idx1] = 1
      for idx2 in range(1, n):
        y[idx2] = 1
        tmp = str(P(dim, x) * P(dim, y))
        out[idx1 -1][idx2 -1] = tmp.replace("*o", "")
        y[idx2] = 0
      x[idx1] = 0
    return out

  @staticmethod
  def Basis(dim):
    """Basis(dim)
       Generate Octonian/Senonian basis for dimension dim."""
    n = int(pow(2,dim))
    x = [0] *n
    basis = []
    for idx in range(1, n):
      x[idx] = 1
      tmp = str(P(dim, x))
      basis.append(tmp.replace("*o", ""))
      x[idx] = 0
    return basis

  @staticmethod
  def Base(dim, name, el=0):
    """Base(dim, name, el=0)
       Return P for basis product all basis as A?0+A?o#.... (or A?o_el) where
       ?=0/1...(dim times) with o#=basis n-index position if 1."""
    Lib._checkType(dim, int, "Base")
    Lib._checkType(el, int, "Base")
    bas = ["o0"] + P.Basis(dim)
    n = int(pow(2,dim))
    buf = [0] *n
    if el:
      find = "o%d" %abs(el)
      if find not in bas:
        raise Exception("Invalid index %d for dim" %el)
    minus = (el < 0)
    for idx,pos in enumerate(bas):
      if not el or pos == find:
        tmp = idx
        ss = ""
        cnt = n
        while cnt > 1:
          cnt //= 2
          ss = ("1" if tmp >= cnt else "0") +ss
          tmp %= cnt
        buf[idx] = S(name +ss, isMinus=minus)
      else:
        buf[idx] = S(0)
    return P(dim, buf)

  @staticmethod
  def Multiply(dim, alpha=0):
    """Multiply(dim, [alpha])
       Return product of A*B using P.Base else for alpha=1/2 return
       (A1+A2o1+...)*(a1+a2o1+...) or (A+Bo1+Co2+...)*(a+bo1+co2+...)."""
    Lib._checkType(dim, int, "Multiply")
    Lib._checkType(alpha, int, "Multiply")
    if dim > 5:
      alpha = 0
    elif dim > 4:
      alpha = 1
    if alpha == 0:
      a = P.Base(dim, "A")
      b = P.Base(dim, "B")
      return "%s * %s = %s" %(a, b, a *b)
    n = int(pow(2,dim))
    if alpha == 1:
      for idx in range(n//2):
        buf[0][idx*2] = S(chr(idx +65)+"1")
        buf[0][idx*2 +1] = S(chr(idx +65)+"2")
        buf[1][idx*2] = S(chr(idx +97)+"1")
        buf[1][idx*2 +1] = S(chr(idx +97)+"2")
    else:
      for idx in range(n):
        buf[0][idx] = S(chr(idx +65))
        buf[1][idx] = S(chr(idx +97))
    return "%s * %s = %s" %(P(dim, buf[0]), P(dim, buf[1]),
                            P(dim, buf[0]) *P(dim, buf[1]))

  @staticmethod
  def Mult(dim, a, b):
    """Mult(dim, a, b)
       Return basis product o_a * o_b as (a?)(b?)o#. See Multiply."""
    Lib._checkType(dim, int, "Mult")
    Lib._checkType(a, int, "Mult")
    Lib._checkType(b, int, "Mult")
    a = P.Base(dim, "A", a)
    b = P.Base(dim, "B", b)
    return "%s * %s = %s" %(a, b, a *b)

  @staticmethod
  def Conj(dim, a):
    """Conj(dim, a)
       Return conjugate of basis product o_a. See Multiply."""
    Lib._checkType(dim, int, "Conj")
    Lib._checkType(a, int, "Conj")
    a = P.Base(dim, "A", a)
    return a.__conj(dim)

################################################################################
class S():
  """Class to process symbols as existing names or text. If getLocals is set the
     symbols are checked in locals and globals otherwise the text is embedded
     as operations. This allows symbols to be included in Matricies and expanded
     later."""
  __calculator = None                    # Last calculator loaded
  __globWordList = []                    # Remember variables
  __keyWordList = []                     # Avoid processing modules
  __loadedCalcs = []                     # Notify any other calc loaded

  def __init__(self, symbol, getLocals=None, isMinus=False):
    """S(symbol, [isTxt])
       Create symbols as existing name or text. If getLocals is set then
       it is a copy of the locals function and can be called to expand the
       symbols else the text is kept inside S()."""
    self.__value = symbol
    self.__getLocals = getLocals
    self.__isMinus = isMinus
    if isinstance(symbol, S):
      self.__value = symbol.__value
      self.__getLocals = symbol.__getLocals
      self.__isMinus = symbol.__isMinus

  def __str__(self):
    """Overload string output."""
    tmp = self.__eval()
    sStr = str(self.__value if tmp is None else tmp)
    if self.__isMinus:
      if sStr.find('+') > 0 or sStr[1:].find('-') > 0:
        sStr = "(%s)" %self.__value
      return "-" +sStr
    return str(sStr)
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if (Lib._isVerbose()):
      tmp = self.__eval()
      sgn = "+"
      if self.__isMinus and tmp is None:
        sgn = "-"
      return "S(%s,%s%s)" %(self.__value, sgn, str(tmp) if tmp else "")
    return str(self)

  def __hash__(self):
    """Allow dictionary access for basis objects."""
    return hash(str(self))

  def __eq__(self, cf):
    """Return True if 2 Ss are the same name."""
    if isinstance(cf, S):
      return (self.__value == cf.__value and self.__isMinus == cf.__isMinus)
    return False

  def __add__(self, sa):
    """Add 2 Ss or a scalar with possible string output."""
    sTmp = self.__eval()
    if isinstance(sa, S):
      saTmp = sa.__eval()
      if sTmp == 0:
        return sa if saTmp is None else saTmp
      if saTmp == 0:
        return self if sTmp is None else sTmp
      if sTmp is None or saTmp is None:
        sStr = str(self.__value)
        if self.__isMinus:
          if sStr.find('+') > 0 or sStr[1:].find('-') > 0:
            sStr = "(%s)" %sStr
          if sStr[0] == "-":
            sStr = sStr[1:]
          else:
            sStr = "-" +sStr
        saStr = str(sa.__value)
        if sa.__isMinus:
          if saStr.find('+') > 0 or saStr[1:].find('-') > 0:
            saStr = "(%s)" %saStr
          elif saStr[0] == "-":
            return S("%s+%s" %(sStr, saStr[1:]))
          return S("%s-%s" %(sStr, saStr))
        return S("%s+%s" %(sStr, saStr))
      return sTmp + saTmp
    else:
      Lib._checkType(sa, (int, float), "add")
      if sTmp is None:
        if sa == 0:
          return self
        sStr = str(self.__value)
        if self.__isMinus:
          if sStr.find('+') > 0 or sStr[1:].find('-') > 0:
            sStr = "(%s)" %sStr
          sStr = "-" +sStr
        if sa < 0:
          return S("%s-%s" %(sStr, -sa))
        return S("%s+%s" %(sStr, sa))
      return sTmp + sa
  __radd__ = __add__

  def __sub__(self, sa):
    """Subtract 2 Ss or a scalar with S output."""
    if isinstance(sa, S):
      tmp = S(sa.__value, sa.__getLocals, not sa.__isMinus)
      return self.__add__(tmp)
    else:
      Lib._checkType(sa, (int, float), "sub")
      return self.__add__(-sa)
  def __rsub__(self, sa):
    """Subtract S from scalar with S output."""
    tmp = S(self.__value, self.__getLocals, not self.__isMinus)
    return tmp.__add__(sa)

  def __neg__(self):
    """Unitary operator for S with string output."""
    return S(self.__value, self.__getLocals, not self.__isMinus)
  def __pos__(self):
    """Unitary + operator for S with string output."""
    return self

  def __mul__(self, sa):
    """Multiplication of 2 Ss or self by scalar."""
    sTmp = self.__eval()
    if isinstance(sa, S):
      saTmp = sa.__eval()
      if sTmp == 0 or saTmp == 0:
        return 0
      if sTmp == 1:
        return sa if saTmp is None else saTmp
      if sTmp is None or saTmp is None:
        sStr = str(self.__value)
        if sStr.find('+') > 0 or sStr[1:].find('-') > 0:
          sStr = "(%s)" %self.__value
        saStr = str(sa.__value)
        if saStr.find('+') > 0 or saStr[1:].find('-') > 0:
          saStr = "(%s)" %sa.__value
        if self.__isMinus != sa.__isMinus:
          return S("%s*%s" %(sStr, saStr), isMinus=True)
        return S("%s*%s" %(sStr, saStr))
      return sTmp * saTmp
    else:
      Lib._checkType(sa, (int, float, list, tuple), "mul")
      if sTmp == 0 or sa == 0:
        return 0
      if sTmp is None:
        if sa == 1:
          return self
        sStr = str(self.__value)
        if sStr.find('+') > 0 or sStr[1:].find('-') > 0:
          sStr = "(%s)" %self.__value
        if self.__isMinus != (True if sa < 0 else False):
          if self.__isMinus:
            return S("%s*%s" %(sStr, sa), isMinus=True)
          return S("%s*%s" %(sStr, -sa), isMinus=True)
        return S("%s*%s" %(sStr, sa))
      return sTmp * sa

  def __rmul__(self, sa):
    """Left multiplication of scalar by S."""
    sTmp = self.__eval()
    Lib._checkType(sa, (int, float, list, tuple), "rmul")
    if sTmp == 0 or sa == 0:
      return 0
    if sTmp == 1:
      return sa if saTmp is None else saTmp
    if sTmp is None:
      if sa == 1:
        return self
      sStr = self.__value
      if self.__getLocals and (sStr.find('+') > 0 or sStr[1:].find('-') > 0):
        sStr = "(%s)" %self.__value
      if self.__isMinus != (True if sa < 0 else False):
        if self.__isMinus:
          return S("-%s*%s" %(sa, sStr))
        return S("-%s*%s" %(-sa, sStr))
      return S("%s*%s" %(sa, sStr))
    return sa * sTmp

  def __div__(self, sa):
    """Attempted division for 2 versors or self by scalar."""
    sTmp = self.___eval()
    if isinstance(sa, S):
      saTmp = sa.__eval()
      if sTmp is None or saTmp is None:
        sStr = self.__value if self.__getLocals else "(%s)" %self.__value
        saStr = sa.__value if sa.__getLocals else "(%s)" %sa.__value
        sStr = '-' +sStr if self.__isMinus else sStr
        if sa.__isMinus:
          return S("%s-%s" %(sStr, sa))
        return S("%s/%s" %(sStr, saStr))
      return sTmp /saTmp
    else:
      Lib._checkType(sa, (int, float), "div")
      if sTmp is None:
        sStr = self.__value if self.__getLocals else "(%s)" %self.__value
        return S("%s/%s" %(sStr, sa))
      return sTmp / sa
  __truediv__ = __div__
  __flordiv__ = __div__

  def __rdiv__(self, sa):
    """Division for number, sa, divided by a CA."""
    Lib._checkType(sa, (int, float), "rdiv")
    sTmp = self.__eval()
    if sTmp is None:
      sStr = self.__value if self.__getLocals else "(%s)" %self.__value
      return S("1/%s*%s" %(sStr, sa))
    return (1/sTmp) * sa
  __rtruediv__ = __rdiv__
  __rflordiv__ = __rdiv__

  def __eval(self):
    """Return exposed value if possible for setting else None."""
    val = None
    if not isinstance(self.__value, Lib._basestr):
      return -self.__value if self.__isMinus else self.__value
    if self.__getLocals:
      name = self.__value
      offs = name.find(".")
      if offs > 0:
        name = name[:offs]
      try:
        val = self.__getLocals[name]
      except:
        val = S.__calculator.getGlobalWord(name)
    if val is not None and self.__isMinus:
      val = -val
    return val

  def __set__(self, val):
    """Set internal value."""
    if isinstance(val, S):
      self.__value = val.__value
      self.__isMinus = val.__isMinus
    else:
      self.__value = val
      self.__isMinus = False

  def __get__(self):
    """Expose expanded value if it exists else return self."""
    return val()

  def val(self):
    """Expose expanded value if it exists else return self."""
    tmp = self.__eval()
    if tmp is None:
      return self
    return tmp

  def L(self, n):
    """Expose the P's L value."""
    tmp = self.__eval()
    if tmp is None:
      return S("%s.L(%d)" %(self.__value, n))
    l = int(pow(2, n -1))
    if len(tmp) < l *2:
      raise Exception("List for %s needs len(%d)" %(self.__value, l *2))
    return S(tmp[:l]) if l > 1 else tmp[0]

  def R(self, n):
    """Expose the P's R value."""
    tmp = self.__eval()
    if tmp is None:
      return S("%s.R(%d)" %(self.__value, n))
    l = int(pow(2, n -1))
    if len(tmp) < l *2:
      raise Exception("List for %s needs len(%d)" %(self.__value, l *2))
    return S(tmp[l:]) if l > 1 else tmp[1]

  @staticmethod
  def version():
    """version()
       Return the module version string."""
    return __version__

  @staticmethod
  def IsCalc(calc):
    """Check if named calculator has been loaded."""
    return (calc in S.__loadedCalcs)

  ###################################################
  ## Calc class help and basis processing methods  ##
  ###################################################
  __keyWordList = []
  __globWordList = []
  __calculator = None

  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for S."""
    help = \
      """Symbolic Algebra Calculator - Process symbols as strings currently."""
    return (("S"), ("S", "P", "math"), "", "default.sym", help, "")

  @staticmethod
  def _setWordLists(Calc):
    """Return list of globals and builtins from primary module as well as
       python keywords."""
    S.__calculator = Calc
    S.__globWordList, S.__keyWordList = Calc.getWordLists()
    S.__keyWordList.append('S')
    S.__keyWordList.extend(keyword.kwlist)  # Add all python keywords

  @classmethod
  def _setCalcBasis(cls, calcs, Calc):
    """Load this other calculator. None currently."""
    S.__loadedCalcs = calcs
    S._setWordLists(Calc)
    return ""

  @classmethod
  def _validBasis(cls, value):
    """Used by Calc to recognise symbols."""
    idx = value.find(".")
    if idx > 0:
      value = value[:idx]
    if value in S.__keyWordList or value in S.__globWordList \
           or S.__calculator.getGlobalWord(value) is not None:
      return False
    return True

  @classmethod
  def _processStore(cls, state):
    """Convert the store array into CA(...) or Q(...) python code.
       Convert to Q(...) if basis is i, j, k and __useQuat.
       The state is a ParseState from Calculator.processTokens()."""
    line = ""
    signTyp = state.signVal
    for vals in state.store:
      scale,val = vals
      isAttribute = False
      isText = False
      excess = ""
      if val:
        if scale in ("-1", "+1"):
          if scale == "-1" and not signTyp:
            signTyp = scale[0] 
          scale = ""
        else:
          if scale[0] == "-" and not signTyp:
            signTyp = scale[0]
          if len(scale) > 1:
            scale = scale[1:] + "*"
          else:
            scale = ""
        idx = val.find(".")
        if idx == 0:
          isAttribute = True
        elif idx > 0:
          isText = not state.startLine
          if isText:
            excess = val[idx:]
            val = val[:idx]
        else:
          isText = not state.startLine
      else:
        val = ""
        if scale[0] == "+" and not signTyp:
          scale = scale[1:]
        else:
          signTyp = scale[0] 
          scale = scale[1:]
      if isAttribute or state.lastName in ("for", "lambda"):
        line += "%s" %val
      elif isText:
        line += signTyp +"%sS('%s',locals()).val()%s" %(scale, val, excess)
      else:
        line += signTyp +"%s" %(scale +val)
      signTyp = "+"
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
    exec(code, globals())
    return None

################################################################################
if __name__ == '__main__':

  from math import *
  from calcR import Calculator

  # P & S Unit test cases for Calc with Tests[0] being init for each case
  # Work in progress TBD
  Tests = [\
    """""",
    """# Test 1 Check both P expansions.
       test = P.Mult(1,0,0); store = P.Multiply(1)
       Calculator.log(store == test, store)""",
    """# Test 2 Check P conjugation
       test = str(P.Conj(2,12)); store = \"-0*o1+(-0-A11*o1)*o2\"
       Calculator.log(store == test, store)""",
    ]

  calc = Calculator(S, Tests)
  S._setWordLists(Calculator)
  calc.processInput(sys.argv)
###############################################################################
