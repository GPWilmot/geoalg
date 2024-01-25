#!/usr/bin/env python3
################################################################################
## File: calcQ.py needs calcR.py and is part of GeoAlg.
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
## CalcQ is a command line calculator that converts basis numbers into
## Quaternions, Versors and Euler Angle rotations.
##
## Quaternions are of the form q = s + _v_ where _v_ is a 3-D vector
## with _v_*_v_ <= 0 and s is a scalar. A pure quaternion has s==0.
## A unit quaternion has _v_*_v_ = -1. A versor has norm = 1 which means
## q = cos(a) + _n_sin(a) * _n_ where _n_ in a unit. This is the rotation group
## O(3) and operates on pure quaternions or is converted to a frame rotation
## matrix to operate on vectors. In this case the pure quaternion is unit length
## and the scalar is the angle of rotation. This gives a rotation matrix with
## twice the rotation angle as the versor angle due to the general rotation
## group operation q' = r * q * r.inverse() where r is a versor.
## Euler angles converted directly to versors for initialisation or
## interpretation of quaternion axial rotations. This module contains classes:
## Q & Euler & calc contains Common & Matrix run help for more information.
## Can be included as a module or run as a command line calculator.
## Assumes calcR.py is in same directory & numpy is imported first, if required.
## Quaternion tests are included at the end of this file.
## Start with either calcQ.py, python calcQ.py or see calcR.py -h.
################################################################################
__version__ = "0.2"
import math
from calcCommon import *
try:
  import numpy as np
except:
  pass
try:
  import matplotlib.pyplot as plt
except:
  plt = None

################################################################################
class Q():
  """Class to perform quaternion rotations in different ways and different
     orders both explicit and implicit. Conversions to versors, unit vector
     quaternions, Euler angles, versor matricies and frame matricies exist.
     The rotation methods are preferable to matrix and r*q*r.inverse().
     Package math methods only work on the scalar part eg sin(q.scalar()).
     """
  __BASE_CHARS = ('i', 'j', 'k')         # Quaternion basis chars
  dumpRepr = False                       # Repr defaults to str

  ##############################################################################
  ## Class overwritten functionality methods
  ##############################################################################
  def __init__(self, w=0, x=0, y=0, z=0):
    """New quaterion Q(w,z,y,z) with defaults 0. w=scalar, x,y,x=i,j,k coef."""
    params = locals().copy()
    for key in params:
      if key != "self":
        Common._checkType(params[key], (int, float), "Q")
    self.w,self.x,self.y,self.z = (w, x, y, z)

  def __float__(self):
    return float(self.w)
  def __int__(self):
    return trunc(self.w)
  def __str__(self):
    """Overload standard string output. Print each member if non-zero as int
       or double taking resolution into account and add vector axis (i,j,k).
       Also ignore unit number."""
    out = ""
    sign = ""
    for i in ((self.w, ""), (self.x, 'i'), (self.y, 'j'), (self.z, 'k')):
      out += Common._resolutionDump(sign, i[0], i[1])
      if out:
        sign = " +"
    return out if out else "0"
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if Common._isVerbose() and Q.dumpRepr:
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
      return self.vectorLen() < precision and \
              abs(self.w -cf) < precision
    if not isinstance(cf, Q):
      return False
    return abs(self.w -cf.w) < precision \
       and abs(self.x -cf.x) < precision \
       and abs(self.y -cf.y) < precision \
       and abs(self.z -cf.z) < precision
  def __ne__(self, cf):
    """Not equal is not automatic. Need this."""
    return not self.__eq__(cf)
  def __lt__(self, cf):
    """Return True if all vector parts smaller or scalar smaller than cf."""
    if isinstance(cf, (int, float)):
      return self.w < cf.w
    return self.x < cf.x and self.y < cf.y and self.z < cf.z
  def __le__(self, cf):
    """Return True if all vector parts smaller or scalar smaller than cf."""
    if isinstance(cf, (int, float)):
      return self.w <= cf.w
    return self.x <= cf.x and self.y <= cf.y and self.z <= cf.z
  def __gt__(self, cf):
    """Return True if all vector parts greater or scalar greater than cf."""
    if isinstance(cf, (int, float)):
      return self.w > cf.w
    return self.x > cf.x and self.y > cf.y and self.z > cf.z
  def __ge__(self, cf):
    """Return True if all vector parts greater or scalar greater than cf."""
    if isinstance(cf, (int, float)):
      return self.w >= cf.w
    return self.x >= cf.x and self.y >= cf.y and self.z >= cf.z

  def __add__(self, q):
    """Add 2 quaternions or a scalar from w."""
    if isinstance(q, Q):
      return Q(self.w +q.w, self.x +q.x, self.y +q.y, self.z +q.z)
    if isinstance(q, Tensor):
      return q.__add__(self)
    Common._checkType(q, (int, float), "add")
    return Q(self.w +q, self.x, self.y, self.z)
  __radd__ = __add__
  def __sub__(self, q):
    """Subtract 2 quaternions or a scalar from w."""
    if isinstance(q, Q):
      return Q(self.w -q.w, self.x -q.x, self.y -q.y, self.z -q.z)
    if isinstance(q, Tensor):
      return q.__add__(-self)
    Common._checkType(q, (int, float), "sub")
    return Q(self.w -q, self.x, self.y, self.z)
  def __rsub__(self, w):
    """Subtract quaternion from a scalar."""
    return self.__neg__().__add__(w)
  def __neg__(self):
    """Unitary - operator for quaternion."""
    return Q(-self.w, -self.x, -self.y, -self.z)
  def __pos__(self):
    """Unitary + operator for quaternion."""
    return self
  def __abs__(self):
    """Unitary abs operator for quaternion."""
    return Q(abs(self.w), abs(self.x), abs(self.y), abs(self.z))
  abs = __abs__

  def __mul__(self, q):
    """Quaternion multiplication for 2 quaternions or self by scalar."""
    if isinstance(q, Q):
      return Q(self.w *q.w -self.x *q.x -self.y *q.y -self.z *q.z,
               self.w *q.x +self.x *q.w +self.y *q.z -self.z *q.y,
               self.w *q.y -self.x *q.z +self.y *q.w +self.z *q.x,
               self.w *q.z +self.x *q.y -self.y *q.x +self.z *q.w)
    if isinstance(q, Tensor):
      return q.__rmul__(self)
    Common._checkType(q, (int, float), "mul")
    return Q(self.w *q, self.x *q, self.y *q, self.z *q)
  __rmul__ = __mul__
  
  def __bool__(self):
    return self != 0
  __nonzero__ = __bool__

  def __div__(self, q):
    """Quaternion division for 2 quaternions or self by scalar."""
    if isinstance(q, Q):
      return self.__mul__(q.inverse())
    Common._checkType(q, (int, float), "div")
    if abs(q) < Common._getPrecision():
      raise Exception("Illegal divide by zero")
    if sys.version_info.major == 2 and isinstance(q, int): # Python v2 to v3
      q = float(q)
    return Q(self.w /q, self.x /q, self.y /q, self.z /q)
  __truediv__ = __div__
  __floordiv__ = __div__
  def __rdiv__(self, q):
    """Quaternion division for number, q, divided by a quaternion."""
    return self.inverse().__mul__(q)
  __rtruediv__ = __rdiv__
  __rfloordiv__ = __rdiv__
  def __mod__(self, q):
    """Modulo % operator for quaternion."""
    Common._checkType(q, (int, float), "mod")
    p = self //q
    return self -p *q
  __rmod__ = __mod__
  def __divmod__(self, q):
    """Modulo  nd div operator for quaternion."""
    Common._checkType(q, (int, float), "mod")
    p = self //q
    return (p, self -p *q)
  __rdivmod__ = __divmod__
  def __floordiv__(self, q):
    """Quaternion div (//) for 2 quaternions or by scalar."""
    if isinstance(q, Q):
      p = self /q
      p = Q(int(p.w), int(p.x), int(p.y), int(p.z))
    else:
      p = Q(int(self.w/q), int(self.x/q), int(self.y/q), int(self.z/q))
    return p
  def __rfloordiv__(self, q):
    """Quaternion div (//) for number, q, divided by a quaternion."""
    p = q /self
    return Q(int(p.w), int(p.x), int(p.y), int(p.z))

  def __rotation(self, r):
    """Versor rotation common method. For self=q=w+a then r'=qrq.inverse()
       with r=0+v then v'=v +wt +axt where t=2axv. [verified]. Force versor."""
    l1 = r.len()
    if abs(l1 -1.0) >= Common._getPrecision():
      if l1 < Common._getPrecision():
        return
      r.x /= l1
      r.y /= l1
      r.z /= l1
    x = self.x *(r.w*r.w +r.x*r.x -r.y*r.y -r.z*r.z) + \
        2 *self.y *(r.x*r.y -r.w*r.z) +2 *self.z *(r.w*r.y +r.x*r.z)
    y = self.y *(r.w*r.w -r.x*r.x +r.y*r.y -r.z*r.z) + \
        2 *self.x *(r.w*r.z +r.x*r.y) +2 *self.z *(r.y*r.z -r.w*r.x)
    z = self.z *(r.w*r.w -r.x*r.x -r.y*r.y +r.z*r.z) + \
        2 *self.x *(r.x*r.z -r.w*r.y) +2 *self.y *(r.w*r.x +r.y*r.z)
    self.x, self.y, self.z = (x, y, z)

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
    return abs(self.x) + abs(self.y) + abs(self.z) < Common._getPrecision()

  def isVersor(self):
    """isVersore()
       Return True if self is of form cos(a) +n*sin(a), |n|==1."""
    return abs(self.len() -1.0) < Common._getPrecision()

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

  def copy(self, w=None, x=None, y=None, z=None):
    """copy([w,x,y,z])
       Return new self with optional new w, x, y or z."""
    params = locals().copy()
    quat = Q(self.w, self.x, self.y, self.z)
    for key in params:
      if params[key] is not None and key != "self":
        Common._checkType(params[key], (int, float), "copy")
        setattr(quat, key, params[key])
    return quat

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
      precision = Common._getPrecision()
    else:
      Common._checkType(precision, float, "trim")
    w = 0 if abs(self.w) < precision else  self.w
    x = 0 if abs(self.x) < precision else  self.x
    y = 0 if abs(self.y) < precision else  self.y
    z = 0 if abs(self.z) < precision else  self.z
    return Q(w, x, y, z)

  def pure(self):
    """pure()
       Return the pure quaternion part of self."""
    return Q(0, self.x, self.y, self.z)

  def vector(self):
    """vector()
       Return the pure quaternion of self as a vector."""
    return Matrix(self.x, self.y, self.z)

  def grades(self, maxSize=0):
    """grades([maxSize])
       Return a list of count of set terms x, y and z with scalar first."""
    Common._checkType(maxSize, int, "grades")
    g = [1 if self.w else 0]
    g.append[0]
    for val in (self.x, self.y, self.z):
      if val:
        g[1] += 1
    return g
      
  def len(self):
    """len()
       Return the sqrt of the sum of the multiplication with the
       complex conjugate."""
    return math.sqrt(self.w*self.w +self.x*self.x +self.y*self.y \
                    +self.z*self.z)

  def vectorLen(self):
    """vectorLen()
       Return the length of the pure quaternion part of self."""
    return math.sqrt(self.x*self.x +self.y*self.y +self.z*self.z)

  def conjugate(self):
    """conjugate()
       Return conjugate of self with i, j, k parts negated."""
    return Q(self.w, -self.x, -self.y, -self.z)

  def inverse(self):
    """inverse()
       Return inverse of self as conjugate/len."""
    l2 = self.len()
    l2 *= l2
    if self.len() < Common._getPrecision():
      raise Exception("Illegal divide by zero")
    return Q(self.w /l2, -self.x /l2, -self.y /l2, -self.z /l2)

  def dot(self, q):
    """dot(q)
       Return dot product of self with quaternion q. See sym."""
    Common._checkType(q, Q, "dot product")
    return -self.x*q.x -self.y*q.y -self.z*q.z

  def cross(self, q):
    """cross(q)
       Return cross product of self with quaternion q. See asym."""
    Common._checkType(q, Q, "cross product")
    return Q(0, self.y *q.z -self.z *q.y,
             self.z *q.x -self.x *q.z,
             self.x *q.y -self.y *q.x)

  def sym(self, q):
    """sym(q)
       Return symmetric product of two Qs. Same as dot product for pure parts"""
    Common._checkType(q, Q, "sym")
    out = (self *q +q *self) *0.5 
    return out.w

  def asym(self, q):
    """asym(q)
       Return anti-symmetric product of two QAs. This is the wedge product."""
    Common._checkType(q, Q, "asym")
    return (self *q -q *self) *0.5 
 
  def associator(self, p, q):
    """associator(p,q)
       Return the associator [self,p,q] = (self * p) *q - self *(p * q),"""
    out = (self * p) *q - self *(p * q)
    out.__entered0 = self.__entered0
    return out
  assoc = associator

  def projects(self, q):
    """projects(q)
       Return (parallel, perpendicular) parts of vector q projected onto self
       interpreted as a plane a*b with parts (in,outside) the plane. If q is
       a*b, a != b then return parts (perpendicular, parallel) to plane of a &
       b. a.cross(b) is not needed as scalar part is ignored."""
    n1 = self.vectorLen()
    if n1 < Common._getPrecision():
      raise Exception("Invalid length for projects.")
    mul = self.pure()
    vect = q.pure()
    mul = mul *vect *mul /float(n1 *n1)
    return (vect +mul)/2.0, (vect -mul)/2.0

  def distance(self, q):
    """distance(q)
       Return the Geodesic norm which is the half angle subtended by
       the great arc of the S3 sphere d(p,q)=|log(p.inverse())*q|."""
    Common._checkType(q, Q, "distance")
    return (self.inverse() *q).log().len()

  def normalise(self, noError=False):
    """normalise([noError])
       Return normalised self - reduces versor error accumulation.
       Raise an error on failure or return 0 if noError."""
    l1 = self.len()
    if l1 < Common._getPrecision():
      if noError:
        return Q(0)
      raise Exception("Illegal length for normalise()")
    return Q(self.w/l1, self.x/l1, self.y/l1, self.z/l1)

  def unit(self):
    """unit()
       Return self=frame with unit vectorLen(). See frame."""
    l1 = self.vectorLen()
    if l1 < Common._getPrecision():
      raise Exception("Frame has zero length pure part for unit().")
    return Q(self.w, self.x /l1, self.y /l1, self.z /l1)

  def frame(self):
    """frame()
       Return self=w+v as a frame=acos(w)*2 +v*len(w+v)/asin(w). See versor."""
    q = self.normalise()
    l1 = q.vectorLen()
    l0 = 0
    if l1 >= Common._getPrecision():
      l0 = self.len() /l1
    w = (q.w +1.0) %2.0 -1.0
    return Q(math.acos(w) *2, q.x *l0, q.y *l0, q.z *l0)

  def versor(self):
    """versor()
       Return self=frame as a versor no magnitude. See frame & normalise."""
    n1 = self.vectorLen()
    if n1 < Common._getPrecision():
      return Q(1)
    sw,cw = Common._sincos(self.w /2.0)
    sw /= n1
    return Q(cw, sw *self.x, sw *self.y, sw *self.z)

  def rotate(self, q):
    """rotate(q)
       Return q rotated by self as versor. Short for self*q*self.inverse()."""
    Common._checkType(q, Q, "rotate")
    q = q.copy()
    q.__rotation(self)
    return q

  def rotation(self, rot):
    """rotation(rot)
       Rotate self inplace by rot converting rot to versor first. See rotate."""
    Common._checkType(rot, Q, "rotation")
    self.__rotation(rot)

  def pow(self, exp):
    """pow(exp)
       For self=q=W+n, W=cos(a) then pow(q,x)=pow(|q|,x)(cos(xa) +sin(xa)n), n unit."""
    Common._checkType(exp, (int, float), "pow")
    n2 = self.x*self.x +self.y*self.y +self.z*self.z
    l1 = math.sqrt(n2 +self.w *self.w)
    w = pow(l1, exp)
    if n2 < Common._getPrecision():
      return Q(w)
    a = math.acos(self.w /l1)
    s,c = Common._sincos(a *exp)
    s *= w /math.sqrt(n2)
    return Q(w *c, self.x *s, self.y *s, self.z *s)
  __pow__ = pow

  def exp(self):
    """exp()
       For self=q=w+v, exp(q)=exp(w)(cos|v|+sin(|v|)v/|v|), inverse of log()."""
    n1 = self.vectorLen()
    if n1 < Common._getPrecision():
      return Q(self.w)
    s,c = Common._sincos(n1)
    exp = pow(math.e, self.w)
    s *= exp /n1
    return Q(c *exp, self.x *s, self.y *s, self.z *s)

  def log(self):
    """log()
       For self=q=w+v, log(self)=log|q|+acos(w/|q|)v/|v|, inverse of exp()."""
    n2 = self.x*self.x +self.y*self.y +self.z*self.z
    l1 = math.sqrt(self.w *self.w +n2)
    if n2 < Common._getPrecision():
      return Q(math.log(l1))
      if l1 == 0.0:
        raise Exception("Log(0) is undefined")
    s = math.acos(self.w /l1) /math.sqrt(n2)
    return Q(math.log(l1), self.x *s, self.y *s, self.z *s)

  def latLonAlt(self):
    """latLonAlt()
       Return geodetic lat(deg)/long(deg)/altitude(m) on WGS-84 for an ECEF
       quaternion vector (see LatLonAlt()). From fossen.biz/wiley/pdf/Ch2.pdf."""
    precision = Common._getPrecision()
    ee3 = 1 -Common._EARTH_ECCENT2
    p = math.sqrt(self.x *self.x +self.y *self.y)
    lat = math.atan2(self.z, p *ee3) # First approx.
    while True:
      lat0 = lat
      sLat,cLat = Common._sincos(lat)
      N = Common.EARTH_MAJOR_M /math.sqrt(cLat *cLat +sLat *sLat *ee3)
      if p >= precision:
        h = p /cLat -N
        lat = math.atan(self.z /p /(1 -Common._EARTH_ECCENT2 *N/(N +h)))
      elif lat >= 0.0:
        h = self.z -Common.EARTH_MINOR_M
        lat = math.pi *0.5
      else:
        h = self.z +Common.EARTH_MINOR_M
        lat = -math.pi *0.5
      if abs(lat -lat0) < precision:
        break
    return Matrix(math.degrees(lat),
                  math.degrees(math.atan2(self.y, self.x)), h)

  def euler(self, noError=False):
    """euler([noError])
       Versors can be converted to Euler Angles & back uniquely for default
       order. Euler parameters are q=cos(W/2) +a, a=n sin(W/2), n unit. So
       rotation angle defined by cosW=2a0a0-1=a0a0-a.a and n sinW=2ae0. XXX
       For r=roll(theta), p=pitch(chi), y=yaw(phi):
         w = cos((p+y)/2)cos(r/2)
           = cos(r/2)cos(p/2)cos(y/2) +sin(r/2)sin(p/2)sin(y/2)
         x = cos((p-y)/2)sin(r/2)
           = sin(r/2)cos(p/2)cos(y/2) -cos(r/2)sin(p/2)sin(y/2)
         y = sin((p-y)/2)sin(r/2)
           = cos(r/2)sin(p/2)cos(y/2) +sin(r/2)cos(p/2)sin(y/2)
         z = sin((p+y)/2)cos(r/2)
           = cos(r/2)cos(p/2)sin(y/2) -sin(r/2)sin(p/2)cos(y/2)
       Euler parameters in terms of euler angles for D=wy-xz, |D|<0.5:
         r = atan(2(wx+yz)/(1-2(x*x+y*y))) = atan(M21/M22)
         p = asin(2D) = asin(-M20)
         y = atan(2(wz+xy)/(1-2(y*y+z*z))) = atan(M01/M00)
         where M=Euler.matrix() cf Q.versorMatrix().
       Otherwise, for D~=+-0.5 (North/South Pole):
         y = 0, p = +-pi/2, r = -+2atan(x/w), respectively.
       For discriminant, D, see SE(3) transformations at
       https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.468.5407."""
    tmp = self
    if not self.isVersor():
      if not noError:
        raise Exception("Illegal versor norm for versorMatrix()")
      tmp = self.normalise()
    disc = tmp.w *tmp.y - tmp.x *tmp.z
    if abs(abs(disc) -0.5) < Common._getPrecision():
      sgn = 2.0 if disc < 0 else -2.0
      return Euler(sgn *math.atan2(tmp.x, tmp.w), -math.pi /sgn, 0.0)
    return Euler(math.atan2(2.0 * (tmp.z * tmp.y + tmp.w * tmp.x),
                      1.0 - 2.0 * (tmp.x * tmp.x + tmp.y * tmp.y)),
                 math.asin(2.0 * disc),
                 math.atan2(2.0 * (tmp.z* tmp.w + tmp.x * tmp.y),
                      1.0 - 2.0 * (tmp.y * tmp.y + tmp.z * tmp.z)))

  def versorMatrix(self, noError=False):
    """versorMatrix([noError])
       This is same as frameMatrix but for a versor with half the angle.
       Substitute 1-c = (w*w+b*b)-(w*w-b*b)=2b*b in frameMatrix
       where c=cosW=w*w-b*b, w=cos(W/2) and q=w+a*b, b=sin(W/2), a*a=-1.
       For tmp=w+v, return I*(2w*w-1) +2*v*vT +2*w*vX, v=tmp.vector()
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
       Opposite of Euler.Matrix for default order with Q.Euler. Set
       noError to norm the versor if not normed."""
    tmp = self
    if not self.isVersor():
      if not noError:
        raise Exception("Illegal versor norm for versorMatrix()")
      tmp = self.normalise()
    wx = tmp.w *tmp.x
    wy = tmp.w *tmp.y
    wz = tmp.w *tmp.z
    xx = tmp.x *tmp.x
    xy = tmp.x *tmp.y
    xz = tmp.x *tmp.z
    yy = tmp.y *tmp.y
    yz = tmp.y *tmp.z
    zz = tmp.z *tmp.z
    return Matrix( \
             (1.0 -2.0 *(yy +zz), 2.0 *(xy -wz), 2.0 *(xz +wy)),
             (2.0 *(xy +wz), 1.0 -2.0 *(xx +zz), 2.0 *(yz -wx)),
             (2.0 *(xz -wy), 2.0 *(yz +wx), 1.0 -2.0 *(xx +yy)))
     
  def frameMatrix(self):
    """frameMatrix()
       Return the rotation matrix for a quaternion frame. This is Q=W+a
       where angle W(rad) is the rotation angle around forced unit a.
       Any 3D frame = Euler Rotation W about axial a       
       which is the eigenvector of the rotation matrix.     _.- r'
       This gives the same rotation as r.rotation(q)      <__)_angle W
       = q.rotate(r) for q=Q.versor() ie angle W/2.       |   /
       For frameMatrix rotating r around a by W           |a /r
       r' = r(WW-a.a) +2a(a.r) +(rxa)sinW                 | /
          = cosW r +(1 -cosW)(r.a)a +sinW axr             |/
          = (r-ara)/2 +cosW(r+ara)/2 +sinW(ar-ra)/2
          = (w +v)r(w -v),      where v=a sin(W/2).
       Rodrigues vector is
       r' = r cosW +(axr)sinW +a(a.r)(1-cosW) from
       r  = r|| +r\/ = (r.a)a -ax(axr)  so
       r' = r +sinW (aX)r +(1-cosW)(aX)(aX)r where aX is cross-
       product matrix so (aX)r=axr and (aX)(aX)r=ax(axr).
       So return I +sinW aX +(1 -cosW)(aX)(aX) [verified].
       Opposite of FrameMatrix."""
    tmp = self.unit()
    s,c = Common._sincos(self.w)
    c1 = 1 -c
    x = tmp.x
    y = tmp.y
    z = tmp.z
    return Matrix( \
             (c +x *x *c1, x *y *c1 -s *z, z *x *c1 +s *y),
             (x *y *c1 +s *z, c +y *y *c1, z *y *c1 -s *x),
             (x *z *c1 -s *y, y *z *c1 +s *x, c +z *z *c1))

  ##############################################################################
  ## Other creators and source inverters
  ##############################################################################
  @staticmethod
  def Basis(numDims=3):
    """Basis([numDims])
       Return a list of (i,j,k) basis elements with value one."""
    return (Q(0,1), Q(0,0,1), Q(0,0,0,1))[:numDims]

  @staticmethod
  def Versor(w=None, x=None, y=None, z=None):
    """Versor([w,z,y,z])
       Quaternion as versor from frame, defaults 0, see versor()."""
    quat = Q().copy(w, x, y, z)
    return quat.versor()

  @staticmethod
  def Euler(roll=0, pitch=0, yaw=0, order=[], implicit=False):
    """Euler([roll, pitch, yaw, order, implicit])
       Creates a versor generated from Euler angles (row, pitch, yaw with
       defaults 0) or Euler class (roll only) and Rzyx ie extrinsic, this and
       order can change.  Default q' = (cz,0,0,sz) *(cy,0,sy,0) *(cx,sx,0,0).
       Order can contain names x, y, z, X, Y, Z or roll, pitch, yaw.
       If order=[3 ints(1-3) etc] then do this rotation order else optimized
       for default order=[1,2,3]=Rzyx. If explicit then order can't have
       repeats. If inplicit then subsequent rotations use the new rotated axis
       for rotation so default is Rz''y'x. Opposite of euler() for a versor."""
    if isinstance(roll, (int, float)):
      roll = Euler(roll, pitch, yaw)
    else:
      Common._checkType(roll, Euler, "Euler")
    Common._checkType(order, (list, tuple), "Euler")
    Common._checkType(implicit, bool, "Euler")
    sx,cx = Common._sincos(roll[0] * 0.5)
    sy,cy = Common._sincos(roll[1] * 0.5)
    sz,cz = Common._sincos(roll[2] * 0.5)
    if not order:
      if not implicit:
        return Q(cx * cy * cz + sx * sy * sz,
                 sx * cy * cz - cx * sy * sz,
                 cx * sy * cz + sx * cy * sz,
                 cx * cy * sz - sx * sy * cz)
      order = [1, 2, 3]
    rs = (sx, sy, sz)
    rc = (cx, cy, cz)
    if len(order) not in (1,2,3):
      raise Exception("Invalid order size")
    out = Q(1.0)
    implicitRot = Q(1.0)
    store = []
    names = ('xXroll', 'yYpitch', 'zZyaw')
    for key in order:
      if isinstance(key, Common._basestr):
        for i,name in enumerate(names):
          if name.find(key) >= 0:
            key = i +1
            break
      else:
        Common._checkType(key, (float, int), "Euler")
      if key in store or key not in (1, 2, 3):
        raise Exception("Invalid order index for Euler: %s" %key)
      args = [rc[key -1], 0, 0, 0]
      args[key] = rs[key -1]
      rot = Q(*args)
      if implicit:
        tmpRot = rot.copy()
        rot.rotation(implicitRot)
        implicitRot *= tmpRot
      else:
        store.append(key)
      out = rot *out
    return out

  @staticmethod
  def LatLon(lat, lng):
    """LatLon(lat, lng)
       Return Earth Centred, Fixed (ECEF) quaternion vector for geodetic
       WGS-84 lat(deg)/lng(deg). From Appendix C - Coorinate Transformations
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
    return Q(0, xMeridian *cLng, xMeridian *sLng, minor *sLat)

  @staticmethod
  def LatLonAlt(lat, lng, alt=0):
    """LatLonAlt(lat, lng, [alt])
       Return Earth Centred, Earth Fixed (ECEF) quaternion vector for geodetic
       WGS-84 lat(deg)/lng(deg)/altitude(m). From fossen.biz/wiley/pdf/Ch2.pdf.
       EarthPolar/EarthMajor = sqrt(1-e*e), e=eccentricity. alt default is 0."""
    Common._checkType(lat, (int, float), "LatLonAlt")
    Common._checkType(lng, (int, float), "LatLonAlt")
    Common._checkType(alt, (int, float), "LatLonAlt")
    sLat,cLat = Common._sincos(math.radians(lat))
    sLng,cLng = Common._sincos(math.radians(lng))
    ee3 = 1 -Common._EARTH_ECCENT2
    N = Common.EARTH_MAJOR_M /math.sqrt(cLat *cLat +sLat *sLat *ee3)
    return Q(0, (N +alt) *cLat *cLng, (N +alt) *cLat *sLng,
                (N *ee3 +alt) *sLat)
  @staticmethod
  def NED(lat, lng):
    """NED(lat, lng)
       Lat/lng Earth Centred-Earth Fixed (ECEF) to North-East-Down (NED)
       frame. Return a versor to perform this rotation. The inverse changes
       from NED to ECEF."""
    Common._checkType(lat, (int, float), "NED")
    Common._checkType(lng, (int, float), "NED")
    sLat,cLat = Common._sincos(math.radians(-lat -90) *0.5)
    sLng,cLng = Common._sincos(math.radians(lng) *0.5)
    return Q(cLng *cLat, -sLng *sLat, cLng *sLat, sLng *cLat)

  @staticmethod
  def FrameMatrix(mat):
    """FrameMatrix(mat)
       Return the quaternion for a frame matrix ie opposite of frameMatrix()
       for a unit vector except than angles outside 90 deg are disallowed.
       tr(mat) = 2cosW +1. If W=0 R=Id. If W=pi R=?."""
    Common._checkType(mat, (Matrix, Tensor), "FrameMatrix")
    if mat.shape()[0] != 3 or mat.shape()[1] != 3:
      raise Exception("Invalid FrameMatrix Matrix size")
    tr = mat.get(0,0) +mat.get(1,1) +mat.get(2,2)
    arg = (tr -1) *0.5
    if abs(arg) > 1:
      raise Exception("Rotation outside 90 deg is ambiguous")
    W = math.acos(arg)
    if W == 0:
      return Q(1)
    x = (mat.get(2,1) -mat.get(1,2))
    y = (mat.get(0,2) -mat.get(2,0))
    z = (mat.get(1,0) -mat.get(0,1))
    a = math.sqrt(x*x +y*y +z*z)
    return Q(W, x/a, y/a, z/a)

  @staticmethod
  def Quad(a ,b, c, val=None, s=5):
     """Quad(a ,b ,c, [val, s])
        Plot a quadratic a*x*x +b*x +c and it's rotated dual if matplotlib
        exists using s as the scale. Report the roots and dual function and
        return the apex values. Alternately, if val is set just return (x,y).
        Uses complex numbers Q(0,1)."""
     def _fnText(*args):
       """Replace Quad(%s,%s,%s) %(a,b,c) with text."""
       fnTxt = ""
       for x in zip(args, ("x^2", "x", "")):
         if x[0] != 0:
           fnTxt += "-" if x[0] < 0 else ("+" if fnTxt else "")
           fnTxt += (resolForm %abs(x[0]) if abs(x[0]) != 1 or not x[1] else "") +x[1]
       return "$%s$" %fnTxt

     if val is not None:
       return (val, a*val*val +b*val +c)
     d = b*b -4*a*c
     r = math.sqrt(d) if d>=0 else Q(0,math.sqrt(-d))
     val = 0
     if a != 0.0:
       r1 = (-b +r)/2.0/a
       r2 = (-b -r)/2.0/a
       val = r1 +(r2 -r1)/2.0 if d >= 0 else r1.scalar()
       sys.stderr.write("Roots: (%s, %s)\n" %(r1, r2))
       sys.stderr.write("Dual: Q.quad(%d, %d, %d)\n" %(-a, -b, c -b*b/a/2.0))
     if plt:
       _fs = 20.0
       xMax = s if s > val *val *2 else val *val *2
       yScale = 1 /xMax
       xScale = math.sqrt(yScale /abs(a)) if a!= 0.0 else 1
       xMax0, xMax1 = int(-xMax +val /xScale), int(xMax +val /xScale +1)
       yScale = abs(a *xScale *xScale +b *xScale)
       xAxis = list(x *xScale for x in range(xMax0, xMax1))
       yAxis = list(y *yScale *10 +c for y in (-xMax, xMax))
       plt.figure(Common.nextFigure())
       plt.plot(xAxis, (0,)*(xMax1 -xMax0))   # default blue
       plt.plot((0,)*2, yAxis, color="C0")    # first colour
       loc = "offset points"
       plt.arrow(0, yAxis[-1]-0.1, 0, 0.1, head_width=.4*xScale, head_length=.6,
                length_includes_head=True, head_starts_at_zero=True, color='C0')
       plt.arrow(xAxis[-1]-0.1, 0, 0.1, 0, head_width=.5, head_length=.4*xScale,
                length_includes_head=True, head_starts_at_zero=True, color='C0')
       plt.annotate('$y$', (0, yAxis[-1]), textcoords=loc, xytext=(5,-5),
                    color='C0', fontsize=_fs)
       plt.annotate('$x$', (xAxis[-1], 0), textcoords=loc, xytext=(-5,5),
                    color='C1', fontsize=_fs)
       plt.annotate('$x \\sqrt{-1}$', (xAxis[-1], 0), textcoords=loc,
              xytext=(-26 -int(_fs), -17 -int(_fs/2.0)), color='C2', fontsize=_fs)
       resolution, resolForm, resolFloat = Common._getResolutions("%s")
       fnVals = list(a*x*x +b*x +c for x in xAxis)
       plt.plot(xAxis, fnVals)
       plt.annotate(_fnText(a,b,c), (xAxis[-1], fnVals[-1]), textcoords=loc,
                    xytext=(-10,-10), ha='right',color='C1', fontsize=_fs)
       if a != 0.0:
         duVals = list(-a*x*x -b*x +c -b*b/a/2.0 for x in xAxis)
         plt.plot(xAxis, duVals)
         du = "Quad(%s,%s,%s)" %(Common.getResolNum(-a), Common.getResolNum(-b),
                                 Common.getResolNum(c -b*b/a/2.0))
         du = _fnText(-a, -b, c -b*b/a/2.0)
         #du = "$-x^2 + 1$"
         plt.annotate(du, (xAxis[-1], duVals[-1]), textcoords=loc,
                      xytext=(-9,0), ha='right',color='C2', fontsize=_fs)
       plt.draw()
       plt.show(block=False)
     else:
       sys.stderr.write("PIP: Matlabplot not installed - no plot available\n")
     return (val, a*val*val +b*val +c)

  @staticmethod
  def IsCalc(calc):
    """Test loading of other calculators. Not used in this calculator."""
    return (calc == "Q")

  ###################################################
  ## Calc class help and basis processing methods  ##
  ###################################################
  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for Q."""
    calcHelp = """Quaternion Calculator - Process numbers with i, j, k appended.
          Euler angles, FrameMatrix and ECEF WGS-84 frames are supported."""
    return (("Q", "R"), ("Q", "math"),
         "default.quat", calcHelp, "Can also use quaternions with basis i,j,k.")

  @classmethod
  def _setCalcBasis(cls, calcs, dummy):
    """Load other calculator. Does nothing for quaternions."""
    return ""

  @classmethod
  def _validBasis(cls, value):
    """Used by Calc to recognise basis chars i, j and k."""
    return len(value) == 1 and value in cls.__BASE_CHARS

  @classmethod
  def _processStore(cls, state):
    """Convert the store array[4] into Q(w,x,y,z) python code. If isMults1/2
       set then double up since MULTS are higher priority then SIGNS.
       The state is a ParseState from Calculator.processTokens()."""
    q = ["0"]*4
    signTyp = state.signVal
    firstCnt = 1 if state.isMults1 else -1
    lastCnt = len(state.store) -1 if state.isMults2 else -1
    line = ""
    for cnt,value in enumerate(state.store):
      val,key = value

      # If basis already entered or single so double up or accum scalar
      isMult = (cnt in (firstCnt, lastCnt) and lastCnt != 0)
      idx = 0 if key is None else cls.__BASE_CHARS.index(key) +1
      if q[idx] != "0" or isMult:
        if idx == 0 and not isMult:  # Duplicate scalar
          val = "+(%s%s)" %(q[0], val)
        elif isMult and q[1:] == ['0']*3:  # Scalar only
          line += q[0]
          signTyp = '+'
          q[0] = "0"
        else:
          line += signTyp +"Q(%s)" %(",".join(q))
          signTyp = '+'
          q = ["0"] *4
      q[idx] = val

    # Dump the remainder
    if q[1:] == ['0']*3:  # Scalar only
      signTyp = q[0][0] if signTyp or q[0][0] == "-" else ""
      line += signTyp +q[0][1:]
    else:
      line += signTyp +"Q(%s)" %(",".join(q))
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

  # Quaternion Unit test cases for Calc with Tests[0] being init for each case
  Tests = [\
    """d30=radians(30); d60=radians(60); d45=radians(45); d90=radians(90)
       e=Euler(pi/6,pi/4,pi/2); q=i+2j+3k""",
    """# Test 1 Rotate via frameMatrix == versor half angle rotation.
       Rx=d60+i; rx=(d60 +i).versor()
       test = Rx.frameMatrix() *q.vector(); store = (rx*q*rx.inverse()).vector()
       Calculator.log(store == test, store)""",
    """# Test 2 Rotate via frameMatrix == versor.rotate(half angle)].
       Rx=d60+j; rx=Q.Versor(d60,0,1)
       test = Rx.frameMatrix() *q.vector(); store = (rx.rotate(q)).vector()
       Calculator.log(store == test, store)""",
    """# Test 3 Rotate versor rotate == rotation of copy.
       Rx=d60+k; rx=math.cos(d30) +k*math.sin(d30)
       test = Rx.frameMatrix() *q.vector(); store = (rx*q*rx.inverse()).vector()
       Calculator.log(store == test, store)""",
    """# Test 4 Euler default rotation == explicit ordered rotation.
       test = Q.Euler(pi/6,pi/4,pi/2); store = Q.Euler(e, order=[1,2,3])
       Calculator.log( store == test, store)""",
    """# Test 5 Euler implicit rotation==other order, Rzyx==Rxy'z'', see Tensor.
       test = Q.Euler(e, order=[3,2,1])
       store = Q.Euler(e, implicit=True)
       Calculator.log(store == test, store)""",
    """# Test 6 Quaternion squared == exp(2*log(q)).
       test = q.pow(2); store = (q.log() *2).exp()
       Common.precision(6E-15)
       Calculator.log(store == test, store)""",
    """# Test 7 Rotate via frameMatrix == versor.versorMatrix(half angle).
       test = (d45+i+j+k).frameMatrix()
       store = (d45+i+j+k).versor().versorMatrix()
       Calculator.log(store == test, store)""",
    """# Test 8 Rotate via versor.versorMatrix() == versor.euler().matrix().
       r= (d45+i+j+k); test= r.frameMatrix(); store= r.versor().euler().matrix()
       Calculator.log(store == test, store)""",
    """# Test 9 Euler Matrix is inverse of versorMatrix.
       test=Tensor(pi/6, pi/4, pi/2)
       store=Euler.Matrix(Q.Euler(*test).versorMatrix())
       Calculator.log(store == Euler(*test), store)""",
    """# Test 10 Geodetic distance = acos(p.w *q.w +p.dot(q)).
       p = Q.Euler(e); q=(d45+q).versor(); store = p.distance(q)
       test = math.acos(p.w *q.w -p.dot(q))
       Calculator.log(store == test, store)""",
    """# Test 11 Length *2 == dot(self +self).
       store = (q *2).len(); test = math.sqrt(-(q +q).dot(q +q))
       Calculator.log(store == test, store)""",
    """# Test 12 Versor *3 /3 == versor.normalise
       Calculator.log(q/q.len() == q.normalise(), q.normalise())""",
    """# Test 13 Check Rodriges formula
       def para(a,r,w): return -a *a.dot(r)
       def perp(a,r,w): return r *math.cos(w) +a.cross(r) *math.sin(w)\\
               +a *a.dot(r) *math.cos(w)
       store = para(i,i+j,d30) +perp(i,i+j,d30); test = (d30+i).versor()
       Calculator.log(store == test.rotate(i+j), store)""",
    """# Test 14 Check lat-long conversion to ECEF xyz and back.
       lat=45; lng=45; test = Tensor(lat,lng)
       store = Q.LatLon(lat,lng); Common.precision(1E-8)
       Calculator.log(test == store.latLonAlt()[:2], store.latLonAlt())""",
    """# Test 15 Check lat-long-alt conversion to ECEF xyz and back.
       lat=45; lng=45; test = Tensor(lat,lng,0)
       store = Q.LatLonAlt(lat,lng,0)
       Calculator.log(store.latLonAlt() == test, test)""",
    """# Test 16 Check lat-long conversion from ECEF to NED. Ryz==Rz'y.
       lat=45; lng=45; store = Q.NED(lat,lng)
       test = Q.Euler(0,-radians(lat+90), radians(lng), order=[1,2,3])
       Calculator.log(store == test, test)""",
    """# Test 17 Check lat-long conversion from ECEF to NED. Ryz==Rz'y.
       lat=45; lng=45; test = Tensor.NED(lat,lng); store = []
       def basis(x): return Q.NED(lat,lng).rotate(x).vector()
       for x in (Q(0,1), Q(0,0,1), Q(0,0,0,1)): store.append(list(basis(x)))
       Calculator.log(test == store, test)""",
    """# Test 18 Check real case lat-long conversions.
       roll,pitch,yaw=(0.1, 0.1, 0.1)
       lat,lng = (-34.9285, 138.6007)
       conv = Tensor((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
       model = Q.Euler(Euler.Matrix(conv))
       egNED = Q.Euler(Euler(roll, pitch, yaw), order=[3,2,1])
       model2NED = (egNED * model).normalise()
       ECEF = (Q.NED(lat, lng) * model2NED).normalise()
       store = ECEF.versorMatrix()
       test = Tensor(\
         ( 0.552107290714106247, 0.63168203529742073, -0.544201567273411735),\
         (-0.341090423374471263,-0.424463728320773503,-0.838741835383363443),\
         (-0.760811975866937606, 0.648697425324424537,-0.0188888261919459843))
       Calculator.log(store == test, store)""",
    """# Test 19 Compare Tensor projection and Q.projects.
       def Ptest(a, b, x):
         G,P,N = Tensor.Rotations(a.unit().vector(), b.unit().vector())
         p = (a * b).projects(x); x0 = P *x.vector()
         return [p[0].vector(), p[1].vector().trim()] == [x0, x.vector()-x0]
       d2 = Ptest(i, j, i+2j);  d3 = Ptest(i, j+2k, i+j+k)
       Calculator.log(d2 and d3, (d2, d3))""",
    """# Test 20 Compare Tensor.Rotations with Q.versor.rotate.
       a,b = Q(0,1), Q(0,0,1,2)/sqrt(5)
       G,P,N = Tensor.Rotations(a.vector(), b.vector())
       rot_30 = (pi/6 +a*b).unit().frameMatrix()
       rot_60 = (pi/3 +a*b).frameMatrix()       # frameMatrix converts to versor
       rot_90 = (pi/2 +a.cross(b+Q(0,1))).frameMatrix()     # works for a.b != 0
       Rot_6 = N +P *cos(pi/6) +G *sin(pi/6)
       Rot_3 = N +P *cos(pi/3) +G *sin(pi/3)
       Rot_2 = N +P *cos(pi/2) +G *sin(pi/2)
       r30,r60,r90 = (rot_30==Rot_6, rot_60==Rot_3, rot_90==Rot_2)
       Calculator.log(r30 and r60 and r90, (r30, r60, r90))"""]

  calc = Calculator(Q, Tests)
  calc.processInput(sys.argv)
###############################################################################
