#!/usr/bin/env python3
#####################################################################
## File: CalCrypt.py - symmetric encryption - G.P.Wilmot (c) Sep 2026
#####################################################################
import os, sys, random, time, argparse

class CalCrypt():
  """Class to create quasi-calibration at level N=4..10 and encrypt
     with setCodes() and rotate() or decrypt using reverse() first.
     Calibrations map to Cayley-Dickson algebras and the Sharp(N)
     algebras in Cl(n), n=2**N-1, calculate them for all N>1."""

  def __init__(self, N, sevenForm=False, verbose=False):
    """Create the initial calibration structure for this level."""
    self.__checkType(N, int, "init", (3, 10))
    n = int(pow(2, N)) -1
    got = set([])
    cal = []
    for tri in self.__comb(n, 3, list(range(1, n +1))):
      pair1 = tuple(tri[0:2])
      pair2 = tuple(tri[1:3])
      pair3 = (tri[0], tri[2])
      if pair1 not in got and pair2 not in got and pair3 not in got:
        got.update((pair1, pair2, pair3))
        cal.append(tri)
    self.__level = N
    self.__dim = n
    self.__cal = list(list(_) for _ in cal) # cal3
    self.__verbose = verbose
    if sevenForm:
      if N > 6:
        dirname = os.path.dirname(__file__)
        basePath = os.path.join(dirname, "CalCrypt%d-7.dat" %N)
        if not os.path.isfile(basePath):
          sys.stderr.write("Error: File does not exist: %s\n" %basePath)
        else:
          self.__cal = self.read(basePath)
      else:
        self.__cal = self.genCal7()
    self.__msgLen = len(self.__cal)     # (n,2) //3 = n!/(n-2)!/2 //3 for cal3
    self.__sgns = [1] *(self.__msgLen)  # Key results

  def __str__(self):
    return str({"level":   self.__level,  "dimension": self.__dim,
                "msg.len": self.__msgLen, "hasForm7": len(self.__cal[0])==7})

  def genCal7(self):
    """Generate and return cal7 from __cal."""
    cal3 = self.__cal
    calDup = set(tuple(_) for _ in cal3)
    msgLen = (8**self.__level -7 *4**self.__level +14 *2**self.__level -8)//168
    out = []
    verbose = (self.__verbose and self.__level > 5)
    if verbose:
      sys.stdout.write("Cal7 level %d, lens %d->%d, Progress%%:" \
                   %(self.__level, len(cal3), msgLen))
      sys.stdout.flush()
      size = (len(cal3) +99) //100
      start = time.time()
      cnt = 0
    adding = True
    while adding:
      adding = False
      for iIdx,iVal in enumerate(cal3[:-2]):
        if verbose:
          cnt += 1
          if cnt % size == 0:
            sys.stdout.write(" %d(%0.2f')" %(cnt //size, (time.time() -start)/60))
            sys.stdout.flush()
        for jIdx in range(iIdx +1, len(cal3) -1):
          jVal = cal3[jIdx]
          for kIdx in range(jIdx +1, len(cal3)):
            kVal = cal3[kIdx]
            join = set(iVal)
            join.update(jVal)
            join.update(kVal)
            if len(join) == 7:
              addIt = True
              lJoin = sorted(join)
              for triIdx in ((0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,4,5)):
                tri = tuple(lJoin[x] for x in triIdx)
                if tri not in calDup:
                  addIt = False
                  break
              if addIt and lJoin not in out:
                out.append(lJoin)
                adding = True
                if len(out) == msgLen:
                  if verbose:
                    sys.stdout.write("\n")
                  return out
    raise Exception("Programming error")

  def rotate(self, array):
    """Pass pairs of dimensions to rotate the code."""
    self.__checkType(array, (list, tuple), "rotate", length=(1, 0))
    list(self.__checkType(_, (list, tuple), "rotate", length=2) for _ in array)
    list(self.__checkType(_[0], int, "rotate", (1, self.__dim)) for _ in array)
    list(self.__checkType(_[1], int, "rotate", (1, self.__dim)) for _ in array)
    self.__rotate(array)
  def __rotate(self, array):
    """Rotate  3 or 7-form __cal with all pairs in array setting __sgns."""
    for pair in array:
      spair = sorted(pair)
      for pos,tri in enumerate(self.__cal):    # Rotate calibration structure
        sgn = self.__sgns[pos]
        form = tri[:]
        for idx,val in enumerate(pair):
          if val in tri:
            offs = tri.index(val)
            form[offs] = pair[1 -idx]
            if idx:
              sgn = -sgn                   # Change signs for 2nd rot. index
        sgn = self.__sort(form, None, sgn)
        self.__cal[pos] = form
        self.__sgns[pos] = sgn
    self.__sort(self.__cal, self.__sgns)
  def __rotate1(self, pair, sortDims):
    """Test code. Rotate __cal as 3- or 7-form with 1 rotation, no sign."""
    spair = sorted(pair)
    for pos,tri in enumerate(self.__cal):    # Rotate calibration structure
      if tri[0] > spair[1]:
        break 
      elif tri[2] >= spair[0]:
        form = tri[:]
        fix = False
        for idx,val in enumerate(pair):
          if val in tri:
            fix = True
            offs = tri.index(val)
            form[offs] = pair[1 -idx]
        if fix:
          sortDims(form)
          self.__cal[pos] = form
    self.__sort(self.__cal)

  def reverse(self, array):
    """Return the array for rotate to decrypt instead of encrypt signs."""
    self.__checkType(array, (list, tuple), "reverse", length=(1, 0))
    list(self.__checkType(_, (list, tuple), "reverse", length=2) for _ in array)
    return list([_[1], _[0]] for _ in reversed(array))

  def setCodes(self, signs, cal=None):
    """Set signs and optionally the calibration structure."""
    self.__checkType(signs, (list, tuple), "setCode", length=self.__msgLen)
    list(self.__checkType(_, int, "setCode", (-1,1)) for _ in signs)
    if cal is not None:
      self.__checkType(cal, (list, tuple), "setCode", length=self.__msgLen)
      list(self.__checkType(_, (list,tuple), "setCode", 
                             length=len(self.__cal[0])) for _ in cal)
      n = self.__dim
      for idx in (0, 1, 2):
        list(self.__checkType(_[idx], int, "setCode", (1, n)) for _ in cal)
    self.__sgns = list(signs)
    if cal is not None:
      self.__setCode(cal)
  def __setCode(self, cal):
      self.__cal = cal[:]

  def getCodes(self):
    """Return signs and the calibration structure."""
    return (self.__sgns, self.__cal)

  def __sort(self, array, signs=None, sgn=1):
    """Sort the list changing parity for pair swaps optionally with __sgns."""
    more = True
    while more:              # Bubble sort - needed to not expose rotations used
      more = False
      for idx in range(1, len(array)):
        if array[idx -1] > array[idx]:
          tmp = array[idx -1]; array[idx -1] = array[idx]; array[idx] = tmp
          sgn = -sgn
          if signs:
            tmp = signs[idx -1]; signs[idx -1] = signs[idx]; signs[idx] = tmp
          more = True
    return sgn
  def __sort1(self, array):
    """Sort __cal ignoring __sgns."""
    array.sort()

  def comb(self, n, r, basis):
    """Yield the combinations of r in n elements for basis of length n."""
    self.__checkType(n, int, "comb", (1,0))
    self.__checkType(r, int, "comb", (0,n))
    self.__checkType(basis, (list, tuple), "comb", length=n)
    return self.__comb(n, r, basis)
  def __comb(self, n, r, basis):
    if r > n //2:
      rng = range(1, len(basis) +1)
      for elem in reversed(list(self.__perm(n, [1] *(n -r), 1))):
        yield tuple(basis[idx -1] for idx in rng if idx not in elem) 
    else:
      for elem in self.__perm(n, [1] *r, 1):
        yield tuple(basis[idx -1]  for idx in elem)

  def __perm(self, n, arr, offset):
    """Return permutation or combination arrays. The arr needs to be set to a
       list of one. Set offset to 0 to get permutations or 1 for combinations
       instead."""
    if offset < 0 or n < 0 or len(arr) > n:
      raise Exception("Invalid parameter for perm or comb: %s" %n)
    if len(arr) == 0:
      yield []
    else:
      for recuse in range(offset if offset else 1, n +1):
        if len(arr) > 1:
          for more in self.__perm(n, arr[1:], recuse if offset else 0):
            arr = [recuse] +more
            dup = False
            for idx,elem in enumerate(arr):
              if elem in arr[idx+1:]:
                dup = True
                break
            if not dup:
              yield arr
        else:
          yield [recuse]

  def __checkType(self, arg, typ, method, size=[], length=None):
    """Raise exception if arg not the correct type in method for size."""
    if not isinstance(arg, typ):
      tmp = str(arg)
      if len(tmp) > 9:
        tmp = str(type(arg))
      raise Exception("Invalid parameter type (%s) for %s" %(tmp, method))
    if length:
      size = length
      arg = len(arg)
    if size:
      if isinstance(size, int):
          if size != arg:
            raise Exception("Invalid %s !=%d for %s" %(arg, size, method))
      elif isinstance(size, (list, tuple)):
        if len(size) == 2 and (arg < size[0] or \
              (size[1] > 0 and arg > size[1])):
          raise Exception("Invalid %s !in [%d,%s] for %s" %(arg, size[0],
                          size[1] if size[1] else "..", method))
      elif size:
        raise Exception("Invalid check %s parameters in %s" %(arg, method))

  def seedRand(self):
    """The system does this by default."""
    random.seed(time.time())

  def randRot(self, stage):
    """Generate random rotation pairs, stage=0: single for 7-form, =1: odd
       dmensions only, =2: even dimensions, try for all, drop overlaps."""
    self.__checkType(stage, int, "randRot", (0,2))
    if stage == 0:   # 7-form case
      size = self.__dim
      start = 1
      step = 1
    else:
      size = self.__msgLen *3 //2 +1
      start = 0 if stage == 1 else 2
      step = 2
    rots = []
    for pos in range(1, size):
      pair = [random.randrange(start, self.__dim, step),
              random.randrange(start, self.__dim, step)]
      if stage == 1: # Sender gets odd dimensions
        pair = [pair[0] +1, pair[1] +1]
      if pair[0] > pair[1]:
        tmp = pair[0]; pair[0] = pair[1]; pair[1] = tmp
      if pair not in rots and pair[0] != pair[1]:
        rots.append(pair)
    return rots

  def pack(self, msg):
    """Change ASCII msg into 7 bits -> sign +1/-1 ints."""
    self.__checkType(msg, str, "pack", length=(1,0))
    if len(msg) *7 > self.__msgLen:
      raise Exception("Message it too long by %d character(s)" \
                              %(len(msg) -self.__msgLen //7))
    signs = []
    for ch in msg:
      for bit in format(ord(ch), "07b"):
        signs.append(-1 if bit == "0" else 1)
    if len(signs) < self.__msgLen -7:
      for bit in format(0x25, "07b"): # Add control-d
        signs.append(-1 if bit == "0" else 1)
    for idx in range(len(signs), self.__msgLen):  # pad with random bits
      signs.append(1 if random.randrange(2) else -1)
    return signs

  def unpack(self, signs):
    """Change sign +1/-1 ints -> 7 bits for ASCII msg."""
    self.__checkType(signs, (list, tuple), "unpack", length=self.__msgLen)
    msg = ""
    for idx in range(0, len(signs) -6, 7):
      bits = ""
      for bit in signs[idx :idx +7]:
        bits += "1" if bit == 1 else "0"
      ch = chr(int(bits, 2))
      if ch == chr(0x25): # EOM control-d
        break
      msg += ch
    return msg

  def save(self, struct, filename):
    """Save a python list with some type formatting of the structure."""
    nl = ""
    try:
      with open(filename, 'w') as fp:
        if isinstance(struct, (list, tuple, set)):
          fp.write("[\\\n")
          for val in struct:
            if isinstance(val, (list, tuple, set)):
              fp.write("%s  %s,\n" %(nl, val))
              nl = ""
            elif isinstance(val, str):
              fp.write("  \"%s\",\n" %val)
              nl = ""
            else:
              fp.write("  %s," %val)
              nl = "\n"
          fp.write("]")
        elif isinstance(struct, str):
          fp.write("%s" %struct)
        else:
          fp.write("%s" %struct)
        fp.write("\n")
    except BaseException as e:
      sys.stderr.write('%s: %s\n' %(type(e).__name__, e))

  def _generateSpace(self, store, parts, selfGetStrFn, selfSetStrFn,
                     status=None, pos=0, timeout=0):
    """CalTest code. Return new rotated 3- or 7-form calibrations as strings."""
    allRots = list((x,y) for x in range(1, self.__dim) \
                         for y in range(x +1, self.__dim +1))
    out = set()
    sortDims = self.__sort if len(self.__cal[0])==7 else self.__sort1
    for offs,calStr in enumerate(store[parts[0] :parts[1]]):
      if status:
        status[pos] = offs
      cal = selfSetStrFn(calStr)
      for rot in allRots:
        self.__setCode(cal)
        self.__rotate1(rot, sortDims)
        newCal = selfGetStrFn()
        if newCal not in store:
          out.add(newCal)
      if timeout and timeout < time.time():
        break
    return out

  @staticmethod
  def read(filename):
    """Read a Python structure (generally saved by save())."""
    with open(filename) as fp:
      return eval(fp.read())

  def processArgs(self, args, sevenForm, stage, msg):
    """Create the transmittion or decrypted file and option key file."""
    if sevenForm and len(self.__cal[0])!=7:
      sys.stderr.write("Can only do single pass transfers\n")
      sevenForm = False
    if stage == 1:
      self.seedRand()
      self.setCodes(self.pack(msg))
      odd = 1 if sevenForm else 0
      found = args.keyExists and not sevenForm and os.path.isfile(base +"1.key")
      if found:
        rot1 = CalCrypt.read(base +"1.key")[0]
      else:
        rot1 = self.randRot(odd) +self.randRot(odd)
        self.save([rot1], base +"1.key")
        if sevenForm:
          sys.stdout.write("Saved key file: %s1.key\n" %base)
        else:
          sys.stdout.write("Transmit private key file: %s1.key\n" %base)
      self.rotate(rot1)
      nextStage = 2 if sevenForm else 5
      self.save([nextStage, level, nextBase, *self.getCodes()], base +"1.cry")
      sys.stdout.write("Transmit public file: %s1.cry\n" %base)
    elif stage == 2:
      self.seedRand()
      self.setCodes(signs, cal)
      rot2 = self.randRot(2) +self.randRot(2)
      self.rotate(rot2)
      self.save([rot2], base +"2.key")
      sys.stdout.write("Saved key file: %s2.key\n" %base)
      self.save([3, level, nextBase, *self.getCodes()], base +"2.cry")
      sys.stdout.write("Transmit public file: %s2.cry\n" %base)
    elif stage == 3:
      self.setCodes(signs, cal)
      rot1 = CalCrypt.read(base +"1.key")[0]
      self.rotate(self.reverse(rot1))
      self.save([4, level, nextBase, *self.getCodes()], base +"3.cry")
      sys.stdout.write("Transmit public file: %s3.cry\n" %base)
    elif stage in (4, 5):
      self.setCodes(signs, cal)
      rot1 = CalCrypt.read(base +"%d.key" %(1 if stage == 5 else 2))[0]
      self.rotate(self.reverse(rot1))
      msg = self.unpack(self.getCodes()[0])
      self.save(msg, base +".txt")
      sys.stdout.write("Message file: %s.txt\n" %base)


_CalCrypt = CalCrypt.__doc__  # Doco for calculators

################################################################################
if __name__ == '__main__':
  import traceback
  class HelpException(Exception):
    """Don't report this exception."""
    pass

  desc = """Each usage creates a file to transmit to the other user. First usage
     enters two or more arguments: level, basename and message or use -m <file>.
     Subsequent use enters the sent file. Level is 4..10 & 3-pass (-t) has no
     private keys and message lengths are: 2, 22, 199, 1687, ., ., .. chars.
     For single pass with the same key for both users the message lengths are:
     5, 22, 93, 381, 1542, 6205, 24893 chars."""
  mode = 0
  for inp in sys.argv[1:]:
    if inp[0] == '-':
      if inp[-1] in "md" or inp in ("--messageFile", "--dataPath"):
        mode -= 1
    else:
      mode += 1
  parser = argparse.ArgumentParser("CalCrypt.py", description=desc)
  if mode == 1:
    parser.add_argument("filename", type=str,
                help="Filename to decode or re-encode [Start with level]")
  elif mode > 1:
    parser.add_argument("level", type=int,
                help="Level = 4..10 determines the message length")
    parser.add_argument("baseName", type=str,
                help="Filename prefix for all output")
    parser.add_argument("message", type=str, nargs='*',
                help="Message words on the command line")
    parser.add_argument("-t", "--threePass", action='store_true',
                help="Start 3 pass messaging")
    parser.add_argument("-k", "--keyExists", action='store_true',
                help="Use an existing key file for single pass")
    parser.add_argument("-m", "--messageFile", type=str, default=None,
                help="Message file if not on the command line")
  parser.add_argument("-f", "--fullHelp", action='store_true',
                help="Print full documentation")
  parser.add_argument("-d", "--dataPath", type=str, default="",
                      help="Output file path (default none)")
  parser.add_argument("-v", "--verbose", action='store_true',
                      help="Enable traceback and logging")
  args = parser.parse_args()
  try:
    if args.fullHelp:
      help(CalCrypt)
      raise HelpException
    if args.dataPath and not os.path.isdir(args.dataPath):
      raise Exception("Data path %s does not exist" %args.dataPath)
    if mode == 1:
      path,filename = os.path.split(args.filename)
      if path or not args.dataPath:
        filename = args.filename
      else:
        filename = os.path.join(args.dataPath, args.filename)
      if not os.path.isfile(filename):
        raise Exception("Filename %s does not exist" %filename)
      stage, level, nextBase, signs, cal = CalCrypt.read(filename)
      if stage not in (2, 3, 4, 5):
        raise Exception("Corrupt transmitted file")
      sevenForm = (stage != 5)
      path,base = os.path.split(nextBase)
      nextBase = nextBase.replace("\\","\\\\")
      path = path if stage == 3 else args.dataPath
      base = (os.path.join(path, base) if path else base)
      msg = None
    elif mode > 1:
      stage = 1
      level = args.level
      if not isinstance(level, int) or level < 4 or level > 10:
        raise Exception("Message level must be in 4..10")
      base = (os.path.join(args.dataPath, args.baseName) \
              if args.dataPath else args.baseName)
      nextBase = base.replace("\\","\\\\")
      sevenForm = args.threePass
      if args.messageFile:
        if args.message:
          parser.print_help()
          raise Exception("Can't have both message and -m option")
        path,filename = os.path.split(args.messageFile)
        if path or not args.dataPath:
          filename = args.messageFile
        else:
          filename = os.path.join(args.dataPath, args.messageFile)
        if not os.path.isfile(filename):
          raise Exception("Message filename %s does not exist" %filename)
        with open(filename) as fp:
          msg = fp.read()
      else:
        msg = " ".join(args.message)
      if len(msg) == 0:
        parser.print_help()
        raise Exception("Message is empty")
    else:
      parser.print_help()
      raise HelpException

    crypt = CalCrypt(level, sevenForm, args.verbose)
    crypt.processArgs(args, sevenForm, stage, msg)

  except HelpException:
    pass
  except Exception as e:
    if args.verbose:
      traceback.print_exc()
    else:
      sys.stderr.write('%s: %s\n' %(type(e).__name__, e))
