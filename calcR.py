#!/usr/bin/env python
###############################################################################
## File: calcR.py needs calcLib.py and is part of GeoAlg.
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
## This module contains classes: Lexer, ParseState, LibTest and Calculator
## used by all basis number calculators. Run help for more info.
## Start with either calcR.py, python calcR.py or see ./calcR.py -h.
###############################################################################
__version__ = "0.6"
import sys, math, os
import platform, glob
import traceback
from calcLib import *
try:
  if platform.system() == "Mac":
    import gnureadline as readline
  else:
    import readline
except:
  readline = None           # No command line editing
try:                        # Need:
  import ply.lex as ply_lex # pip install ply or apt-get install python3-ply
except:                     # else use local copy (only need lex.py)
  try:                      # from www.dabeaz.com/ply or github.com/dabeaz/ply
    lexPath = os.path.dirname(__file__.replace(os.getcwd(), ""))
    if lexPath != os.sep:
      sys.path.append(lexPath)
    import lex as ply_lex   # Versions 3.0 - 3.10 only
  except:
    ply_lex = None          # Normal calculator - no basis numbers

###############################################################################
#  Calculator tokeniser, parser and processor.                                #
###############################################################################
class Lexer:
  """Tokenise python code separating numbers from names and all symbols.
     This is used by Calculator.__parseTokens to handle basis numbers.
     Basis numbers in strings are also converted to the basis class."""
  def __init__(self):
    self.__lex = ply_lex.lex(module=self)
    self.__linePos = 0
    self.__lineNum = 0
    self.__oldLineNum = 0
  tokens = ('NAME', 'NUMBER', 'EQUALS', 'QUOTES', "SIGNS", "MULTS", "BRACKS",
            'COMMENT', 'NEWLINE')

  # Regular expression rules for simple token types
  t_NAME   = r'\.?[a-zA-Z_]+[a-zA-Z_0-9]*'
  t_NUMBER = r'[0-9]*\.?[0-9]+(E(\+|-)?[0-9]+)?'
  t_EQUALS = r'=='
  t_QUOTES = r'"""'
  t_SIGNS  = r'(\+|-)'
  t_MULTS  = r'(\*|//|/)'
  t_BRACKS = r'(\(|\))'
  t_COMMENT = r'\#'
  literals = '~!@$%^&?[]{}|,"<>=:;\'\\. \t' 
  t_ignore = '\r'

  # Define a rule so we can track line numbers
  def t_NEWLINE(self, tok):
    R'\n+'
    tok.lexer.lineno += len(tok.value)
    self.__linePos = tok.lexpos
    self.__lineNum = tok.lexer.lineno
    return tok

  def t_error(self, tok):
    """Report errors and continue."""
    sys.stderr.write("Illegal character: %s at line %d\n"
                     %(tok.value[:10], self.__lineNum -self.__oldLineNum))
    tok.lexer.skip(1)

  def reset(self, line):
    """Reset the input to restart tokenising."""
    self.__lex.input(line)
    self.__oldLineNum = self.__lineNum -1

  def process(self):
    """Return lexer.lex to iterate through tokens."""
    return self.__lex

################################################################################
class ParseState:
  """Basis number state for __parseTokens() and _processStore()."""
  def __init__(self):
    self.token = None
    self.lastTyp = ""
    self.startLine = True
    self.extendLine = False
    self.reset()
  def reset(self):
    self.store = []        # pairs of (value, basis)
    self.signVal = ""      # only if lastTyp == SIGNS
    self.aftFill = ""
    self.lastName = ""     # only for calcS variables
    self.lastBasis = 0
    self.isMults1 = False
    self.isMults2 = False
    self.multVal = ""      # For Python2 turn / into /float
  def __repr__(self):
    mul1 = self.multVal if self.isMults1 else ""
    mul2 = self.multVal if self.isMults2 else ""
    return "Basis:%s %sState:%s%s%s" %(self.lastBasis, self.signVal,
                                       mul1, self.store, mul2)

###############################################################################
class LibTest():
  """Class to provide common test processing and logging code."""
  __inTests    = None          # Test strings for running
  __testIndent = 7             # Test string indents
  __testRng    = []            # Array of running test numbers
  __testCnt    = 0             # Counter for running tests
  __testPass   = False         # Status for running tests

  @staticmethod
  def log(test, store=None):
    """Lib equality status and logging for each test case."""
    if LibTest.__inTests:
      if not test:
        LibTest.__testPass = False
      tst = LibTest.__testRng[LibTest.__testCnt]
      LibTest.__testCnt += 1
      if store is None:
        sys.stdout.write("Test %d\n" %tst)
      else:
        sys.stdout.write("Test%d: %s\n" %(tst, "PASS" if test else "FAIL"))
        if Lib._isVerbose():
          sys.stdout.write(str(store) +'\n')

  @staticmethod
  def testCnt():
    """Return the number of tests available to run."""
    return len(LibTest.__inTests) -1 if LibTest.__inTests else 0

  @staticmethod
  def logResults():
    """Summarise the results of an all tests run."""
    if LibTest.__inTests:
      sys.stdout.write("All tests passed\n" if LibTest.__testPass \
                                       else "Some tests failed\n")

  @staticmethod
  def _initRunTests(tests, indent=None):
    """Setup to run internal tests."""
    if tests and len(tests) > 1 and ply_lex:
      LibTest.__inTests = tests
      if indent is not None:
        LibTest.__testIndent = indent

  @staticmethod
  def __initTestExec(nummbers, firstTest):
    """Initialise testing cases from Calculator."""
    if LibTest.__inTests:
      LibTest.__testPass = True
      if firstTest:
        LibTest.__testCnt = 0
        LibTest.__testRng = []
      LibTest.__testRng += nummbers

  @staticmethod
  def getTestLines(number, firstTest):
    """Return exec lines for __inTests assuming this has at least 2 fields.
       number is blank to run all tests or the str(test number). Expect 
       Lib.log to be included at the end of each test and runtest[0] is
       initialisation for all tests. Copies all to history except Lib.log
       and adds logResults to end if the all tests are run."""
    if not LibTest.__inTests:
      raise Exception("No tests available to run")
    Lib.precision(1E-15)  # Reset precision in case a test changed it
    if not number:
      first,last = (1, len(LibTest.__inTests))
      LibTest.__initTestExec(range(first,last), firstTest)
    elif number.isdigit() and int(number) in range(1,len(LibTest.__inTests)):
      first,last = (int(number), int(number) +1)
      LibTest.__initTestExec([first], firstTest)
    else:
      raise Exception("Test argument should be in range 1..%d" \
                               %LibTest.testCnt())
    code = ""
    for idx,lines in enumerate(LibTest.__inTests[first:last]):
      for pos,line in enumerate(lines.splitlines()):
        if line[0] != "#":
          break
      for block in ((lines,0,pos), (LibTest.__inTests[0],0,-1),
                    (lines,pos,-1)):
        pos = len(block[0]) if block[2] < 0 else block[2] 
        for line in block[0].splitlines()[block[1]:pos]:
          if line[:LibTest.__testIndent] == " " *LibTest.__testIndent:
            code += line[LibTest.__testIndent:] +'\n'
            if number and Lib._isVerbose():
              sys.stdout.write(line[LibTest.__testIndent:] +'\n')
            if line.find("Calculator.log") < 0:
              if readline:
                readline.add_history(line[LibTest.__testIndent:])
          else:
            code += line +'\n'
            if readline:
              readline.add_history(line)
            if number and Lib._isVerbose():
              sys.stdout.write(line +'\n')
      if Lib._isVerbose() and idx < last -first -1:
        code += 'sys.stdout.write("\\n")\n'
    if not number:
      code += "Calculator.logResults()\n"
    return code[:-1]

################################################################################
class Calculator:
  """Command line processor parsing basis numbers into python code.Help uses the
     doc strings and class variable documents strings (starting with one _)."""
  # Firstly, all words processed by parsing
  __USEFUL_WORDS  = ("quit", "exit", "help", "show", "clear", "save",
                     "load", "test", "version", "calc", "vars",
                     "verbose", "precision", "resolution", "free",
                     "date", "time")
  __USEFUL_LIBS   = ("verbose", "precision", "resolution",  "free",
                     "date", "time")
  __USEFUL_CMDS   = ("load", "test", "version")        # Words with evaluation
  __USEFUL_FILE   = ("show", "clear", "save", "load")  # Ordered filename words
  __PYTHON_WORDS  = ("print", "raise", "from", "with", # Exec not eval
                     "with", "global", "raise", "import",
                     "for", "while", "exec", "eval", "del")
  __PYTHON_FUNCS  = ("def",)                   # Exec & no expand fn
  __inCls = None                               # Current processor
  __firstCls = None                            # Initial __inCls
  __oldCls = {}                                # Previous __inCls's
  __classList = ["Lib", "Euler", "Matrix", "Tensor"]  # From calcLib
  __moduleList = []                            # Calc classes loaded
  __promptList = []                            # Subordinate modules
  __history = []                               # For saving and showing
  __lastCmd = ""                               # Add load & test at end
  __prompt = ""                                # Current calculator name
  __saveVar = None                             # Variable passed to save

  def __init__(self, clsType, tests=[], indent=None):
    """Singleton calculator for basis numbers."""
    modList, clsList, ijk, fDef, cHelp, eHelp  = clsType._getCalcDetails()
    LibTest._initRunTests(tests, indent)
    Calculator.__inCls = clsType              # Current calculator
    Calculator.__firstCls = clsType           # Initial __inCls
    Calculator.__oldCls[clsType.__name__] = clsType
    Calculator.__classList.extend(clsList)
    Calculator.__moduleList.append(modList[0])
    Calculator.__promptList.extend(modList[1:])
    Calculator.__prompt = modList[0]          # Current calculator name
    Calculator.__cHelp = cHelp                # Calculator help text
    Calculator.__eHelp = eHelp                # Load extra help text
    Calculator.__default = fDef               # Load file default value
    Calculator.__firstCls._processExec(False, ijk)
    self.__line = ""                          # Store to history if OK
    self.__lexer = Lexer() if ply_lex else None

  @staticmethod
  def replaceTests(tests):
    LibTest._initRunTests(tests)

  @staticmethod
  def version(*args):
    """Print the common version with optional versions for other modules."""
    vers = Lib.version()
    idx = 0
    while idx +1 < len(args):
      if __version__ != args[idx +1]:
        vers += "(%s: %s)" %(args[idx], args[idx +1])
      idx += 2
    sys.stdout.write(vers +'\n')

  @staticmethod
  def getWordLists():
    """Return list of globals and builtins from primary module and a list
       of internal words for symbolic processing."""
    code = "dir(__builtins__)"
    ans = Calculator.__firstCls._processExec(True, code)
    return ans, list(Calculator.__USEFUL_WORDS)

  @staticmethod
  def getGlobalWord(key):
    """Return value if key is in globals from primary module."""
    try:
      code = "globals()['%s']" %key
      return Calculator.__firstCls._processExec(True, code)
    except:
      return None

  @staticmethod
  def log(test, store=None):
    """Lib equality status and logging for each test case."""
    LibTest.log(test, store)

  @staticmethod
  def logResults():
    """Summarise the results of an all tests run."""
    LibTest.logResults()

  @staticmethod
  def test(numbers, firstTest):
    """Run test numbers or all tests."""
    name = Calculator.__firstCls.__name__
    name = "R" if name == "Real" else name
    code = ""
    for number in numbers.split(','):
      tst = number.strip() if number else ""
      Calculator.__lastCmd = "test(%s)" %tst
      if code:
        code += "\n"
      code += LibTest.getTestLines(tst, firstTest)
      firstTest = False
    for line in code.splitlines():
      Calculator.__history.append(line)
    return code

  class ExecError(Exception):
    """Don't report exceptions outside exec() if already reported inside.
       Force raising of this exception instead to catch it outside."""
    pass

  @staticmethod
  def fixFilename(filename):
    """Add home path & extention if not already there."""
    return Lib.fixFilename(filename, os.path.dirname(__file__),
                     os.path.splitext(Calculator.__default)[1])

  @staticmethod
  def quit(filename=None):
    """Exit the calculator."""
    sys.exit(1 if filename else 0)
  exit = quit

  @staticmethod
  def show(filename=None):
    """List history, filename contents or defailt file contents."""
    if filename:
      sys.stdout.write(Lib.readText(filename).replace('\\', "\\\\") +'\n')
    else:
      sys.stdout.write("\n".join(Calculator.__history).replace('\\', "\\\\") \
                        +'\n')
  @staticmethod
  def load(filename=None, noError=False):
    """Load filename or default file and add to history."""
    if not filename:
      filename = Calculator.__default
    if not os.path.splitext(filename)[1]:
      filename += os.path.splitext(Calculator.__default)[1]
    Lib._storeName(filename)
    code = Lib.readText(Calculator.fixFilename(filename))
    if not code:
      if not noError:
        raise Exception("File %s is empty" %filename)
    elif Lib._isVerbose() and readline:
      readline.read_history_file(Calculator.fixFilename(filename))
    if Lib._isVerbose():
      Calculator.__lastCmd = "load(%s)" %filename
    return code

  @staticmethod
  def save(filename=None, varname=None):
    """Save history or variable to filename or history to default file."""
    if not filename:
      filename = Calculator.__default
    if varname is not None:
      Lib._save(filename, varname, Calculator.__saveVar,
          os.path.dirname(__file__), os.path.splitext(Calculator.__default)[1])
    else:
      with open(Calculator.fixFilename(filename), "a") as fp:
        for line in Calculator.__history:
          if line.strip()[:4] not in Calculator.__USEFUL_WORDS:
            fp.write(line +'\n')
    sys.stdout.write("File saved\n")

  @staticmethod
  def clear(filename=None):
    """Clear history or named or default file."""
    if filename:
      os.remove(filename)
    else:
      Calculator.__history = []

  @staticmethod
  def vars(lVars, gVars):
    """List local and global variables to line length 80."""
    allV,allF = set(),set()
    for name,var in lVars.items():
      if isinstance(var, (int, float, list, tuple, Lib._basestr)):
        if name[0] != "_":
          allV.add(name)
    for name,var in lVars.items():
      if str(type(var)) == "<type 'function'>":
        allF.add(name +"()")
    for name,var in gVars.items():
      if isinstance(var, (int, float, list, tuple, Lib._basestr)):
        if name[0] != "_":
          allV.add(name)
    for name,var in gVars.items():
      if str(type(var)) == "<type 'function'>":
        allF.add(name +"()")
    size = 0
    for name in allV:
      size = max(size, len(name))
    for name in allF:
      size = max(size, len(name))
    width = 80 // (size +2) +1
    fmt = "%%%ds" %(size +2)
    for pos in range(0, len(allV) +len(allF), width):
      for name in (sorted(allV) +sorted(allF))[pos:pos +width]:
        sys.stdout.write(fmt %name)
      sys.stdout.write('\n')

  @staticmethod
  def calc(calc=None):
    """Load a different calculator or list them."""
    names = []
    path = os.path.dirname(__file__) +"/calc*.py"
    for name in (os.path.basename(fName) for fName in glob.glob(path)):
      name = name[4:-3]
      if name and len(name) < 3:
        names.append(name)
    calcs = [None] if calc is None else calc.split(',')
    newCls = None
    ijkMsg = ""
    for calc in calcs:
      if calc is None:
        sys.stdout.write("Calculator%s: %s\n" %("s" if len(names) > 1 else "",
                                        ", ".join(names)))
      elif calc in names:
        mod = "calc%s" %calc
        clsName = "Real" if calc == "R" else calc
        if clsName in Calculator.__oldCls:
          newCls = Calculator.__oldCls[clsName]
        else:
          code = 'if importlib:\n'
          code += '  pkg = importlib.import_module("calc%s")\n' %calc
          code += '  globals()["%s"] = getattr(pkg, "%s")\n' %(clsName, clsName)
          Calculator.__firstCls._processExec(False, code)
          code = 'globals()["%s"] if importlib else None' %clsName
          newCls = Calculator.__firstCls._processExec(True, code)
          if newCls is None:
            raise Exception("No importlib: run %s.py from the command line"%mod)
        modList, clsList, ijk, fDef, cHelp, eHelp = newCls._getCalcDetails()
        Calculator.__firstCls._processExec(False, ijk)
        msg = newCls._setCalcBasis(Calculator.__moduleList, Calculator)
        if msg:
          ijkMsg = msg
        if calc not in Calculator.__moduleList: # Not already loaded
          Calculator.__oldCls[clsName] = newCls
          Calculator.__cHelp = cHelp
          Calculator.__eHelp = eHelp
          for mod in clsList:
            if mod not in Calculator.__classList:
              Calculator.__classList.append(mod)
          Calculator.__moduleList.append(modList[0])
          for mod in modList[1:]:
            if mod not in Calculator.__promptList:
              Calculator.__promptList.append(mod)
          Calculator.__Tests = None
          if calc not in Calculator.__promptList:      # Stick to biggest calc
            Calculator.__inCls = newCls 
            Calculator.__prompt = modList[0]
            Calculator.__default = fDef
      else:
        raise Exception("No such calculator: calc%s" %calc)
    if newCls:
      for cls in Calculator.__oldCls.values():
        cls._setCalcBasis(Calculator.__moduleList, Calculator)
      msg = "Variables i,j,k reset to " if ijkMsg else ""
      if len(Calculator.__moduleList) > 1:
        msg = "Enabled " +", ".join(Calculator.__moduleList)
        msg += " and i,j,k reset to " if ijkMsg else ""
      if msg + ijkMsg:
        sys.stdout.write(msg +ijkMsg +'\n')

  @staticmethod
  def help(cls=None, obj=None, path=None):
    """Print introduction, obj list or obj's documentation."""
    if not cls:
      ext = os.path.splitext(Calculator.__default)[1]
      tmp = "history%s default%s or named file"
      opt = ""
      if not readline and sys.platform != "win32":
        opt += "PIP: readline not installed - no command line history\n"
      if not ply_lex:
        opt += "PIP: ply not installed - no parsing of basis numbers\n"
      test = "" if LibTest.testCnt()==0 else "%9s test or test(1..%d,..)%s\n"\
             %("", LibTest.testCnt(), " - run all tests or just some")
      extra = ""
      libWords = ", ".join(Calculator.__USEFUL_LIBS)
      libWords += " "*(21 -len(libWords))
      for more in Calculator.__oldCls.values():
        if more and more != Calculator.__inCls:
          if Calculator.__eHelp:
            extra += '           %s\n' %Calculator.__eHelp
      sys.stdout.write('%s\n' %Calculator.__cHelp +extra \
          +'Commands: help or help(%s)\n'%"|".join(Calculator.__classList) \
          +'          calc or calc(<calcs>) - list or add calculators\n'  \
          +'          show or show(<file>)  - display %s\n' %(tmp %(",", "()"))\
          +'          clear or clear<file>) - clear %s\n' %(tmp %(" or", ext))\
          +'          load or load(<files>) - load %s\n' %(tmp %(" from", ext))\
          +'          save or save(<file>)  - append %s\n' %(tmp %(" to", ext))\
          +'            or save(<file>,var) - save var to file\n'\
          +'          version, vars         - list all versions, vars/fns\n'\
          +test \
          +'          ' +libWords +' - See help(Lib.*)\n'\
          +'          quit or exit or control-d[z/CR] - exit[Windows]\n' \
          +opt)
      if path:
        fNames = list((os.path.basename(fName) for fName in glob.glob(path)))
        if len(fNames) == 0:
          sys.stdout.write('No files to load\n')
        else:
          avWidth = sum(map(len, fNames)) //len(fNames) +3
          form = 'Files to load:'
          totWidth = 80
          if sys.platform !="win32":
            try:
              disWidth = int(os.popen('stty size', 'r').read().split()[1])
              totWidth = max(len(form) +avWidth, disWidth) -1
            except:
              pass
          out = ""
          for nam in fNames:
            tmp = ((avWidth -len(out) %avWidth) if len(out) %avWidth else 0) +1
            if len(form) +len(out) +len(nam) +tmp > totWidth and out != "":
              sys.stdout.write(form +out +"\n")
              form = " "*len(form)
              out = ""
              tmp = 1
            out += " " *tmp +nam
          sys.stdout.write(form +out +"\n")
    elif not hasattr(cls, "__name__"):
      if isinstance(cls, dict):
        var = "_%s" %obj
        if var in cls and isinstance(cls[var], Lib._basestr):
          sys.stdout.write("   %s\n" %cls[var])
        elif obj in cls:
          if hasattr(cls[obj], "__call__"): # For functions
            if cls[obj].__doc__:
              sys.stdout.write("   %s\n" %cls[obj].__doc__)
            else:
              sys.stdout.write("   Function has no documentation\n")
          else:
            sys.stdout.write("   Variable of type: %s\n" %type(cls[obj]))
        else:
          sys.stdout.write("   Variable does not exist\n")
      else:
        raise Exception("Invalid help parameter")
    else:
      Calculator.__help(cls, obj)

  @staticmethod
  def __help(cls, more, include=[]):
    """Lib internal documentation function for any class."""
    if more:
      Lib._checkType(more, Lib._basestr, "help")
      if more == cls.__name__:
        more = "__init__"
      if hasattr(cls, str(more)):
        doc = "   %s\n" %getattr(cls, str(more)).__doc__
        sys.stdout.write(doc if doc else \
                         "No documentation for this method." +'\n')
      else:
        sys.stdout.write("Method does not exist.\n")
    elif not cls.__doc__:
      sys.stdout.write(cls.__name__ +" has no help\n")
    else:
      sys.stdout.write(cls.__name__ +'\n')
      sys.stdout.write("     " +cls.__doc__ +'\n')
      inherit = dir(float) if cls is Real else None
      if cls is Matrix:
        if "np" in globals() and "numpy" in sys.modules:
          inherit = dir(np)
        elif "numpy" in globals() and "numpy" in sys.modules:
          inherit = dir(numpy)
        else:
          inherit = dir(list)
      for name in dir(cls):
        doc = None
        if inherit and name in inherit:
          pass
        elif hasattr(getattr(cls, name), "__call__"): # For functions
          if cls.__name__ not in ("Lib", "Calculator"):
            include.append("__init__")
          if name[0] != '_' or name in include:
            doc = getattr(cls, name).__doc__
            if name == "__init__":
              name = cls.__name__
        elif name[0] != '_':                    # For variables
          if '_' +name in dir(cls):
            doc = getattr(cls, "_%s" %name)
        if doc:
          name0 = name
          sdoc = doc.split("\n")
          sname = sdoc[0].split()
          if doc[:len(name0)] == name0:
            l = sdoc[0].find(")")
            if l > 0:
              name = sdoc[0][:l+1]
          if len(sdoc) > 1:
            doc = "\n".join(sdoc[1:]).lstrip()
          if doc and doc.find('\n') >= 0:
            doc = "See help(%s.%s)" %(cls.__name__, name0)
          sys.stdout.write("%-20s - %s\n" %(name, doc if doc else "None"))

  def __parseUsefulWord(self, isAns, line, ansAssign, firstWord):
    """Change __parseTokens USEFUL_WORDS and USEFUL_CMDS into useful Lib
       methods. Process commands with optional argument. Argument is
       assumed to be a module if help, a filename with default for file commands
       and ignored for other commands. Quotes are removed if found. Return text
       for exec() or exception. Return "" if quitting. USEFUL_CMDS return the
       code to run using lex which is not nested so is added after expansion.
       The ansAssign argument is ignored."""
    doFirstLoad = False
    firstTest = True
    isAns = False    # Could return some values - TBD
    if firstWord and line.find(firstWord) >= 0:
      word = firstWord
      firstWord = ""
    else:
      word = ""
      raise Exception("Programming error")
      for uWord in self.__USEFUL_WORDS:
        if line.lstrip().find(uWord) == 0:
          word = uWord
          break
    pline = line.replace(word, "").strip()
    pos1 = pline.find("(")
    pos2 = pline.find(")")
    param = pline[pos1 +1:pos2] if pos1 < pos2 and pos2 > 0 else ""
    extra = pline[pos2 +1:].strip()                # Anything after word or )
    if word and extra:
      if line.find(word) == 0:
        raise Exception("Command arguments must be inside brackets")
      code = line
    elif not word or extra or pline[:pos1].strip():  # Ignore if not single cmd
      code = line
    else:  # Separate out parameters
      if param and len(param) > 1:   # Remove quotes
        if param[0] == '"' and param[-1] == '"' or \
           param[0] == "'" and param[-1] == "'":
          param = param[1:-1]
      param = param.strip()
      if word in self.__USEFUL_FILE:  # File cmds can use default filename
        extra = ""
        if word == "load" and param == "noError=True":
          param = ""
          doFirstLoad = True
        if word == "save":
          buf = param.split(',')
          if len(buf) > 1:
            if len(buf) > 2:
              raise Exception("Too many variables to save")
            param = buf[0].strip()
            extra = ",'%s'" %buf[1].strip()
            Calculator.__saveVar = self.__processExec((True, buf[1]))
        if param:
          pline = "('%s'%s)" %(param, extra)
      elif word == "calc" and param:
        pline = "('%s')" %param.upper()
      elif word == "vars":
        if pline:
          raise Exception("Command vars has no parameters")
        pline = "(locals(), globals())"
      elif word == "help":
        if param:
          pos1 = param.find(".")
          if pos1 > 0:
            if param[:pos1].strip() in Calculator.__classList:
              pline = "(%s, '%s')" %(param[:pos1].strip(),
                                     param[pos1+1:].strip())
            else:
              pline = "(locals(), '%s')" %param[pos1+1:].strip()
          elif param in self.__USEFUL_WORDS:
            pline = "()"
          elif param not in Calculator.__classList:
            pline = "(locals(), '%s')" %param
        else:
          path = os.path.dirname(__file__)
          ext = os.path.splitext(Calculator.__default)[1]
          pline = "(path='%s')" %self.fixFilename("*")
      if word in self.__USEFUL_CMDS:        # Expand & parse __USEFUL_CMDS
        if word == "load":    # Need extra large files setup
          loadLine = Calculator.load(param, noError=doFirstLoad)
        elif word == "test":
          loadLine = Calculator.test(param, firstTest)
          firstTest = False
        else:    # version - ignore parameters
          pline = "'R',%s" %__version__
          for mod in Calculator.__moduleList:
            if mod != "R":
              pline += ",'%s',%s.version()" %(mod, mod)
          loadLine = "Calculator." +word +"(%s)" %pline
        self.__lexer.reset(loadLine)
        bufs = self.__parseTokens(False, isAns) # No eval words in scripts
        if len(bufs) != 1:
          if len(bufs) != 0:
            raise Exception("Command word processing error: %s" %line)
          bufs = ((isAns, "", []),)
        isAns,code = bufs[0][:2]
      elif word in self.__USEFUL_LIBS:
        code = 'sys.stdout.write("%s\\n" %(Lib.' +word \
                                +(pline if pline else "()") +'))'
      else:
        code = "Calculator." +word +(pline if pline else "()")
    return isAns,code

  def __processExec(self, buf):
    """Call processExec within calc? adding catch block for exec."""
    code = buf[1]
    cnt = 0
    if not code:
      return None
    elif not buf[0]:  # isAns so do eval with return ans
      tmp = "try:\n"
      if Lib._isVerbose():
        tmp += "  import traceback\n"
      elif code.strip()[0] == "#":
        tmp += "  None\n"
      for line in code.splitlines():
        tmp += "  %s\n" %line
      code = tmp +"except Exception as e:\n"
      if Lib._isVerbose():
        code += "  traceback.print_exc()\n"
      else:
        code += "  sys.stdout.write('%s: %s\\n' %(type(e).__name__, e))\n"
      code += "  raise Calculator.ExecError(e)\n"
    return Calculator.__firstCls._processExec(buf[0], code)

  def __parseTokens(self, isUsefulWord=True, isAns=True):
    """Not a full parser because it outputs code for python to parse. It only
       changes numbers with names recognised by _validBasis() and appends these
       to a stored number array which is changed into code by _processStore().
       If a variable is assigned or print is used then python exec() is needed
       else isAns is set and ans=eval() should be used. Conversion of numbers
       to the class are ignored inside an entered class and # is seen as a
       start of a comment even if within a string. Limitations:
       * Does not escape single quote strings (only " and 3"s)
       * Does not escape quotes so use single quote within strings
       * startLine is used by calcS to not expand the first variable[,...].
       Return list of (isAns, code) for semicolons in input."""
    SpaceChars = (' ', '\t', "NEWLINE")
    bufs = []               # Output lines, isAns, doUsefulWords, ansAssigns
    ansAssigns = []         # Line positions of ans replacements
    code = ""               # Current output line
    doUsefulWord = ""       # Process this special word
    doLineExpand = True     # Do expand if not special word or function
    noBrackExpand = 0       # Don't expand basis inside calc class
    loadCmdExpand = 0       # Process load cmd (nested in file or not)
    state = ParseState()    # Store basis & numbers for conversion
    isComment = False       # Ignore commented text
    isAnsAssign = False     # History needs expanded ans
    quoteCnt = 0            # Ignore inside double quotes only
    quotesCnt = 0           # Ignore inside triple double quotes only
    bracketCnt = 0          # Ignore inside brackets if noBrackExpand
    checkStore = False      # Process the state immediately
    signVal = ""            # Current sign for number or name
    notEmpty = False        # Check line is not empty
    for token in self.__lexer.process():
      #print(token, state, noBrackExpand, "->", code[-20:])
      isSpaced = False
      if token.type == "NEWLINE" and not quotesCnt:
        isSpaced = True
        isComment = False
        checkStore = True
        if state.extendLine:
          if state.store:
            token.value = ""
        else:
          doLineExpand = True
          state.startLine = True
          if state.store:
            state.aftFill += "\n"
            token.value = ""
      elif isComment:
        pass
      elif (token.type == "COMMENT" and not quotesCnt):
        isComment = True
        checkStore = True
        isAns = False
      elif token.type == "QUOTES":
        quotesCnt = 1 - quotesCnt
        quoteCnt = 0
        checkStore = quotesCnt
      elif token.type == '"':
        if not quotesCnt:
          quoteCnt = 1 - quoteCnt
          checkStore = quoteCnt
      elif quoteCnt or quotesCnt:
        pass
      elif token.type == "BRACKS":
        if token.value == '(':
          bracketCnt += 1
          if loadCmdExpand == 2 and bracketCnt == 1:
            token.value = '("'
        else:
          bracketCnt -= 1
          if bracketCnt <= noBrackExpand:
            noBrackExpand = 0
          if loadCmdExpand and bracketCnt == 0:
            if loadCmdExpand == 2:
              token.value = '")'
            loadCmdExpand = 0
        checkStore = True
      elif token.type == "EQUALS":
        checkStore = True
      elif token.type == "SIGNS":
        code += signVal
        signVal = token.value
        token.value = ""
      elif token.type == "MULTS":
        if state.store:
          if sys.version_info.major == 2: # Upgrade Python2 to v3
            if token.value == "/" and state.store[-1][0].find(".") < 0 \
              and state.store[-1][0].find("E") < 0:
              state.store[-1][0] += ".0"  # 1/3 is 0 in v2
          state.isMults2 = True
          code += Calculator.__inCls._processStore(state)
        else:
          code += state.signVal
          state.signVal = ""
        state.multVal = token.value
      elif token.type == "NUMBER":
        sgn = signVal if signVal else "+"
        if not state.store:
          state.signVal = "+" if signVal else ""
        if state.lastTyp == "MULTS":
          state.isMults1 = True
          if sys.version_info.major == 2 and state.multVal == "/":
            if token.value.find(".") < 0 and token.value.find("E") < 0:
              token.value += ".0"
        state.store.append([sgn +token.value, None])
        token.value = ""
        signVal = ""
      elif token.type == "NAME":
        validBasis = Calculator.__inCls._validBasis(token.value)
        if validBasis and doLineExpand:
          sgn = "-1" if signVal == "-" else "+1"
          if state.store:
            if state.lastTyp == "NAME":
              if token.value[0] != ".":
                raise Exception("Invalid basis duplication")
              state.store[-1][1] += token.value
            elif state.lastBasis and validBasis != state.lastBasis:
              code += Calculator.__inCls._processStore(state)
              state.signVal = "+"
              state.store.append([sgn, token.value])
            elif state.lastTyp == "NUMBER":
              state.store[-1][1] = token.value
            else:
              state.store.append([sgn, token.value])
          else:
            state.signVal = "+" if signVal else ""
            if state.lastTyp == "MULTS":
              state.isMults1 = True
            state.store.append([sgn, token.value])
          token.value = ""
          signVal = ""
          state.lastBasis = validBasis
        else:
          if state.store:
            code += Calculator.__inCls._processStore(state)
          code += signVal
          signVal = ""
          if token.value in Calculator.__moduleList:
            noBrackExpand = bracketCnt +1
          elif token.value == "ans":
            ansAssigns.append(token.lexpos)
            token.value = 'eval("ans")'
            isAnsAssign = True
          elif state.startLine:
            if token.value in self.__USEFUL_WORDS:
              if isUsefulWord:
                doUsefulWord = token.value
                if bracketCnt == 0 and token.value == "load":
                  loadCmdExpand = 1
              elif token.value == "load":
                if bracketCnt == 0:
                  loadCmdExpand = 2
                  token.value = "Lib._checkName"
              doLineExpand = False
            elif token.value in self.__PYTHON_WORDS:
              isAns = False
            elif token.value in self.__PYTHON_FUNCS:
              doLineExpand = False
              isAns = False
      else:  # All OTHER tokens
        isSpaced = (token.type in SpaceChars)
        if token.type == '=':
          if noBrackExpand and len(state.store) == 1 \
                           and state.store[0][0] == "+1":
            code += state.store[0][1]
            state.reset()
          else:
            doLineExpand = True
          if bracketCnt == 0 and state.lastTyp not in ("!", "<", ">"):
            isAns = False
            if isAnsAssign:
              raise Exception("Can't assign to ans")
          checkStore = True
        elif isSpaced:
          if state.store:
            state.aftFill += " "
            token.value = ""
        elif token.type == "\\":
          if state.store:
            token.value = ""
          code += signVal
          signVal = ""
        elif token.type == "," and loadCmdExpand == 1:
          bufs.append((isAns, code +")", ansAssigns, doUsefulWord))
          code,ansAssigns = "", []
          token.value = "load("    # Fix load dependencies
        else:
          checkStore = True
      if checkStore:
        checkStore = False
        if state.store:
          code += Calculator.__inCls._processStore(state)
        code += signVal
        signVal = ""
      code += token.value
      if not (isSpaced or isComment or quoteCnt or quotesCnt):
        state.extendLine = (token.type == "\\")
        state.lastTyp = token.type
        if token.type in (':', ';'):
          if token.type == ';':
            if notEmpty and isUsefulWord:
              bufs.append((isAns, code[:-1], ansAssigns, doUsefulWord))
              code,isAns,ansAssigns,doUsefulWord = "", True, [], ""
            state.startLine = True
            isAnsAssign = False
            doLineExpand = True
            loadCmdExpand = 0
            notEmpty = False
          else:
            notEmpty = True
        elif token.type not in (',', "NAME"):
          state.startLine = False
          notEmpty = True
        elif not state.store:
          state.lastName = token.value
          notEmpty = True
        else:
          notEmpty = True
    if state.store:
      code += Calculator.__inCls._processStore(state)
    if code and notEmpty:
      bufs.append((isAns, code, ansAssigns, doUsefulWord))
    return bufs

  def __getInput(self, runExec):
    """Tokenise the line with ply.lex and partially parse it to change basis
       numbers into Class constructors and keywords to Lib methods.
       Return list of (isAns,code) which is processed by python as
         ifAns: eval(code); else: exec(code).
       If lastLine then a backslash was entered as the last char of the 
       previous input and lines should be accumulated with the backslash
       replaced by a new line character and isAns is set False."""
    out = []   # list of (isAns, code)
    prompt = "calc%s> " %Calculator.__prompt
    if runExec:
      line = runExec
    else:
      try:
        line = ""
        while True:
          if sys.version_info < (3, 0):
            line += raw_input(prompt)  # Use raw_input with readline fixes
          else:
            line += input(prompt) 
          if line and line[-1] == '\\':
            line += '\n'
          else:
            break
      except EOFError:
        line = "quit\n" 
        sys.stdout.write("\n")
    if line:
      self.__line = line
      idx = line.find('(')
      idx = len(line) if idx < 0 else idx
      if self.__lexer:
        self.__lexer.reset(line)
        out = self.__parseTokens()
      else:
        out = [(False, line, []),]
    return out

  def processInput(self, args):
    """Process the command line options and commands or run processor.
       Processor loops over input lines parsing then executing the converted
       line until exit. Commands are run immediately."""
    runExec = ""
    checkCalc = False
    doLoad = os.path.isfile(os.path.join(os.path.dirname(__file__),
                                         Calculator.__default))
    try:
      for idx,arg in enumerate(args[1:]):
        opt = arg if (arg and arg[0] == '-') else "--"
        if "h" in opt or arg == "--help":
          sp = "          "
          raise Exception(Calculator.__cHelp +"\n" \
            + "%sEnter a command such as help which lists all commands.\n" %sp \
            + "%sNo command line command/calculation runs the calculator.\n"%sp\
            + "%sOptions set full resolution, logging & stop auto load." %sp)
        elif opt == "--":
          runExec = " ".join(args[idx +(2 if arg == "--" else 1):])
          break
        for pos,ch in enumerate(opt):
          if ch == "r" or arg == "--resolution":
            Matrix.Resolution(0)
          elif ch == "v" or arg == "--verbose":
            Lib.verbose(True)
          elif ch == "n" or arg == "--noLoad":
            doLoad = False
          elif ch != '-':
            raise Exception("Invalid option: %s" %opt)
      doCmd = "load(noError=True)" if doLoad else ""
      ans = None
      while True:
        try:
          if doCmd:
            bufs = self.__getInput(doCmd)
          else:
            bufs = self.__getInput(runExec)
          anyAns,anyAssign = False,False
          assignAns = []
          for buf in bufs:
            if buf[3]: # doUsefulWord - run lex again
              if Lib._isVerbose():
                sys.stdout.write("LOG: " +buf[1] +'\n')
              tmpAns = self.__processExec(self.__parseUsefulWord(*buf))
              if runExec and Calculator.__inCls != Calculator.__firstCls:
                checkCalc = True
              Calculator.__saveVar = None
            else:
              if checkCalc:
                raise Exception("Can't change calculator on the command line")
              if Lib._isVerbose():
                sys.stdout.write("LOG: " +buf[1] +'\n')
              assignAns.append(ans)
              tmpAns = self.__processExec(buf[:2])
            if buf[2]:
              anyAssign = True
            if buf[0]:   # isAns
              anyAns = True
              if isinstance(tmpAns, float):
                resol, resolForm, resolFloat = Lib._getResolutions()
                ans = resolForm %tmpAns
                if ans.find(".") < 0 and tmpAns != int(tmpAns):
                  ans = resolFloat %tmpAns
                sys.stdout.write("ans = %s\n" %ans)
              elif tmpAns is not None:
                ans = tmpAns
                if isinstance(ans, tuple):
                  ans = (ans,)
                sys.stdout.write("ans = %s\n" %ans)
          if anyAssign:
            for idx in reversed(range(len(bufs))):
              buf = bufs[idx]
              for pos in reversed(buf[2]):  # Expand ans in history
                self.__line = "%s(%s)%s" %(self.__line[:pos],
                                assignAns[idx], self.__line[pos+3:])
            if readline:        # Push to cmdline history
              readline.add_history(self.__line)
          Calculator.__history.append(self.__line)
          self.__line = ""
        except Calculator.ExecError as e:
          pass  # Already reported
        except KeyboardInterrupt:
          if Lib._isVerbose():
            traceback.print_exc()
        except Exception as e:
          if Lib._isVerbose():
            traceback.print_exc()
          else:
            sys.stdout.write('%s: %s\n' %(type(e).__name__, e))
        if Calculator.__lastCmd:
          if readline:
            readline.add_history(Calculator.__lastCmd)
          Calculator.__lastCmd = ""
        if runExec and not doCmd:
          break
        doCmd = None
    except KeyboardInterrupt:
      if Lib._isVerbose():
        traceback.print_exc()
    except Exception as e:
      cmd = "%s" %os.path.basename(args[0])
      opts = "[-r|--resolution] [-v|--verbose] [-n|--noLoad]"
      outLines = ("Usage: %s [-h|--help]" %cmd,
        "Usage: %s %s [<cmd|calculation>]" %(cmd, opts),
        "Summary:  %s" %e,
        "PIP: readline not installed - no command line history",
        "PIP: ply not installed - no parsing of basis numbers")
      for line in outLines[:3]:
        sys.stderr.write(line +'\n')
      if not readline and sys.platform != "win32":
        sys.stderr.write(outLines[3] +'\n')
      if not ply_lex:
        sys.stderr.write(outLines[4] +'\n')

################################################################################
## Basis numbers for Calculator are just precise floats with resolution.      ##
################################################################################
class Real(float):
  """Class to emulate floats with just equality precision and print resolution
     modified. It has no methods, use math methods instead eg sin(pi) but change
     float to Real eg Real(pi) or pi*1. Change to complex numbers using calc(Q).
     """
  __loadedCalcs = []                     # Notify any other calc loaded
  def __float__(self):
    return super(self)
  def __int__(self):
    return trunc(self)
  def __str__(self):
    """Overload string output. Printing taking resolution into account."""
    return Lib.getResolNum(float.__float__(self))
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if Lib._isVerbose():
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __add__(self, q):
    """Add 2 floats & return a Real."""
    Lib._checkType(q, (int, float), "add")
    return Real(float.__add__(self, q))
  __radd__ = __add__
  def __neg__(self):
    """Use multiplication for negation."""
    return self.__mul__(-1)
  def __sub__(self, q):
    """Subtract 2 floats & return a Real."""
    Lib._checkType(q, (int, float), "sub")
    return Real(float.__sub__(self, q))
  def __rsub__(self, q):
    """Subtract Real from scalar with Real output."""
    return self.__neg__().__add__(q)
  def __mul__(self, q):
    """Multiply 2 floats & return a Real."""
    Lib._checkType(q, (int, float), "mul")
    return Real(float.__mul__(self, q))
  __rmul__ = __mul__
  def __div__(self, q):
    """Divide 2 floats & return a Real."""
    Lib._checkType(q, (int, float), "div")
    return Real(float.__div__(self, q))
  def __rdiv__(self, q):
    return Real(float.__div__(float(q), self))
  def __mod__(self, q):
    """Modulo % operator for Real."""
    Lib._checkType(q, (int, float), "mod")
    return Real(float.__mod__(self, q))
  __rmod__ = __mod__
  def __floordiv__(self, q):
    """Real div (//) for 2 Reals. This is an int as float."""
    Lib._checkType(q, (int, float), "floor")
    return Real(float.__floordiv__(self, q))
  __rfloordiv__ = __floordiv__
  def __eq__(self, cf):
    """Overload float compare taking resolution into account."""
    precision = Lib._getPrecision()
    return (abs(self -cf) < precision)
  __pow__ = pow
  def grades(self, maxSize=0):
    """grades([maxSize])
       Return a list with scalar set or not set."""
    Lib._checkType(maxSize, int, "grades")
    return [1 if self else 0]

  @staticmethod
  def version():
    """version()
       Return the module version string."""
    return __version__

  @staticmethod
  def Basis(numDims):
    """Basis(numDims)
       Return no basis elements since there are none."""
    return ()

  @staticmethod
  def IsCalc(calc):
    """Check if named calculator has been loaded."""
    return (calc in Real.__loadedCalcs)

  #########################################################
  ## Calculator class help and basis processing methods  ##
  #########################################################
  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for Real."""
    calcHelp = """Calculator - Simple calculator to build more complex processors.
          Use calc(Q) for complex and quaternion calculations."""
    return (("R"), ("Real", "math"), "", "default.calc", calcHelp, 
            "Use scalar method instead of Real numbers.")

  @classmethod
  def _setCalcBasis(cls, calcs, dummy):
    """Load this other calculator. Does nothing for reals/floats."""
    Real.__loadedCalcs = calcs
    return ""

  @classmethod
  def _validBasis(cls, value):
    """Used by Calc to recognise basis chars."""
    return False

  @classmethod
  def _processStore(cls, state):
    """No convertion needed as there are no basis chars."""
    line = ""
    signTyp = state.signVal
    for num,bas in state.store:
      if num.find(".") < 0:  # Don't convert int to Real
        line += num
      else:
        line += "Real(%s)" %num
    line += state.aftFill
    state.reset()
    if line[0] == "+" and not signTyp:
      return line[1:]
    return line

  @staticmethod
  def _processExec(isAns, code):
    """Run exec() or eval() within this module's scope."""
    if isAns:
      global ans
      ans = eval(code, globals())
      return ans
    else:
      exec(code, globals())
      return None

################################################################################
if __name__ == '__main__':
  from math import *
  import traceback
  try:
    import importlib
  except:
    importlib = None          # No changing of calculators

  Tests = [\
    """""",
    """# Test 1 Check Tensor.Rotations identities. From Wikipedia Matrix Exp.
       def Rtest(G,P,N):
         b1 = (P == -G * G)
         b2 = (P * P == P)
         b3 = (P *G == G)
         b4 = (G *P == G)
         b5 = (N *N == N)
         b6 = (N *P == Tensor.Diag([0] *G.shape()[0]))
         b7 = (N *G == Tensor.Diag([0] *G.shape()[0]))
         return b1 and b2 and b3 and b4 and b5 and b6 and b7

       # Case for G = Tensor((0,-1),(1,0))==i, P=I, so R(a) = I*cos(a) +G*sin(a)
       d2 = Rtest(*Tensor.Rotations(Tensor(1,0), Tensor(0,1)))
       d3 = Rtest(*Tensor.Rotations(Tensor(1,0,0), Tensor(0,1,2) /sqrt(5)))
       Calculator.log(d2 and d3, (d2, d3))""",
    """# Test 2 Euler Matrix is inverse of Euler.matrix.
       test=Euler(pi/6, pi/4, pi/2).matrix()
       store=Euler.Matrix(test).matrix()
       Calculator.log(store == test, store)""",
    """# Test 3 Euler 10-D Matrix is inverse of Euler.matrix.
       test=Matrix(list((x *0.01 for x in range(1,46))))
       store=Euler.Matrix(Euler(*test).matrix())
       Calculator.log(Matrix(store) == test, store)""",
    """# Test 4 Euler 15-D Matrix is inverse of Euler.matrix.
       test=Matrix(list((x *0.01 for x in range(1,106))))
       store=Euler.Matrix(Euler(*test).matrix())
       Lib.precision(1.5E-9)
       Calculator.log(Matrix(store) == test, store)""",
    """# Test 5 Tensor.diag gives the trace
       n=6; store = sum(Tensor.Diag(list(range(1,n+1))).diag()); test=n*(n+1)/2
       Calculator.log(store == test, store)""",
    """# Test 6 Tensor.diag(vector) gives the dot product
       v=Tensor(1,2,3,4,5,6); store=sum(v.diag(v)); test=sum((x*x for x in v))
       Calculator.log(store == test, store)""",
       ]
  calc = Calculator(Real, Tests)
  calc.processInput(sys.argv)

elif sys.version_info.major != 2:  # Python3
  def execfile(fName):
    """To match Python2's execfile need: from pathlib import Path
       exec(Path(fName).read_text())."""
    exec(Lib.readText(fName))
