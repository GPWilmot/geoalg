#!/usr/bin/env python
###############################################################################
## File: calcR.py needs calcCommon.py and is part of GeoAlg.
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
## This module contains classes: Lexer, ParseState, CommonTest and Calculator
## used by all basis number calculators. Run help for more info.
## Start with either calcR.py, python calcR.py or see ./calcR.py -h.
###############################################################################
__version__ = "0.2"
import sys, math, os
import platform, glob
import traceback
from calcCommon import *
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
  literals = '~!@$%^&?[]{}|,"<>=:;\'\\\. \t' 
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
    self.lastVal = None    # only if lastTyp == SIGNS
    self.startLine = True
    self.extendLine = False
    self.reset()
  def reset(self):
    self.store = []        # pairs of (value, basis)
    self.signTyp = ""
    self.lastBasis = 0
    self.isNewLine = False
    self.isMults1 = False
    self.isMults2 = False
  def __repr__(self):
    sgn = self.lastVal if self.lastTyp in ("SIGNS", "MULTS") else "None"
    mul1 = "|" if self.isMults1 else ""
    mul2 = "|" if self.isMults2 else ""
    return "State:%s%s%s LastType:%s Basis:%s" %(mul1, self.store, mul2,
                                                 sgn, self.lastBasis)

###############################################################################
class CommonTest():
  """Class to provide common test processing and logging code."""
  __inTests    = None          # Test strings for running
  __testIndent = 7             # Test string indents
  __testCnt    = 0             # Counter for running tests
  __testPass   = False         # Status for running tests

  @staticmethod
  def log(test, store=None):
    """Common equality status and logging for each test case."""
    if CommonTest.__inTests:
      if not test:
        CommonTest.__testPass = False
      sys.stdout.write("Test%d: %s\n" %(CommonTest.__testCnt,
                       "PASS" if test else "FAIL"))
      CommonTest.__testCnt += 1
      if Common._isVerbose() and store is not None:
        sys.stdout.write(str(store) +'\n')

  @staticmethod
  def testCnt():
    """Return the number of tests available to run."""
    return len(CommonTest.__inTests) -1 if CommonTest.__inTests else 0

  @staticmethod
  def logResults():
    """Summarise the results of an all tests run."""
    if CommonTest.__inTests:
      sys.stdout.write("All tests passed\n" if CommonTest.__testPass \
                                       else "Some tests failed\n")

  @staticmethod
  def _initRunTests(tests, indent=None):
    """Setup to run internal tests."""
    if tests and len(tests) > 1 and ply_lex:
      CommonTest.__inTests = tests
      if indent is not None:
        CommonTest.__testIndent = indent

  @staticmethod
  def __initTestExec(case):
    """Initialise testing cases from Calculator."""
    if CommonTest.__inTests:
      CommonTest.__testPass = True
      CommonTest.__testCnt = case

  @staticmethod
  def getTestLines(number=None):
    """Return exec lines for __inTests assuming this has at least 2 fields.
       number is blank to run all tests or the str(test number). Expect 
       Common.log to be included at the end of each test and runtest[0] is
       initialisation for all tests. Copies all to history except Common.log
       and adds logResults to end if the all tests are run."""
    if not CommonTest.__inTests:
      raise Exception("No tests available to run")
    Common.precision(1E-15)  # Reset precision in case a test changed it
    if not number:
      first,last = (1, len(CommonTest.__inTests))
      CommonTest.__initTestExec(1)
    elif number.isdigit() and int(number) in range(1,len(CommonTest.__inTests)):
      first,last = (int(number), int(number) +1)
      CommonTest.__initTestExec(first)
    else:
      raise Exception("Test argument should be in range 1..%d" \
                               %CommonTest.testCnt())
    code = ""
    for idx,lines in enumerate(CommonTest.__inTests[first:last]):
      for pos,line in enumerate(lines.splitlines()):
        if line[0] != "#":
          break
      for block in ((lines,0,pos), (CommonTest.__inTests[0],0,-1),
                    (lines,pos,-1)):
        pos = len(block[0]) if block[2] < 0 else block[2] 
        for line in block[0].splitlines()[block[1]:pos]:
          if line[:CommonTest.__testIndent] == " " *CommonTest.__testIndent:
            code += line[CommonTest.__testIndent:] +'\n'
            if number and not Common._isVerbose():
              sys.stdout.write(line[CommonTest.__testIndent:] +'\n')
            if line.find("Calculator.log") < 0:
              if readline:
                readline.add_history(line[CommonTest.__testIndent:])
          else:
            code += line +'\n'
            if readline:
              readline.add_history(line)
            if number and not Common._isVerbose():
              sys.stdout.write(line +'\n')
      if Common._isVerbose() and idx < last -first -1:
        code += 'sys.stdout.write("\\n")\n'
    if not number:
      code += "Calculator.logResults()\n"
    return code[:-1]

################################################################################
class Calculator:
  """Command line processor parsing basis numbers into python code.Help uses the
     doc strings and class variable documents strings (starting with one _)."""
  __USEFUL_WORDS  = ("version", "verbose", "precision", "resolution",
                     "help", "quit", "exit", "save", "load", "show", "test",
                     "clear", "calc", "vars")       # Words processed by parsing
  __USEFUL_CMDS   = ("load", "test", "version")     # Words with evaluation
  __USEFUL_FILE   = ("show", "clear", "load", "save")  # Ordered filename words
  __PYTHON_WORDS  = ("print", "raise", "from", "with", # Exec not eval
                     "with", "global", "raise", "import",
                     "for", "while")
  __PYTHON_FUNCS  = ("def",)                          # Exec & no expand fn
  __PYTHON_STARTS = ("in", "lambda",)                 # No expand 'til!.,
  __oldCls = {}                                # Previous __inCls's
  __classList = ["Common", "Matrix", "Euler"]  # From calcCommon
  __moduleList = []                            # Calc classes loaded
  __promptList = []                            # Subordinate modules
  __history = []                               # For saving and showing
  __lastCmd = ""                               # Add load at end
  __prompt = ""                                # Current calculator name

  def __init__(self, clsType, tests=[], indent=None):
    """Singleton calculator for basis numbers."""
    modList, clsList, default, cHelp, eHelp  = clsType._getCalcDetails()
    CommonTest._initRunTests(tests, indent)
    Calculator.__inCls = clsType              # Current calculator
    Calculator.__firstCls = clsType           # Initial __inCls
    Calculator.__oldCls[clsType.__name__] = clsType
    Calculator.__classList.extend(clsList)
    Calculator.__moduleList.append(modList[0])
    Calculator.__promptList.extend(modList[1:])
    Calculator.__prompt = modList[0]          # Current calculator name
    Calculator.__cHelp = cHelp                # Calculator help text
    Calculator.__eHelp = eHelp                # Load extra help text
    Calculator.__default = default            # Load file default value
    self.__lines = ""                         # Store to history if OK
    self.__lexer = Lexer() if ply_lex else None

  @staticmethod
  def version(*args):
    """Print the common version with optional versions for other modules."""
    vers = Common.version()
    idx = 0
    while idx +1 < len(args):
      if __version__ != args[idx +1]:
        vers += "(%s: %s)" %(args[idx], args[idx +1])
      idx += 2
    sys.stdout.write(vers +'\n')

  @staticmethod
  def verbose(verbosity=None):
    """Toggle or set verbosity for traceback reporting."""
    Common.verbose(verbosity)

  @staticmethod
  def precision(precise=None):
    """Set equality precision or reset to default e-15."""
    Common.precision(precise)

  @staticmethod
  def resolution(digits=None):
    """Set print format digits or reset to default 17."""
    Common.resolution(digits)

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
    """Common equality status and logging for each test case."""
    CommonTest.log(test, store)

  @staticmethod
  def logResults():
    """Summarise the results of an all tests run."""
    CommonTest.logResults()

  @staticmethod
  def test(number=None):
    """Run test number or all tests."""
    name = Calculator.__firstCls.__name__
    name = "R" if name == "Real" else name
    if Calculator.__prompt not in Calculator.__promptList +[name]:
      raise Exception("Invalid calculator for %s tests" %name)
    Calculator.__lastCmd = "test(%s)" %number.strip() if number else ""
    return CommonTest.getTestLines(number.strip() if number else None)

  class ExecError(Exception):
    """Don't report exceptions outside exec() if already reported inside.
       Force raising of this exception instead to catch it outside."""
    pass

  @staticmethod
  def quit(filename=None):
    """Exit the calculator."""
    sys.exit(1 if filename else 0)
  exit = quit

  @staticmethod
  def show(filename=None):
    """List history, filename contents or defailt file contents."""
    if filename:
      sys.stdout.write(Common.readText(filename).replace('\\', "\\\\") +'\n')
    else:
      sys.stdout.write("\n".join(Calculator.__history).replace('\\', "\\\\") +'\n')

  @staticmethod
  def load(filename=None):
    """Load filename or default file and add to history."""
    if not filename:
      raise Exception("No filename entered")
    code = Common.readText(filename)
    if not code:
      raise Exception("File empty")
    for line in code.splitlines():
      Calculator.__history.append(line)
    if Common._isVerbose():
      if readline:
        readline.read_history_file(filename)
      Calculator.__lastCmd = "load(%s)" %filename
    return code

  @staticmethod
  def save(filename=None):
    """Save history to filename or to default file."""
    if not filename:
      raise Exception("No filename entered")
    with open(filename, "a") as fp:
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
  def vars(lVars):
    """List local variables to line length 80."""
    lv = []
    for name,var in lVars.items():
      if isinstance(var, (int, float, list, tuple, Common._basestr)):
        if name[0] != "_":
          lv.append(name)
    for name,var in lVars.items():
      if str(type(var)) == "<type 'function'>":
        lv.append(name +"()")
    size = 0
    for name in lv:
      size = max(size, len(name))
    width = 80 // (size +2) +1
    fmt = "%%%ds" %(size +2)
    for pos in range(0, len(lv), width):
      for name in lv[pos:pos +width]:
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
    topCls = None
    for calc in calcs:
      if calc is None:
        sys.stdout.write("Calculator%s: %s\n" %("s" if len(names) > 1 else "",
                                        ", ".join(names)))
      elif calc in names:
        mod = "calc%s" %calc
        if calc not in Calculator.__moduleList: # Not already loaded
          clsName = "Real" if calc == "R" else calc
          code = 'if importlib:\n'
          code += '  pkg = importlib.import_module("calc%s")\n' %calc
          code += '  globals()["%s"] = getattr(pkg, "%s")\n' %(clsName, clsName)
          Calculator.__firstCls._processExec(False, code)
          code = 'globals()["%s"] if importlib else None' %clsName
          newCls = Calculator.__firstCls._processExec(True, code)
          if newCls is None:
            raise Exception("No importlib: run %s.py from the command line"%mod)
          Calculator.__oldCls[clsName] = newCls
          modList, clsList, default, Calculator.__cHelp, Calculator.__eHelp \
                               = newCls._getCalcDetails()
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
            Calculator.__default = default
        topCls = Calculator.__inCls
      else:
        raise Exception("No such calculator: calc%s" %calc)
    if topCls:
        msg = topCls._setCalcBasis(Calculator.__moduleList, Calculator)
        if msg:
          sys.stdout.write(msg +'\n')

  @staticmethod
  def help(cls=None, obj=None, path=None):
    """Print introduction, obj list or obj's documentation."""
    if not cls:
      tmp = "history%s default or named file"
      opt = ""
      if not readline:
        opt += "\nPIP: readline not installed - no command line history"
      if not ply_lex:
        opt += "\nPIP: ply not installed - no parsing of basis numbers"
      test = "" if CommonTest.testCnt()==0 else "%9s test or test(1..%d)%s\n"\
             %("", CommonTest.testCnt(), "  - run all tests or just one")
      extra = ""
      for more in Calculator.__oldCls.values():
        if more and more != Calculator.__inCls:
          if Calculator.__eHelp:
            extra += '           %s\n' %Calculator.__eHelp
      sys.stdout.write('%s\n' %Calculator.__cHelp +extra \
          +'Commands: help or help(%s)\n'%"|".join(Calculator.__classList) \
          +'          calc or calc(<calc>) - list or change calculator\n' \
          +'          show or show(<file>) - display %s\n' %(tmp %",")\
          +'          load or load(<file>) - load %s\n' %(tmp %" from") \
          +'          save or save(<file>) - append %s\n' %(tmp %" to") \
          +test \
          +'          clear or clear<file>) - clear %s\n' %(tmp %"or") \
          +'          precision, resolution, verbose, version - see Common\n' \
          +'          vars                 - list local variables/functions\n'\
          +'          quit or exit or ^d   - exit (^d is control-d)' \
          +opt +'\n')
      if path:
        fNames = list((os.path.basename(fName) for fName in glob.glob(path)))
        if len(fNames) == 0:
          sys.stdout.write('No files to load\n')
        else:
          sys.stdout.write('Files to load: %s\n' %" \t".join(fNames))
    elif not hasattr(cls, "__name__"):
      raise Exception("Invald help parameter")
    else:
      Calculator.__help(cls, obj)

  @staticmethod
  def __help(cls, more, include=[]):
    """Common internal documentation function for any class."""
    if more:
      Common._checkType(more, Common._basestr, "help")
      if more == cls.__name__:
        more = "__init__"
      if hasattr(cls, str(more)):
        doc = "   %s\n" %getattr(cls, str(more)).__doc__
        sys.stdout.write(doc if doc else \
                         "No documentation for this method." +'\n')
      else:
        sys.stdout.write("Method does not exist.\n")
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
          if cls.__name__ not in ("Common", "Calculator"):
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

  def __parseUsefulWord(self, firstWord, isAns, line):
    """Change __parseTokens USEFUL_WORDS and USEFUL_CMDS into useful Common
       methods. Process commands with optional argument. Argument is
       assumed to be a module if help, a filename with default for file commands
       and ignored for other commands. Quotes are removed if found. Return text
       for exec() or exception. Return "" if quitting. USEFUL_CMDS return the
       code to run using lex which is not nested so is added after expansion."""
    code = ""
    for sline in line.split(';'):
      if firstWord and sline.find(firstWord) >= 0:
        word = firstWord
        firstWord = ""
      else:
        word = ""
        for uWord in self.__USEFUL_WORDS:
          if sline.lstrip().find(uWord) == 0:
            word = uWord
            break
      pline = sline.replace(word, "").strip()
      pos1 = pline.find("(")
      pos2 = pline.find(")")
      param = pline[pos1 +1:pos2] if pos1 < pos2 and pos2 > 0 else None
      extra = pline[pos2 +1:].strip()                # Anything after word or )
      if word and extra:
        raise Exception("Command arguments must be inside brackets")
      if not word or extra or pline[:pos1].strip():  # Ignore if not single cmd
        code += sline +';'
      else:  # Separate out parameters
        if param and len(param) > 1:   # Remove quotes
          if param[0] == '"' and param[-1] == '"' or \
             param[0] == "'" and param[-1] == "'":
            param = param[1:-1]
        if word in self.__USEFUL_FILE:  # File cmds can use default filename
          if param:                     # Maybe use default filename extension
            if not os.path.splitext(param)[1]:
              param += os.path.splitext(Calculator.__default)[1]
          elif pos1 >= 0 or word in self.__USEFUL_FILE[2:]: # Use default file
            param = Calculator.__default
          if param:
            if not os.path.dirname(param):   # Use default path
              path = os.path.dirname(__file__)
              param = os.path.join(path, param)
            pline = "('%s')" %param
        elif word == "calc" and param:
          pline = "('%s')" %param.upper()
        elif word == "vars":
          pline = "(locals())"
        elif word == "help":
          if param:
            pos1 = param.find(".")
            if pos1 > 0:
              pline = "(%s, '%s')" %(param[:pos1], param[pos1+1:])
          else:
            path = os.path.dirname(__file__)
            ext = os.path.splitext(Calculator.__default)[1]
            pline = "(path='%s/*%s')" %(path, ext)
        if word in self.__USEFUL_CMDS:        # Expand & parse __USEFUL_CMDS
          if word == "load":
            line = Calculator.load(param)
          elif word == "test":
            line = Calculator.test(param)
          else:    # version - ignore parameters
            pline = "'R',%s" %__version__
            for mod in Calculator.__moduleList:
              if mod != "R":
                pline += ",'%s',%s.version()" %(mod, mod)
            line = "Calculator." +word +"(%s)" %pline
          self.__lexer.reset(line)
          isAns,code = self.__parseTokens(True) # No usefulwords in scripts
        else:
          code += "Calculator." +word +(pline if pline else "()")
        if len(code) > 0 and code[-1] != '\n':
          code += ';'
    return isAns,code[:-1]

  def __parseTokens(self, noUsefulWord=False):
    """Not a full parser because it outputs code for python to parse. It only
       changes numbers with names recognised by _validBasis() and appends these
       to a stored number array which is changed into code by _processStore().
       If a variable is assigned or print is used then python exec() is needed
       else isAns is set and ans=eval() should be used. Conversion of numbers
       to the class are ignored inside an entered class and # is seen as a
       start of a comment even if within a string. Limitations:
       * Does not escape single quote strings (only " and 3"s)
       * Does not escape quotes so use single quote within strings
       * startLine is used by calcS to not expand the first variable[,...]
       """
    SpaceChars = (' ', '\t', "NEWLINE")
    code = ""               # Output line
    isAns = True            # Use eval instead of exec
    doUsefulWord = ""       # Process this special word
    doLineExpand = True     # Do expand if not special word or function
    noBrackExpand = 0       # Don't expand basis inside calc class
    state = ParseState()    # Store basis & numbers for conversion
    isComment = False       # Ignore commented text
    quoteCnt = 0            # Ignore inside double quotes only
    quotesCnt = 0           # Ignore inside triple double quotes only
    bracketCnt = 0          # Ignore inside brackets if noBrackExpand
    checkStore = False      # Process the state immediately
    for token in self.__lexer.process():
      #print(token, state, noBrackExpand, "->", code[-20:])
      if token.type == "NEWLINE" and quotesCnt %2 == 0:
        if state.extendLine:
          quoteCnt = 0
          state.startLine = True
          if state.store:
            token.value = ""
        else:
          checkStore = True
          isComment = False
          doLineExpand = True
          if state.store:
            state.isNewLine = True
            token.value = ""
      elif isComment:
        pass
      elif (token.type == "COMMENT" and quotesCnt %2 == 0):
        isComment = True
        isAns = False
        checkStore = True
      elif token.type == "QUOTES":
        quotesCnt += 1
        quoteCnt = 0
        checkStore = True
      elif token.type == '"':
        quoteCnt += 1
        checkStore = True
      elif quoteCnt %2 == 1 or quotesCnt %2 == 1:
        pass
      elif token.type == "BRACKS":
        if token.value == '(':
          bracketCnt += 1
        else:
          bracketCnt -= 1
          if bracketCnt <= noBrackExpand:
            noBrackExpand = 0
        checkStore = True
      elif noBrackExpand:
        if bracketCnt < noBrackExpand:
          noBrackExpand = 0 
        checkStore = True
      elif token.type == "EQUALS":
        checkStore = True
      elif token.type == "SIGNS":
        if state.lastTyp == "SIGNS":
          code += state.lastVal
        state.lastVal = token.value
        token.value = ""
      elif token.type == "MULTS":
        if state.store:
          if sys.version_info.major == 2: # Upgrade Python v2 to v3
            if token.value == "/" and state.store[-1][0].find(".") < 0 \
              and state.store[-1][0].find("E") < 0:
              state.store[-1][0] += ".0"  # 1/3 is 0 in v2
          state.isMults2 = True
          code += Calculator.__inCls._processStore(state)
        if state.lastTyp == "SIGNS":
          code += state.lastVal
        state.lastVal = token.value
      elif token.type == "NUMBER":
        sgn = state.lastVal if state.lastTyp == "SIGNS" else ""
        if state.store and not sgn:
          sgn = '+'
        if state.lastTyp == "MULTS":
          state.isMults1 = True
        if sys.version_info.major == 2 and state.lastVal == "/":
          if token.value.find(".") < 0 and token.value.find("E") < 0:
            token.value += ".0"
        state.store.append([sgn +token.value, None])
        token.value = ""
      elif token.type == "NAME":
        validBasis = Calculator.__inCls._validBasis(token.value)
        if validBasis and doLineExpand:
          sgn = "-1" if state.lastTyp == "SIGNS" and state.lastVal == "-" \
                     else "+1"
          if state.store:
            if state.lastTyp == "NAME":
              if token.value[0] != ".":
                raise Exception("Invalid basis duplication")
              state.store[-1][1] += token.value
            elif state.lastBasis and validBasis != state.lastBasis:
              code += Calculator.__inCls._processStore(state)
              state.store.append([sgn, token.value])
            elif state.lastTyp == "NUMBER":
              state.store[-1][1] = token.value
            else:
              state.store.append([sgn, token.value])
          else:
            if state.lastTyp == "MULTS":
              state.isMults1 = True
            elif state.lastTyp == "SIGNS":
              code += "+"
            state.store.append([sgn, token.value])
          token.value = ""
          state.lastBasis = validBasis
        else:
          if state.store:
            code += Calculator.__inCls._processStore(state)
          if state.lastTyp == "SIGNS":
            code += state.lastVal
          if token.value in Calculator.__moduleList:
            noBrackExpand = bracketCnt +1 # Don't expand basis inside calc class
          elif token.value in self.__PYTHON_STARTS:
            state.startLine = True
          elif state.startLine:
            if token.value in self.__USEFUL_WORDS:
              if noUsefulWord:
                raise Exception("Multiple commands are not allowed")
              doUsefulWord = token.value
              noUsefulWord = True
              doLineExpand = False
            elif token.value in self.__PYTHON_WORDS:
              isAns = False
            elif token.value in self.__PYTHON_FUNCS:
              doLineExpand = False
              isAns = False
      else:  # All OTHER tokens
        if token.type == '=':
          doLineExpand = True
          if bracketCnt == 0 and state.lastTyp not in ("<", ">"):
            isAns = False
        if token.type in SpaceChars:
          if state.store or state.lastTyp == "SIGNS":
            token.value = ""
        elif token.type == "\\":
          if state.store:
            token.value = ""
          elif state.lastTyp == "SIGNS":
            code += state.lastVal
        else:
          checkStore = True
      if checkStore:
        checkStore = False
        if state.store:
          code += Calculator.__inCls._processStore(state)
        if state.lastTyp == "SIGNS":
          code += state.lastVal
        state.lastVal = ""
      code += token.value
      if not (token.type in SpaceChars or isComment or \
                  quoteCnt %2 != 0 or quotesCnt %2 != 0):
        state.extendLine = (token.type == "\\")
        state.lastTyp = token.type
        if token.type in (':', ';'):
          state.startLine = True
          doLineExpand = True
          if token.type == ';': #TBD
            isAns = False
        elif token.type not in (',', "NAME"):
          state.startLine = False
    if state.store:
      code += Calculator.__inCls._processStore(state)
    if doUsefulWord:    # Run lex again if necessary
      isAns, code = self.__parseUsefulWord(doUsefulWord, isAns, code)
    return isAns,code

  def __getInput(self, runExec):
    """Tokenise the line with ply.lex and partially parse it to change basis
       numbers into Class constructors and keywords to Common methods.
       Return isAns,code which is processed by python as
         ifAns: eval(code); else: exec(code).
       If lastLine then a backslash was entered as the last char of the 
       previous input and lines should be accumulated with the backslash
       replaced by a new line character and isAns is set False."""
    isAns,code = False,None
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
      self.__lines = line
      idx = line.find('(')
      idx = len(line) if idx < 0 else idx
      try:
        if self.__lexer:
          self.__lexer.reset(line)
          isAns,code = self.__parseTokens()
          if Common._isVerbose():
            sys.stdout.write("LOG: " +code +'\n')
        else:
          code = line
        if not isAns:
          tmp = "try:\n"
          if Common._isVerbose():
            tmp += "  import traceback\n"
          for line in code.splitlines():
            tmp += "  %s\n" %line
          code = tmp +"except Exception as exception:\n"
          if Common._isVerbose():
            code += "  traceback.print_exc()\n"
          else:
            code += "  sys.stdout.write(str(exception) +'\\n')\n"
          code += "  raise Calculator.ExecError(exception)\n"
      except Exception as e:
        isAns,code = False, 'sys.stdout.write("Error: %s\\n")' %e
        if Common._isVerbose():
          traceback.print_exc()
    elif line:
      code = line
    return isAns,code

  def processInput(self, args):
    """Process the command line options and commands or run processor.
       Processor loops over input lines parsing then executing the converted
       line until exit. Commands are run immediately."""
    runExec = ""
    doLoad = os.path.isfile(os.path.join(os.path.dirname(__file__),
                                         Calculator.__default))
    doCalc = False
    calcCmd = []
    try:
      for idx,arg in enumerate(args[1:]):
        opt = arg if (arg and arg[0] == '-') else "--"
        if "h" in opt or arg == "--help":
          sp = "          "
          raise Exception(Calculator.__cHelp +"\n" \
            + "%sEnter command such as help which list all commands.\n" %sp \
            + "%sNo command line command/calculation runs calculator.\n" %sp \
            + "%sOptions set full resolution, logging, load & calc cmds." %sp)
        elif doCalc:
          calcCmd.append(arg)
          doCalc = False
          opt = []
        elif opt == "--":
          runExec = " ".join(args[idx +(2 if arg == "--" else 1):])
          break
        for pos,ch in enumerate(opt):
          if doCalc:
            if ch == "," or not calcCmd:
              calcCmd.append("" if ch == "," else ch)
            else:
              calcCmd[-1] += ch
            if pos == len(opt) -1:
              doCalc = False
          elif ch == "r" or arg == "--resolution":
            Matrix.Resolution(0)
          elif ch == "v" or arg == "--verbose":
            Common.verbose(True)
          elif ch == "n" or arg == "--noLoad":
            doLoad = False
          elif ch == "c" or arg == "--calc":
            doCalc = True
          elif ch != '-':
            raise Exception("Invalid option: %s" %opt)
      doCmd = "load" if doLoad else ""
      if calcCmd:
        doCmd += "%scalc(%s)" %(";" if doCmd else "", ",".join(calcCmd))
      while True:
        try:
          if doCmd:
            isAns,code = self.__getInput(doCmd)
          else:
            isAns,code = self.__getInput(runExec)
          if code:
            ans = Calculator.__firstCls._processExec(isAns, code)
          else:
            ans = None
          if isAns:
            if isinstance(ans, float):
              resol, resolForm, resolFloat = Common._getResolutions()
              flt = resolForm %ans
              if flt.find(".") < 0 and ans != int(ans):
                flt = resolFloat %ans
              sys.stdout.write("ans = %s\n" %flt)
            elif ans is not None:
              sys.stdout.write("ans = %s\n" %str(ans))
          else:
            for line in self.__lines.splitlines():
              Calculator.__history.append(line)
            self.__lines = ""
        except Calculator.ExecError:
          pass  # Already reported
        except KeyboardInterrupt:
          if Common._isVerbose():
            traceback.print_exc()
        except Exception as e:
          if Common._isVerbose():
            traceback.print_exc()
          else:
            sys.stdout.write(str(e) +'\n')
        if Calculator.__lastCmd:
          if readline:
            readline.add_history(Calculator.__lastCmd)
          Calculator.__lastCmd = ""
        if runExec and not doCmd:
          break
        doCmd = None
    except KeyboardInterrupt:
      if Common._isVerbose():
        traceback.print_exc()
    except Exception as e:
      cmd = "%s" %os.path.basename(args[0])
      opts = "[-r|--resolution] [-v|--verbose] [-n|--noLoad]"
      opts += " [-c|--calc list]"
      outLines = ("Usage: %s [-h|--help]" %cmd,
        "Usage: %s %s [<cmd|calculation>]" %(cmd, opts),
        "Summary:  %s" %e,
        "PIP: readline not installed - no command line history",
        "PIP: ply not installed - no parsing of basis numbers")
      for line in outLines[:3]:
        sys.stderr.write(line +'\n')
      if not readline:
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
  def __str__(self):
    """Overload string output. Printing taking resolution into account."""
    return Common.getResolNum(self)
  def __repr__(self):
    """Overwrite object output using __str__ for print if !verbose."""
    if Common._isVerbose():
      return '<%s.%s object at %s>' % (self.__class__.__module__,
             self.__class__.__name__, hex(id(self)))
    return str(self)
  def __add__(self, q):
    """Add 2 floats & return a Real."""
    Common._checkType(q, (int, float), "add")
    return Real(float.__add__(self, q))
  __radd__ = __add__
  def __neg__(self):
    """Use multiplication for negation."""
    return self.__mul__(-1)
  def __sub__(self, q):
    """Subtract 2 floats & return a Real."""
    Common._checkType(q, (int, float), "sub")
    return Real(float.__sub__(self, q))
  def __rsub__(self, q):
    """Subtract Real from scalar with Real output."""
    return self.__neg__().__add__(q)
  def __mul__(self, q):
    """Multiply 2 floats & return a Real."""
    Common._checkType(q, (int, float), "mul")
    return Real(float.__mul__(self, q))
  __rmul__ = __mul__
  def __div__(self, q):
    """Divide 2 floats & return a Real."""
    Common._checkType(q, (int, float), "div")
    return Real(float.__div__(self, q))
  __rdiv__ = __div__
  def __mod__(self, q):
    """Modulo % operator for Real."""
    Common._checkType(q, (int, float), "mod")
    return Real(float.__mod__(self, q))
  __rmod__ = __mod__
  def __floordiv__(self, q):
    """Real div (//) for 2 Reals. This is an int as float."""
    Common._checkType(q, (int, float), "floor")
    return Real(float.__floordiv__(self, q))
  __rfloordiv__ = __floordiv__
  def __eq__(self, cf):
    """Overload float compare taking resolution into account."""
    precision = Common._getPrecision()
    return (abs(self -cf) < precision)
  __pow__ = pow
  def grades(self, maxSize=0):
    """grades([maxSize])
       Return a list with scalar set or not set."""
    Common._checkType(maxSize, int, "grades")
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
    """Test loading of other calculators. Not used in this calculator."""
    return (calc == "R")

  #########################################################
  ## Calculator class help and basis processing methods  ##
  #########################################################
  @staticmethod
  def _getCalcDetails():
    """Return the calculator help, module heirachy and classes for Real."""
    calcHelp = """Calculator - Simple calculator to build more complex processors.
          Use calc(Q) for complex and quaternion calculations."""
    return (("R"), ("Real", "math"), "default.calc", calcHelp, 
            "Use scalar method instead of Real numbers.")

  @classmethod
  def _setCalcBasis(cls, calcs, dummy):
    """Load other calculator. Does nothing for reals/floats."""
    return ""

  @classmethod
  def _validBasis(cls, value):
    """Used by Calc to recognise basis chars."""
    return False

  @classmethod
  def _processStore(cls, state):
    """No convertion needed as there are no basis chars."""
    line = "+".join("Real(%s)" %x[0] for x in state.store)
    if line.find(".") < 0:  # Don't convert int to Real
      line = (" ".join(x[0] for x in state.store))
    if state.isNewLine:
      line += '\n'
    state.reset()
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
       Common.precision(1.5E-10)
       Calculator.log(Matrix(store) == test, store)""",
    """# Test 5 Tensor.diag gives the trace
       n=6; store = sum(Tensor.Diag(range(1,n+1)).diag()); test = n*(n+1)/2
       Calculator.log(store == test, store)""",
    """# Test 6 Tensor.diag(vector) gives the dot product
       v=Tensor(1,2,3,4,5,6); store=sum(v.diag(v)); test=sum((x*x for x in v))
       Calculator.log(store == test, store)""",
       ]
  calc = Calculator(Real, Tests)
  calc.processInput(sys.argv)

elif sys.version_info.major != 2:  # Python 3
  def execfile(fName):
    """To match Python2's execfile need: from pathlib import Path
       exec(Path(fName).read_text())."""
    exec(Common.readText(fName))
