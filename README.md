# Geometric Algebra Calculators - Quaternions, Octonions and Clifford Algrebra (geoalg)

This project provides 7 calculators interpreted by Python to allow basis elements such as *i, j, k, e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub>* to be entered at the prompt:
* calcCA - Clifford Algebra calculator to understand and extend quaternions
* calcQ - Quaternion calculator that includes complex numbers
* calcO - Octonion extended calculator includes split octonians and sedenions to multiple levels
* calcR - Real numbers only but the basis for other calculators along with calcCommon
* calcS - Symbolic expansion only used to test n-D CA rotations and sedenion products

Any calculator can be started from the command line or commands entered directly. For example
*  ./calcQ.py test         # Runs all quaternion test code
*  python calcR.py help    # Displays top level command help

Commands include maintaining macro files and switching to other calculators from the prompt. This has the advantage of enabling comparison between different algebras, especially for testing the defined tests in the first calculator. The calculators interpret basis numbers and change then into relevant classes. eg
*      2 + i = Q(2, 1)  and  2 + e1 + e12 = CA(2, 1, e12=1)

These basis classes offer libraries to perform many functions. Foremost is rotations and due to the axial formulation of quaternion rotations and the confusion between vectors and versors, these are easier to understand in CA and then convert back to Q. 

The common library the following classes:
* Common - Physics parameters and utilities such as permutation and combination expansions
* Matrix - Interface to numpy if it exists otherwise set to Tensor
* Tensor - Simple list based 1 & 2-D matricies for testing and basis multiplication tables
* Euler  - Multi-dimensional Euler angle to rotation matries in any order & intrisic or not

The Matrix class interfaces to numpy, if it exists, but without numpy provides enough functionality to compare rotations defined by matrix exponentiation to Q and CA results. The Euler class provides n-D rotations in arbitrary order and optionally implicit. These can generate n-D matrix representations and can be converted back into CA algebra (or Q algebra, if 3-D) for explicit rotations in standard order. The matrix operations without numpy use the Tensor class which are simple but are designed to allow for unit tests.

The Tensor class has this name because these matricies can contain basis numbers. Eg if g_lo = Tensor(e0,e1,e2,e3) and g_hi = Tensor(-e0,e1,e2,e3) then the metric tensor is (g_lo*g_hi.transpose()).scalar() = Tensor.Trace([1]*4). The Tensor class can also be used for the Dixon algebra (the product of real, complex, quaternion and octernion numbers) as Tensor(R, Q(1), Q, O). Addition is via + and multiplication via Tensor.diag(vector). All these algebras can be loaded with the command: calc(Q,CA,O) and exposed with the Basis method. Eg Q(1) = Q.Basis(1) is the quaternions.

O(3) is octonions which has 3 generators. O(4) is the Sedenions (sedecim is Latin for sixteen) which are not only non-associative, like the octonions, but also alternate. This means the associator is not only non-zero it is also not anti-symmetric. For example, o1234.assoc(o234,o13)==0 but o1234.assoc(o13,o234)==2o3. This makes Sedenions and all higher order octonions power-associative, whereby x.assoc(y,y)==0, for all x, y in O.Basis(n). Just like CA has both signatures, the octonions here have both signatures. O(2,1) defines the split octonions with elements like o12u1, the former being the quaternions and the later, split-octonians which use unit bases. Unity bases, u<hex>, have the same multiplication table as octonions but positive diagonal elements. These commute with the o<hex> numbers when entered. Note that CA is a graded algebra with e123 being a 3-form but octonions are not graded and the hex numbers are only used to invoke the general Cayley-Dickson multiplication rule.

The octonians are included in this list of geometric algebras because it is related to the cross product in 7 dimensions. The calculations in spin7_g2.ca show how Clifford algegra simplifies the usual derivation. It goes further to show a commuting subset of Spin(7) that defines the Lie Exceptional algebra G2 which are the automorphisms of the octonions. This is easy to see using Clifford algebra. The file "Book-The Algebra of Geometry" shows a new way to derive and understand Clifford algebra from simplices such as the triangle and tetrahedron. My claim is that I can get you to think in 5 dimensions in five minutes but 7 dimensions takes a bit longer. The book is a work in progress and includes only Chapter 1 at the moment but some of the chapters are outlined already and the Spin(7) and Spin(15) chapters will document the connections to G2, octonians and sedonions taken from the paper I am trying to publish.


### Future Work:
* calcCA - Allow duotridecimal numbers (0..V) to support CA(31)
* calcCA - Change to bit wise multiplication instead of string manipulation using previous CA work
* calcO - Use matrices to derive matrix rotations and O.Euler for > 3-D
* Common - Combinations optimisation using previous CA work

