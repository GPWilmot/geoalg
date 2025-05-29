# Geometric Algebra Calculators - Quaternions, Octonions, Sedenions and Clifford Algrebra (geoalg)

This project provides 5 calculators interpreted by Python to allow basis elements such as *i, j, k, e<sub>1</sub>, e<sub>2</sub>, e<sub>3</sub>* to be entered at the prompt:
* calcCA - Clifford Algebra calculator to understand and extend quaternions
* calcQ - Quaternion calculator that covers complex numbers
* calcO - Octonion extended calculator includes split octonions and sedenions to multiple levels
* calcR - Real numbers only but the basis for other calculators along with calcLib
* calcS - Symbolic expansion only used to test n-D CA rotations and sedenion products

Any calculator can be started from the command line or commands entered directly. For example
*  ./calcQ.py test          # Runs all quaternion test code
*  python3 calcR.py help    # Displays top level command help

Commands include maintaining macro files and switching to other calculators from the prompt. This has the advantage of enabling comparison between different algebras, especially for testing the defined tests in the first calculator. The calculators interpret basis numbers and change them into relevant classes. eg
*      2 + i = Q(2, 1)  and  2 + e1 + e12 = CA(2, 1, e12=1)

These basis classes offer libraries to perform many functions. Foremost is rotations and due to the axial formulation of quaternion rotations and the confusion between vectors and versors, these are easier to understand in CA and then convert back to Q. 

The common library has the following classes:
* Lib    - Physics parameters and utilities such as permutation and combination expansions
* Matrix - Interface to numpy if it exists otherwise set to Tensor
* Tensor - Simple list based 1 & 2-D matricies for testing and basis multiplication tables
* Euler  - Multi-dimensional Euler angles to rotation matries in any order, optionally implicit

The Matrix class interfaces to numpy, if it exists, but without numpy it provides enough functionality to compare rotations defined by matrix exponentiation to Q and CA results. The Euler class provides n-D rotations in arbitrary order and implicit or explicit. These can generate n-D matrix representations and can be converted back into CA algebra (or Q algebra, if 3-D) for explicit rotations in standard order. The matrix operations without numpy use the Tensor class which are simple but are designed to allow for unit tests.

The Tensor class has this name because these matricies can contain basis numbers. Eg if g_lo = Tensor(e0,e1,e2,e3) and g_hi = Tensor(-e0,e1,e2,e3) then the metric tensor is (g_lo*g_hi.transpose()).scalar() = Tensor.Trace([1]*4). The Tensor class can also be used for the Dixon algebra (the product of real, complex, quaternion and octernion numbers) as Tensor(R, Q(1), Q, O). Addition is via + and multiplication via Tensor.diag(vector). All these algebras can be loaded with the command: calc(Q,CA,O) and exposed with the Basis method. Eg Q(1) = Q.Basis(1) is the quaternions.

O(3) is octonions which has 3 generators. O(4) is the sedenions (sedecim is Latin for sixteen) which are not only non-associative, like the octonions, but also alternate. This means the associator is not only non-zero it is also not anti-symmetric. For example, o1234.assoc(o234,o13)==0 but o1234.assoc (o13,o234)==2o3. This makes sedenions and all higher order octonions power-associative, whereby x.assoc(y,y)==0, for all x, y in O.Basis(n). Just like CA has both signatures, the octonions here have both signatures. O(2,1) defines the split-octonions with elements like o12u3 whereby u3*u3=1. Unity bases, u<hex>, have a similar multiplication table as octonions but positive diagonal elements for odd parity. O(1,2) and O(0,3) are also isomorphic to the split-octonions and other representations can be entered manually. Note that CA is a graded algebra with e123 being a 3-form but octonions are not graded and the hex indices are only used to invoke a plus sign in the Cayley-Dickson multiplication rule. The use of the graded notation allows the Cayley-Dickson rule to be automated and exposes the underlying structure with quaternions such as o1, o2, o12 easily recognised.

The octonions are included in this list of geometric algebras because it is provides a cross product in 7 dimensions. The calculations in spin7_g2.ca show how Clifford algegra simplifies the usual derivation. It goes further to show a commuting subset of Spin(7) can be used to define the Exceptional Lie algebra G2 which are the automorphisms of the octonions. This is easy to see using Clifford algebra. The file "Book-The Algebra of Geometry.pdf" shows a new way to derive and understand Clifford algebra from simplices such as the triangle and tetrahedron. My claim is that I can get you to think in 5 dimensions in five minutes but 7 dimensions takes a bit longer. The book is a work in progress and includes only Chapters 1 and 2 with an outline of the others. Pfaffians needed to be added to chapter 1 because of their close connection to simplices. The Spin(7) and Spin(15) chapters will document the connections to G2, octonions and sedenions now published as the preprint arXiv:2505.06011 [math.RA], shown in an earlier version in the file "Article-Construction of G2 using Clifford Algebra.pdf". This would be followed by chapters on ultracomplex numbers published as the preprint arXiv:2505.11747 and given by an early version in the file "Article-Structure of the Sedenion Series.pdf". 

In researching the history of Clifford algebra, like other authors I found this subject is richly documented and does not need repeating. In one of the articles I came across the statement "Long live the vector algebra war" which I took to heart. Realising this was written by researchers at Adelaide University, South Australia, I have now returned to studies in order to publish more research. I can be reached via gdotpdotwilmotzero at Google's facilities.

### Future Work:
* calcCA - Allow trigintadecimal(?) basis (hex+) numbers (0..V) to support CA(31)
* calcCA - Change to bitwise multiplication instead of string manipulation using previous CA work
* calcO - Fix deriving matrices for rotations and Euler angles for 3-D
* Lib - Combinations optimisation using previous CA work

