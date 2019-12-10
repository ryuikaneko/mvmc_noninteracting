#!/bin/bash

prog1=./vmcdry.out
prog2=./vmc.out

python2.7 ./make_init.py | tee dat_exact_U0
${prog1} StdFace.def
## just in case, use complex fij
mv orbitalidx.def orbitalidx.def.orig
sed 's/^ComplexType.*/ComplexType 1/g' orbitalidx.def.orig > orbitalidx.def
##
mpiexec -np 1 ${prog2} namelist.def zqp_ipt.def
