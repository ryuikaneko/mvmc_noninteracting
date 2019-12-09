#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import numpy as np
#import scipy as scipy
#import scipy.linalg as linalg
#import scipy.sparse.linalg as spr_linalg
#import time
import argparse
#from numba import jit

def parse_args():
    parser = argparse.ArgumentParser(description='prepare zqp_ipt.def')
#    parser.add_argument('-L',metavar='L',dest='L',type=int,default=14,help='set L')
#    parser.add_argument('-filling',metavar='filling',dest='filling',type=np.float64,default=1.0,help='set filling')
#    parser.add_argument('-phasex',metavar='phasex',dest='phasex',type=np.float64,default=0.0,help='set phase x')
    parser.add_argument('-L',metavar='L',dest='L',type=int,default=12,help='set L')
    parser.add_argument('-filling',metavar='filling',dest='filling',type=np.float64,default=0.5,help='set filling')
    parser.add_argument('-phasex',metavar='phasex',dest='phasex',type=np.float64,default=0.0,help='set phase x')
    parser.add_argument('-g',metavar='v0',dest='v0',type=np.float64,default=0.0,help='set Gutzwiller v0')
    parser.add_argument('-v',metavar='v1',dest='v1',type=np.float64,default=0.0,help='set n.n. Jastrow v1')
    parser.add_argument('-output',metavar='output_file',dest='output_file',type=str,default='zqp_ipt.def',help='set output file name')
    return parser.parse_args()

## output header
def print_header(output_file,L):
    f = open(output_file,'w')
    Ns = L
    f.write('========================\n')
    f.write('NOrbitalAP            %d\n' % Ns)
    f.write('========================\n')
    f.write('========================\n')
    f.write('========================\n')
    f.close()

## output fij
def print_fij(output_file,L,fij):
    f = open(output_file,'a')
    for i in range(L):
        f.write('%3d %23.16e %23.16e\n' % \
            (i,fij[i].real,fij[i].imag))
    f.close()

## output wf
def print_wf(output_file,L,ene,v0,v1,fij):
    f = open(output_file,'w')
    ## <H>
    f.write('%23.16e %23.16e 0.0 ' % (ene,0.0))
    ## <H^2>
    f.write('%23.16e %23.16e 0.0 ' % (ene*ene,0.0))
    ## v0
    NPg = 1
    for i in range(NPg):
        f.write('%23.16e %23.16e 0.0 ' % (v0,0.0))
    ## v1
    NPj = L//2
    for i in range(NPj):
        if i==0:
            f.write('%23.16e %23.16e 0.0 ' % (v1,0.0))
        else:
            f.write('%23.16e %23.16e 0.0 ' % (0.0,0.0))
    ## fij
    for i in range(L):
        f.write('%23.16e %23.16e 0.0 ' % (fij[i].real,fij[i].imag))
    f.close()

## calc fij
#@jit(nopython=True)
def func_enek(kx):
    return -2.0*np.cos(kx)

#@jit(nopython=True)
def calc_enek(phasex,L):
    enek = np.zeros(L,dtype=np.float64)
    for intkx in range(L):
        kx = (intkx+phasex) * 2.0*np.pi/float(L)
        enek[intkx] = func_enek(kx)
    return enek

#@jit(nopython=True)
def calc_fk(L,filling,enek):
    Ne = int(L*filling*0.5+0.0001)
    fk = np.zeros(L,dtype=np.float64)
    enesort = sorted(enek)
#    print(enesort)
#    print(enesort[Ne-1],enesort[Ne])
    mu = 0.5*(enesort[Ne-1]+enesort[Ne])
    gap = -enesort[Ne-1]+enesort[Ne]
    enesum = 0.0
    for intkx in range(L):
        if enek[intkx] <= mu:
            fk[intkx] = 1.0
            enesum += enek[intkx]
        else:
            fk[intkx] = 0.0
    return fk,mu,gap,enesum*2.0

#@jit(nopython=True)
def calc_fij(phasex,L,fk):
    fij = np.zeros(L,dtype=np.complex128)
    for intkx in range(L):
        kx = (intkx+phasex) * 2.0*np.pi/float(L)
        for i in range(L):
            fij[i] += fk[intkx] * np.exp(1j*kx*i)
    return fij*8.0/L/L
#    return fij/L/L

## main
def main():
    args = parse_args()
    L = args.L
    filling = args.filling
    phasex = args.phasex
    v0 = args.v0
    v1 = args.v1
    output_file = args.output_file

    enek = calc_enek(phasex,L)
    fk,mu,gap,enesum = calc_fk(L,filling,enek)
    fij = calc_fij(phasex,L,fk)
#    print(enek)
#    print(fk)
    print('L',L)
    print('Ns',L)
    print('Nup,Ndn',int(L*filling*0.5+0.0001))
    print('Ne',int(L*filling*0.5+0.0001)*2)
    print('phasex',phasex)
    print('mu',mu)
    print('gap',gap)
    print('ene',enesum)
#    print(fij)
#    print_header(output_file,L)
#    print_fij(output_file,L,fij)
    print_wf(output_file,L,enesum,v0,v1,fij)

if __name__ == "__main__":
    main()
