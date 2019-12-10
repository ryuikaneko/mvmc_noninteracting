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
    parser.add_argument('-L',metavar='L',dest='L',type=int,default=4,help='set L')
    parser.add_argument('-filling',metavar='filling',dest='filling',type=np.float64,default=1.0,help='set filling')
    parser.add_argument('-phasex',metavar='phasex',dest='phasex',type=np.float64,default=0.0,help='set phase x')
    parser.add_argument('-phasey',metavar='phasey',dest='phasey',type=np.float64,default=0.5,help='set phase y')
    parser.add_argument('-g',metavar='v0',dest='v0',type=np.float64,default=0.0,help='set Gutzwiller v0')
    parser.add_argument('-v',metavar='v1',dest='v1',type=np.float64,default=0.0,help='set n.n. Jastrow v1')
    parser.add_argument('-output',metavar='output_file',dest='output_file',type=str,default='zqp_ipt.def',help='set output file name')
    return parser.parse_args()

## output header
def print_header(output_file,L):
    f = open(output_file,'w')
    Ns = L*L
    f.write('========================\n')
    f.write('NOrbitalAP            %d\n' % Ns)
    f.write('========================\n')
    f.write('========================\n')
    f.write('========================\n')
    f.close()

## output fij
def print_fij(output_file,L,fij):
    f = open(output_file,'a')
    for j in range(L):
        for i in range(L):
            f.write('%3d %23.16e %23.16e\n' % \
                (j*L+i,fij[i,j].real,fij[i,j].imag))
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
    NPj = L*L/2+1
    for i in range(NPj):
        if i==0 or i== L/2 or i== L/2+1:# triangular
#        if i==0 or i== L/2:# square
            f.write('%23.16e %23.16e 0.0 ' % (v1,0.0))
        else:
            f.write('%23.16e %23.16e 0.0 ' % (0.0,0.0))
    ## fij
    for j in range(L):
        for i in range(L):
            f.write('%23.16e %23.16e 0.0 ' % (fij[i,j].real,fij[i,j].imag))
    f.close()

## calc fij
#@jit(nopython=True)
def func_enek(kx,ky):
    return -2.0*(np.cos(kx)+np.cos(ky)+np.cos(kx-ky))# triangular
#    return -2.0*(np.cos(kx)+np.cos(ky))# square

#@jit(nopython=True)
def calc_enek(phasex,phasey,L):
    enek = np.zeros((L,L),dtype=np.float64)
    for intkx in range(L):
        kx = (intkx+phasex) * 2.0*np.pi/float(L)
        for intky in range(L):
            ky = (intky+phasey) * 2.0*np.pi/float(L)
            enek[intkx,intky] = func_enek(kx,ky)
    return enek

#@jit(nopython=True)
def calc_fk(L,filling,enek):
    Ne = int(L*L*filling*0.5+0.0001)
    fk = np.zeros((L,L),dtype=np.float64)
#    enesort = sorted(enek.ravel())
    enesort = sorted(enek.flatten())
#    print(enesort)
#    print(enesort[Ne-1],enesort[Ne])
    mu = 0.5*(enesort[Ne-1]+enesort[Ne])
    gap = -enesort[Ne-1]+enesort[Ne]
    enesum = 0.0
    for intkx in range(L):
        for intky in range(L):
            if enek[intkx,intky] <= mu:
                fk[intkx,intky] = 1.0
                enesum += enek[intkx,intky]
            else:
                fk[intkx,intky] = 0.0
    return fk,mu,gap,enesum*2.0

#@jit(nopython=True)
def calc_fij(phasex,phasey,L,fk):
    fij = np.zeros((L,L),dtype=np.complex128)
    for intkx in range(L):
        kx = (intkx+phasex) * 2.0*np.pi/float(L)
        for intky in range(L):
            ky = (intky+phasey) * 2.0*np.pi/float(L)
            for i in range(L):
                for j in range(L):
                    fij[i,j] += fk[intkx,intky] * np.exp(1j*(kx*i+ky*j))
    return fij*8.0/L/L
#    return fij/L/L

## main
def main():
    args = parse_args()
    L = args.L
    filling = args.filling
    phasex = args.phasex
    phasey = args.phasey
    v0 = args.v0
    v1 = args.v1
    output_file = args.output_file

    enek = calc_enek(phasex,phasey,L)
    fk,mu,gap,enesum = calc_fk(L,filling,enek)
    fij = calc_fij(phasex,phasey,L,fk)
#    print(enek)
#    print(fk)
    print('L',L)
    print('Ns',L*L)
    print('Nup,Ndn',int(L*L*filling*0.5+0.0001))
    print('Ne',int(L*L*filling*0.5+0.0001)*2)
    print('phasex',phasex)
    print('phasey',phasey)
    print('mu',mu)
    print('gap',gap)
    print('ene',enesum)
#    print(fij)
#    print_header(output_file,L)
#    print_fij(output_file,L,fij)
    print_wf(output_file,L,enesum,v0,v1,fij)

if __name__ == "__main__":
    main()
