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
#    parser.add_argument('-L',metavar='L',dest='L',type=int,default=2,help='set L')
#    parser.add_argument('-L',metavar='L',dest='L',type=int,default=3,help='set L')
    parser.add_argument('-L',metavar='L',dest='L',type=int,default=4,help='set L')
#    parser.add_argument('-L',metavar='L',dest='L',type=int,default=5,help='set L')
    parser.add_argument('-Lorb',metavar='Lorb',dest='Lorb',type=int,default=2,help='set Lorb')
    parser.add_argument('-filling',metavar='filling',dest='filling',type=np.float64,default=1.0,help='set filling')
    parser.add_argument('-phasex',metavar='phasex',dest='phasex',type=np.float64,default=0.0,help='set phase x')
#    parser.add_argument('-phasey',metavar='phasey',dest='phasey',type=np.float64,default=0.0,help='set phase y')
    parser.add_argument('-phasey',metavar='phasey',dest='phasey',type=np.float64,default=0.5,help='set phase y')
    parser.add_argument('-g',metavar='v0',dest='v0',type=np.float64,default=0.0,help='set Gutzwiller v0')
    parser.add_argument('-v',metavar='v1',dest='v1',type=np.float64,default=0.0,help='set n.n. Jastrow v1')
    parser.add_argument('-output',metavar='output_file',dest='output_file',type=str,default='zqp_ipt.def',help='set output file name')
    return parser.parse_args()

## output header
def print_header(output_file,L,Lorb):
    f = open(output_file,'w')
    Ns = L*L*Lorb*Lorb
    f.write('========================\n')
    f.write('NOrbitalAP            %d\n' % Ns)
    f.write('========================\n')
    f.write('========================\n')
    f.write('========================\n')
    f.close()

## output fij
def print_fij(output_file,L,Lorb,fij):
    f = open(output_file,'a')
    for j in range(L):
        for i in range(L):
#            for orb1 in range(Lorb):
#                for orb2 in range(Lorb):
            for orb1 in range(Lorb-1,-1,-1):
                for orb2 in range(Lorb-1,-1,-1):
#                    f.write('%3d %23.16e %23.16e\n' % \
#                        (((j*L+i)*Lorb+orb1)*Lorb+orb2,fij[j,i,orb1,orb2].real,fij[j,i,orb1,orb2].imag))
                    f.write('%3d %23.16e %23.16e\n' % \
                        (((j*L+i)*Lorb+(1-orb1))*Lorb+(1-orb2),fij[j,i,orb1,orb2].real,fij[j,i,orb1,orb2].imag))
    f.close()

## output wf
def print_wf(output_file,L,Lorb,ene,v0,v1,fij):
    f = open(output_file,'w')
    ## <H>
    f.write('%23.16e %23.16e 0.0 ' % (ene,0.0))
    ## <H^2>
    f.write('%23.16e %23.16e 0.0 ' % (ene*ene,0.0))
    ## v0
##    NPg = 1
    NPg = Lorb
    for i in range(NPg):
        f.write('%23.16e %23.16e 0.0 ' % (v0,0.0))
    ## v1
##        NPj = L*L/2+1
    # honycomb case
    if L%2 == 1:
        NPj = L*L*Lorb-1
    else:
        NPj = L*L*Lorb+2
#
    for i in range(NPj):
        if i==0:
            f.write('%23.16e %23.16e 0.0 ' % (v1,0.0))
        else:
            f.write('%23.16e %23.16e 0.0 ' % (0.0,0.0))
    ## fij
#    for j in range(L):
#        for i in range(L):
#            f.write('%23.16e %23.16e 0.0 ' % (fij[i,j].real,fij[i,j].imag))
    for j in range(L):
        for i in range(L):
#            for orb1 in range(Lorb):
#                for orb2 in range(Lorb):
            for orb1 in range(Lorb-1,-1,-1):
                for orb2 in range(Lorb-1,-1,-1):
                    f.write('%23.16e %23.16e 0.0 ' % \
                        (fij[j,i,orb1,orb2].real,fij[j,i,orb1,orb2].imag))
##                    print('%3d %3d %3d %3d %3d %23.16e %23.16e 0.0 ' % \
##                        (j,i,orb1,orb2,((j*L+i)*Lorb+orb1)*Lorb+orb2,fij[j,i,orb1,orb2].real,fij[j,i,orb1,orb2].imag))
#                    print('%3d %3d %3d %3d %3d %23.16e %23.16e 0.0 ' % \
#                        (j,i,orb1,orb2,((j*L+i)*Lorb+(1-orb1))*Lorb+(1-orb2),fij[j,i,orb1,orb2].real,fij[j,i,orb1,orb2].imag))
    f.close()

## calc fij
#@jit(nopython=True)
#def func_enek(kx,ky,orb):
def func_enek(kx,ky):
    enecmplx = -(1.0 + np.exp(-1j*kx) + np.exp(-1j*ky))
#    ene = np.abs(enecmplx)
#    sign = 1.0-2.0*orb
#    return sign*ene
    ham = np.array([[0,enecmplx],[np.conjugate(enecmplx),0]])
    ene, vec = np.linalg.eigh(ham)
#    print("kx,ky,ene",kx,ky,ene)
#    print("vec",vec[:,0],vec[:,1])
#    return ene, vec
    return ene, vec.transpose(1,0)

#@jit(nopython=True)
def calc_enek(phasex,phasey,L,Lorb):
    enek = np.zeros((L,L,Lorb),dtype=np.float64)
    veck = np.zeros((L,L,Lorb,Lorb),dtype=np.complex128)
    for intkx in range(L):
        kx = (intkx+phasex) * 2.0*np.pi/float(L)
        for intky in range(L):
            ky = (intky+phasey) * 2.0*np.pi/float(L)
#            for orb in range(Lorb):
#                enek[intkx,intky,orb] = func_enek(kx,ky,orb)
            enek[intkx,intky], veck[intkx,intky] = func_enek(kx,ky)
    return enek, veck

#@jit(nopython=True)
def find_minus_k(intkx,intky,phasex,phasey,L):
    intminuskx = int((-(intkx+phasex)+L+0.0001)%L)
    intminusky = int((-(intky+phasey)+L+0.0001)%L)
    return intminuskx, intminusky

#@jit(nopython=True)
def calc_fk(L,Lorb,filling,enek,veck,phasex,phasey):
    Ne = int(L*L*Lorb*filling*0.5+0.0001)
    fk = np.zeros((L,L,Lorb,Lorb),dtype=np.complex128)
#    enesort = sorted(enek.ravel())
    enesort = sorted(enek.flatten())
#    print(enesort)
#    print(enesort[Ne-1],enesort[Ne])
    mu = 0.5*(enesort[Ne-1]+enesort[Ne])
    gap = -enesort[Ne-1]+enesort[Ne]
    enesum = 0.0
    for intkx in range(L):
        for intky in range(L):
            intmkx, intmky = find_minus_k(intkx,intky,phasex,phasey,L)
#            print(intkx,"-->",intmkx," ",intky,"-->",intmky)
            for orb1 in range(Lorb):
                if enek[intkx,intky,orb1] <= mu:
                    enesum += enek[intkx,intky,orb1]
                    for orb2 in range(Lorb):
                        for orb3 in range(Lorb):
                            fk[intkx,intky,orb2,orb3] = veck[intkx,intky,orb1,orb2] * veck[intmkx,intmky,orb1,orb3]
    return fk,mu,gap,enesum*2.0

#@jit(nopython=True)
def calc_fij(phasex,phasey,L,Lorb,fk):
    fij = np.zeros((L,L,Lorb,Lorb),dtype=np.complex128)
    for intkx in range(L):
        kx = (intkx+phasex) * 2.0*np.pi/float(L)
        for intky in range(L):
            ky = (intky+phasey) * 2.0*np.pi/float(L)
            for j in range(L):
                for i in range(L):
                    for orb1 in range(Lorb):
                        for orb2 in range(Lorb):
                            fij[j,i,orb1,orb2] += fk[intkx,intky,orb1,orb2] * np.exp(1j*(kx*i+ky*j))
    return fij*8.0/L/L
#    return fij/L/L

## main
def main():
    args = parse_args()
    L = args.L
    Lorb = args.Lorb
    filling = args.filling
    phasex = args.phasex
    phasey = args.phasey
    v0 = args.v0
    v1 = args.v1
    output_file = args.output_file

    enek, veck = calc_enek(phasex,phasey,L,Lorb)
    fk,mu,gap,enesum = calc_fk(L,Lorb,filling,enek,veck,phasex,phasey)
    fij = calc_fij(phasex,phasey,L,Lorb,fk)
#    print(enek)
#    print(fk)
    print('L',L)
    print('Lorb',Lorb)
    print('Ns',L*L*Lorb)
    print('Nup,Ndn',int(L*L*Lorb*filling*0.5+0.0001))
    print('Ne',int(L*L*Lorb*filling*0.5+0.0001)*2)
    print('phasex',phasex)
    print('phasey',phasey)
    print('mu',mu)
    print('gap',gap)
    print('ene',enesum)
#    print(fij)
#    print_header(output_file,L,Lorb)
#    print_fij(output_file,L,Lorb,fij)
    print_wf(output_file,L,Lorb,enesum,v0,v1,fij)

if __name__ == "__main__":
    main()
