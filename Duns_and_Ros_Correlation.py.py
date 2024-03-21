import streamlit as st
from math import log, exp, pi

from math import log, inf, exp, pi

def mass(WOR, SGo, SGw):
    a = SGo * 350 * (1 / (1 + WOR))
    b = SGw * 350 * (WOR / (1 + WOR))
    m = a + b
    return m

def den_liq(WOR, SGo, SGw):
    a = SGo * 62.4 * (1 / (1 + WOR))
    b = SGw * 62.4 * (WOR / (1 + WOR))
    m = a + b
    return m

def avg_p(P1, P2):
    p = (P1 + P2) / 2 + 14.7
    return p

def avg_t(T1, T2):
    T = ((T1) + (T2)) / 2
    return T

def Z(P1, P2, T1, T2, SGg):
    P = avg_p(P1, P2)
    T = avg_t(T1, T2) + 460
    Ppc = 677 + 15.0 * SGg + 37.5 * SGg ** 2
    Tpc = 168 + 325 * SGg + 12.5 * SGg ** 2
    Ppr = P / Ppc
    Tpr = T / Tpc
    A = 1.39 * (Tpr - 0.92) ** 0.5 - 0.36 * Tpr - 0.101
    B = (0.62 - 0.23 * Tpr) * Ppr + ((0.066 / (Tpr - 0.86)) - 0.037) * Ppr ** 2 + (0.32 / (10 ** 9 * (Tpr - 1))) * Ppr ** 6
    C = 0.132 - 0.32 * (log(Tpr))
    K = 0.3106 - 0.9 * Tpr + 0.1824 * Tpr ** 2
    D = 10 ** K
    z = A + (1 - A) * exp(-B) + C * Ppr ** D
    return z

def avg_den_gas(SGg, P1, P2, T1, T2):
    d = SGg * 0.0764 * (avg_p(P1, P2) / 14.7) * (520 / (avg_t(T1, T2) + 460)) * (1 / Z(P1, P2, T1, T2, SGg))
    return d

def water_vis(T1, T2):
    from math import exp
    vis = exp(1.003 - 1.479 * 10 ** - 2 * avg_t(T1, T2) + 1.982 * 10 ** -5 * (avg_t(T1, T2) + 460) ** 2)
    return vis

def liq_vis(T1, T2, API, WOR):
    vis = oil_vis(T1, T2, API) * (1 / (1 + WOR)) + water_vis(T1, T2) * (WOR / (1 + WOR))
    return vis

def oil_vis(T1, T2, API):
    Z = 3.0324 - (0.02023 * API)
    Y = 10 ** Z
    X = Y * ((avg_t(T1, T2) - 460) ** -1.163)
    vis = max(((10 ** X) - 1).real, 0)
    return vis
    
def oil_st(P1,P2,T1,T2,API): 
  P = avg_p(P1,P2) 
  T = avg_t(T1,T2)+460
  C=1-0.024*P**0.45 
  if P>5000: 
    st = 0 
  elif P>3997: 
    st=1 
  elif T<=68: 
    st=(39-0.2571*API)*C 
  elif T<=100: 
    st = -1.5*C*(T-68)/32 + (39-0.2571*API)*C 
  else: 
    st = (37.5-0.2571*API)*C 
  return st 

def water_st(T1,T2,P1,P2): 
  T = avg_t(T1,T2)+460 
  P = avg_p(P1,P2) 
  if T<=74: 
    st = 75 - 1.108*P**0.349 
  elif T>280: 
    st = 53 - 0.1048*P**0.037 
  else: 
    st = ((53 - 0.1048*P**0.037)-(75 - 1.108*P**0.349))*(T-74)/206 + (75 - 1.108*P**0.349)
  return st

def liq_st(WOR,P1,P2,T1,T2,API):
    o = oil_st(P1,P2,T1,T2,API)
    w = water_st(P1,P2,T1,T2)
    st = o * (1 + (1 + WOR)) + w * (WOR + (1 + WOR))
    return st

def RS(SGg, API, T1, T2, P1, P2):
    x = 0.0125 * API - 0.0009 * (avg_t(T1, T2) + 460)
    RS = SGg * ((avg_p(P1, P2) / 18.2 + 1.4) * 10 ** x) ** 1.2048
    return RS

def Bo(SGg, API, T1, T2, P1, P2, SGo):
    Bo = 0.9759 + 0.000120 * (
            (RS(SGg, API, T1, T2, P1, P2) * (SGg / SGo) ** 0.5) + 1.25 * (avg_t(T1, T2) + 460)) ** 1.2
    return Bo

def area(D):
    from math import pi
    area = pi * D ** 2 / 4
    return area

def liq_vis_no(T1, T2, API, WOR, SGo, SGw):
    Nl = 0.15726 * liq_vis(T1, T2, API, WOR) * (1 / (den_liq(WOR, SGo, SGw) ** 3)) ** 0.25
    return Nl

def VSL(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, API, Bw=1):
    a = 5.61 * (Qo + Qw) / (86400 * area(D))
    b = Bo(SGg, API, T1, T2, P1, P2, SGo) * (1 / (1 + WOR))
    c = Bw * (WOR / (1 + WOR))
    vsl = a * (b + c)
    return vsl

def VSG(Qg, D):
    vsg = Qg / area(D)
    return vsg

def LVN(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, SGw,API, Bw=1):
    a = VSL(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, API, Bw=1)
    b = den_liq(WOR, SGo, SGw)
    c = liq_st(WOR,P1,P2,T1,T2,API)
    lvn = abs(1.938 * a * (b / c) ** 0.25)
    return lvn

def NGV(Qo, Qw, GLR, SGo, SGg, API, T1, T2, P1, P2, WOR,Qg,D,SGw):
    Ngv = 1.98 * VSG(Qg, D) * abs((den_liq(WOR, SGo, SGw) / liq_vis(T1, T2, API, WOR))) ** 0.25
    return Ngv

def Nd(D, WOR, SGo, SGw, T1, T2, API):
    Nd = 120.872 * D * ((den_liq(WOR, SGo, SGw) / liq_vis(T1, T2, API, WOR)) ** 0.5)
    return Nd

def L1L2(D, WOR, SGo, SGw, T1, T2, API):
    nd = Nd(D, WOR, SGo, SGw, T1, T2, API)
    if nd < 40:
        L1 = 2
        L2 = 0.1 * nd + 0.25
    elif 40 <= nd < 70:
        L1 = (19 / 6) - (1 / 30 * nd)
        L2 = 1
    else:
        L1 = 0.9
        L2 = 1.1
    return [L1, L2]

def Regime(D, WOR, SGo, SGw, T1, T2, API,Qo,Qw,GLR,SGg,P1,P2,Qg):
    L1 = L1L2(D, WOR, SGo, SGw, T1, T2, API)[0]
    L2 = L1L2(D, WOR, SGo, SGw, T1, T2, API)[1]
    ngv = NGV(Qo, Qw, GLR, SGo, SGg, API, T1, T2, P1, P2, WOR,Qg,D,SGw)
    nlv = LVN(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, SGw,API, Bw=1)
    try:
        if ngv < (L1 + L2 * nlv):
            a = 'Bubble'
        elif ngv < (L1 + L2 * nlv) and ngv < (50 + 36 * nlv):
            a = 'Slug'
        else:
            a = 'Mist'
    except:
        a = 'Mist'
    return a

def holdup(D,WOR,SGo,SGw,T1,T2,API,P1,P2,Qg,Qo,Qw,SGg,GLR):
    a = Regime(D, WOR, SGo, SGw, T1, T2, API,Qo,Qw,GLR,SGg,P1,P2,Qg)
    dg = (avg_p(P1, P2) * 28.96 * SGg) / (10.73 * (avg_t(T1, T2) + 460))
    dl = den_liq(WOR, SGo, SGw)
    vsg = VSG(Qg, D)
    Vm = VSL(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, API, Bw=1) + vsg
    from math import exp
    if a == 'Bubble':
        vbs = 1.41 * (oil_st(P1,P2,T1,T2,API) * (dl - dg)) ** 0.25
        Vbf = 1.2 * Vm + vbs
        Hl = 1 - (vsg / Vbf)
    elif a == 'Slug':
        ne = 32.4 * D ** 2 * (dl - dg) / liq_st(WOR)
        nv = (D ** 3 * dl * (dl - dg) / liq_vis(T1, T2, API, WOR)) ** 0.5
        if nv >= 250:
            m = 10
        elif nv <= 18:
            m = 25
        else:
            m = 69 * nv ** -0.35
        C = 0.345 * (1 - exp(-0.029 * nv)) * (1 - exp((3.37 - ne) / m))
        vbs = C * (32.4 * D * (dl - dg) / dl) ** 0.5
        Vbf = 1.2 * Vm + vbs
        Hl = 1 - (vsg / Vbf)
    else:
        Hl = 1 / (1 + vsg / VSL(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, API, Bw=1))
    return Hl


# The rest of the functions are not provided in the original code and need to be defined
# before running the `press_drop()` function.

def Gm(Qo,Qw,P1,P2,SGg,T1,T2,WOR,SGo,SGw,Qg,D): 
    Ql = Qo + Qw 
    dg = (avg_p(P1, P2) * 28.96 * SGg) / (10.73 * avg_t(T1, T2)) 
    gl = den_liq(WOR, SGo, SGw) * Ql / area(D) 
    gg = dg * Qg / area(D) 
    gm = gl + gg 
    return gm 

def NRE(D,T1,T2,API,Qo,Qw,P1,P2,SGg,WOR,SGo,SGw,Qg): 
    nre = Gm(Qo,Qw,P1,P2,SGg,T1,T2,WOR,SGo,SGw,Qg,D) * D / liq_vis(T1, T2, API, WOR) 
    return nre 

def Slip(Qo,Qw,l,Qg,D,WOR,SGo,SGw,T1,T2,API,P1,P2,SGg,GLR,e=0): 
    from math import log, inf, exp 
    Ql = Qo + Qw 
    L = l / (Ql + Qg) 
    y = holdup(D,WOR,SGo,SGw,T1,T2,API,P1,P2,Qg,Qo,Qw,SGg,GLR)**2 
    if y > 1.2: 
        s = log(2.2 * y - 2) 
    else: 
        s = log(y) / (-0.0523 + 3.182 * log(y) - 0.8725 * log(y)**2 + 0.01853 * log(y)**4) 
    fc = inf 
    fest = 0.001 
    while abs(fc - fest) > 0.0001: 
        fc = (1.74 - 2 * log(2 * (e / D) + (18.7 / (NRE(D,T1,T2,API,Qo,Qw,P1,P2,SGg,WOR,SGo,SGw,Qg) * fest**0.5))))**-2 
        fest = (fc + fest) / 2 
    ftp = exp(s) * fc 
    return ftp 

def frictional_press_drop(P1,P2,SGg,WOR,SGo,SGw,Qg,D,Qo,Qw,T1,T2,API,GLR,l): 
    dg = (avg_p(P1, P2) * 28.96 * SGg) / (10.73 * (460 + avg_t(T1, T2))) 
    dl = den_liq(WOR, SGo, SGw) 
    vsg = VSG(Qg, D) 
    vsl = VSL(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, API, Bw=1) 
    hl = holdup(D,WOR,SGo,SGw,T1,T2,API,P1,P2,Qg,Qo,Qw,SGg,GLR) 
    ftp = Slip(Qo,Qw,l,Qg,D,WOR,SGo,SGw,T1,T2,API,P1,P2,SGg,GLR,e=0) 
    Vm = VSL(Qo, Qw, WOR, D, SGg, SGo, T1, T2, P1, P2, API, Bw=1) + vsg 
    G = dl * vsl + dg * vsg 
    dp = (dl * hl + (1 - hl) * dg + (ftp * G * Vm) / (2 * 32.2 * D)) / 144  * l
    return dp

def gravitational_press_drop(WOR, SGo, SGw, l):
    p_drop = den_liq(WOR, SGo, SGw) * l / 144
    return p_drop 
    

# Create a Streamlit app
def main():
    st.title('Pressure Drop Calculator')

    Qo = st.number_input('Oil flow rate(in stb/day)', value=1000)
    Qw = st.number_input('Water flow rate(in stb/day)', value=500)
    Qg = st.number_input('Gas flow rate(in stb/day)', value=200)
    GLR = st.number_input('Gas-Liquid Ratio(in scf/stb)', value=5000)
    SGo = st.number_input('Oil Specific Gravity', value=0.8)
    SGg = st.number_input('Gas Specific Gravity', value=0.65)
    SGw = st.number_input('Water Specific Gravity', value=1)
    API = st.number_input('API(in degrees)', value=30)
    T1 = st.number_input('Temperature at tubing entrance(in degrees Fahrenheit)', value=70)
    T2 = st.number_input('Temperature at end of tubing(in degrees Fahrenheit)', value=80)
    P1 = st.number_input('Pressure at the tubing entrance(in psia)', value=2000)
    P2 = st.number_input('Pressure at the end of tubing(in psia)', value=1500)
    l = st.number_input('Length of tubing(in feet)', value=100)
    WOR = Qw / Qo
    D = st.number_input('Tubing Diameter(in inches)', value=10)

    if st.button('Calculate'):
        frictional_losses= int(frictional_press_drop(P1,P2,SGg,WOR,SGo,SGw,Qg,D,Qo,Qw,T1,T2,API,GLR,l))
        st.write(f'Pressure Drop due to friction: {frictional_losses} psia')
        gravity_losses = int(gravitational_press_drop(WOR, SGo, SGw, l))
        st.write(f'Pressure Drop due to gravity: {gravity_losses} psia')
        total_loss = frictional_losses + gravity_losses
        st.success(f'Total Pressure Drop: {total_loss} psia')
if __name__ == '__main__':
    main()
