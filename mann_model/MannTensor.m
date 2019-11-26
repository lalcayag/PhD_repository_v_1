function [Phi,Phi11,Phi22,Phi33] = MannTensor(k1,k2,k3,gamma_par,L,alphaepsilon,ElementChoice)

if nargin == 6
    ElementChoice = 'All';
end

%%%--- Definition of the wave vector and simplified kL
k=sqrt(k1.^2+k2.^2+k3.^2);
kL=k.*L;

%%%--- Eq. (3.6) in Mann (1994)
Beta=gamma_par.*(kL.^(-2/3)).*(HypergeometricFunction(1/3,17/6,4/3,-(kL.^(-2)),1,50).^(-1/2));
%%%--- Matlab-own hypergeom function (too slow)
%Beta=gamma_par.*(kL.^(-2/3)).*(hypergeom([1/3 17/6],4/3,-(kL.^(-2))).^(-1/2));

%%%--- Eq. (3.13) in Mann (1994)
k30=k3+Beta.*k1;
k0=sqrt(k1.^2+k2.^2+k30.^2);
kL0=k0.*L;

%%%--- Eq. (2.17) in Mann (1994)
Ek0=alphaepsilon.*(L.^(5/3))*(kL0.^4)./((1+kL0.^2).^(17/6));

%%%--- Eq. (3.16) in Mann (1994)
C1=Beta.*(k1.^2).*(k0.^2-2*(k30.^2)+Beta.*k1.*k30)./((k.^2).*(k1.^2+k2.^2));

%%%--- Eq. (3.17) in Mann (1994)
C2=(k2.*(k0.^2)./((k1.^2+k2.^2).^(3/2))).*...
    atan2(Beta.*k1.*sqrt(k1.^2+k2.^2),(k0.^2)-k30.*k1.*Beta);

%%%--- Eq. (3.15) in Mann (1994)
zeta1=C1-(k2./k1).*C2;
zeta2=(k2./k1).*C1+C2;

switch ElementChoice
    case 11
        %%%--- Eq. (3.19) in Mann (1994)
        Phi=(Ek0./(4*pi*(k0.^4))).*(k0.^2-k1.^2-2*k1.*k30.*zeta1+(k1.^2+k2.^2).*(zeta1.^2));
    case 22
        %%%--- Eq. (3.20) in Mann (1994)
        Phi=(Ek0./(4*pi*(k0.^4))).*(k0.^2-k2.^2-2*k2.*k30.*zeta2+(k1.^2+k2.^2).*(zeta2.^2));
    case 33
        %%%--- Eq. (3.18) in Mann (1994)
        Phi=(Ek0./(4*pi*(k.^4))).*(k1.^2+k2.^2);
    case 12
        %%%--- Eq. (3.21) in Mann (1994)
        Phi=(Ek0./(4*pi*(k0.^4))).*(-k1.*k2-k1.*k30.*zeta2-k2.*k30.*zeta1+(k1.^2+k2.^2).*zeta1.*zeta2);
    case 13
        %%%--- Eq. (3.22) in Mann (1994)
        Phi=(Ek0./(4*pi*(k0.^2).*(k.^2))).*(-k1.*k30+(k1.^2+k2.^2).*zeta1);
    case 23
        %%%--- Eq. (3.23) in Mann (1994)
        Phi=(Ek0./(4*pi*(k0.^2).*(k.^2))).*(-k2.*k30+(k1.^2+k2.^2).*zeta2);
    otherwise
        Phi11=(Ek0./(4*pi*(k0.^4))).*(k0.^2-k1.^2-2*k1.*k30.*zeta1+(k1.^2+k2.^2).*(zeta1.^2));
        Phi22=(Ek0./(4*pi*(k0.^4))).*(k0.^2-k2.^2-2*k2.*k30.*zeta2+(k1.^2+k2.^2).*(zeta2.^2));
        Phi33=(Ek0./(4*pi*(k.^4))).*(k1.^2+k2.^2);

        Phi12=(Ek0./(4*pi*(k0.^4))).*(-k1.*k2-k1.*k30.*zeta2-k2.*k30.*zeta1+(k1.^2+k2.^2).*zeta1.*zeta2);
        Phi13=(Ek0./(4*pi*(k0.^2).*(k.^2))).*(-k1.*k30+(k1.^2+k2.^2).*zeta1);
        Phi23=(Ek0./(4*pi*(k0.^2).*(k.^2))).*(-k2.*k30+(k1.^2+k2.^2).*zeta2);

        if nargout == 1
            Phi = [Phi11; Phi22; Phi33; Phi12; Phi13; Phi23];                
        else
            Phi = [];
        end
end