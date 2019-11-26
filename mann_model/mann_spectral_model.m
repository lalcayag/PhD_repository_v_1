%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Mann spectral model by AP                                           %%%
%%% Do not distribute without authorization from AP                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear

%%%--- initialize the k vector
logkinput=log(0.0001):0.2:log(100);
kinput=exp(logkinput); 

%%%--- Mann parameters
Gamma_par=3;
L=50;
alphaepsilon=0.01;

%%%--- definition and setup of the wavenumber arrays
k1=kinput;
k2=fliplr(-k1);
k2(1,length(k2)+1)=0;
k2(1,length(k2)+1:length(k2)+length(k1))=k1;
k3=k2;
[k2grid,k3grid]=meshgrid(k2,k3);

%%%--- number of wave numbers
nk1=length(k1);

%%%--- initializing the Psi array
Psi=NaN*ones(4,nk1);

for ik=1:nk1
    disp(ik)
    Psi11K=MannTensor(k1(ik)*ones(size(k2grid)),ones(length(k3),1)*k2,k3'*ones(1,length(k2)),Gamma_par,L,alphaepsilon,11); 
    Psi22K=MannTensor(k1(ik)*ones(size(k2grid)),ones(length(k3),1)*k2,k3'*ones(1,length(k2)),Gamma_par,L,alphaepsilon,22);
    Psi33K=MannTensor(k1(ik)*ones(size(k2grid)),ones(length(k3),1)*k2,k3'*ones(1,length(k2)),Gamma_par,L,alphaepsilon,33);
    Psi13K=MannTensor(k1(ik)*ones(size(k2grid)),ones(length(k3),1)*k2,k3'*ones(1,length(k2)),Gamma_par,L,alphaepsilon,13);
     
    ksum11=trapz(k3,trapz(k2,Psi11K));
    ksum22=trapz(k3,trapz(k2,Psi22K));
    ksum33=trapz(k3,trapz(k2,Psi33K));
    ksum13=trapz(k3,trapz(k2,Psi13K));
    
    Psi(1,ik)=ksum11;
    Psi(2,ik)=ksum22;
    Psi(3,ik)=ksum33;
    Psi(4,ik)=ksum13;
end

%%%--- variance estimation
var11=2*trapz(k1,Psi(1,:));
var22=2*trapz(k1,Psi(2,:));
var33=2*trapz(k1,Psi(3,:));
var13=2*trapz(k1,Psi(4,:));

figure;
semilogx(k1,k1.*Psi(1,:),'k-','linewidth',1.5);
hold on
semilogx(k1,k1.*Psi(2,:),'r-','linewidth',1.5);
hold on
semilogx(k1,k1.*Psi(3,:),'b-','linewidth',1.5);
hold on
semilogx(k1,k1.*Psi(4,:),'g-','linewidth',1.5);
grid on
set(get(gca,'XLabel'),'Fontsize',14,'Interpreter','latex','String','$k_1$')
set(get(gca,'YLabel'),'Fontsize',14,'Interpreter','latex','String','$k_1~F(k_1)$')
hh=legend('$u$','$v$','$w$','$uw$');
set(hh,'Interpreter','latex')
title(['$\Gamma$=',num2str(Gamma_par),', $\alpha\varepsilon^{2/3}$=',num2str(alphaepsilon),', $L$=',num2str(L)],'Fontsize',14,'Interpreter','latex')
set(gca,'XTickLabel',{'0.0001','0.001','0.01','0.1','1','10','100'},'XTick',[0.0001 0.001 0.01 0.1 1 10 100])

%%


