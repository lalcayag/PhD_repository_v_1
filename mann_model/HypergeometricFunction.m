function F = HypergeometricFunction(a,b,c,z,method,accuracy)

if ~exist('method','var')
    method = 1;
end

if method == 1
    
    if ~exist('accuracy','var')
        npoints = 100;
    else
        npoints = accuracy;
    end
    
    n = 0:1:npoints;
    nfact = factorial(n);
    Ga = gamma(a);
    Gb = gamma(b);
    Gc = gamma(c);
    Gcab = gamma(c - a - b);
    Gca = gamma(c - a);
    Gcb = gamma(c - b);    
    Gan = gamma(a + n);
    Gbn = gamma(b + n);
    Gcn = gamma(c + n);
    ai = a;
    bi = c - b;
    ci = c;
    Gai = gamma(ai);
    Gbi = gamma(bi);
    Gci = gamma(ci);
    Gain = gamma(ai + n);
    Gbin = gamma(bi + n);
    Gcin = gamma(ci + n);
         
    
    zsize = size(z);    
    F = zeros(zsize(1),zsize(2));
    
    for i = 1:zsize(1)
        for j = 1:zsize(2)
            
            if (z(i,j)) == 1
                F(i,j) = Gc.*Gcab./(Gca.*Gcb);
            elseif (z(i,j) >= -0.5 && z(i,j) < 1) || (z(i,j) > 1 && z(i,j) < 2)
                zi = z(i,j);
                zn = zi.^n;

                Gprod = (Gan.*Gbn.*zn)./(Gcn.*nfact);
                Gprod = Gprod(~isnan(Gprod));
                F(i,j) = (Gc/(Ga*Gb))*sum(Gprod);
            else
                zi = z(i,j)/(z(i,j) - 1);
                q0 = (1 - z(i,j))^(-a);
                zn = zi.^n;

                Gprod = (Gain.*Gbin.*zn)./(Gcin.*nfact);
                Gprod = Gprod(~isnan(Gprod));
                F(i,j) = q0*(Gci/(Gai*Gbi))*sum(Gprod);            

            end              
%             if abs(z(i,j)) == 1
%                 Gc = gamma(c);
%                 Gcab = gamma(c - a - b);
%                 Gca = gamma(c - a);
%                 Gcb = gamma(c - b);
%                 F(i,j) = Gc.*Gcab./(Gca.*Gcb);
%             elseif (z(i,j) >= -0.5 && z(i,j) < 1) || (z(i,j) > 1 && z(i,j) < 2)
%                 zi = z(i,j);
%                 n = 0:1:npoints;
%                 Gan = gamma(a + n);
%                 Gbn = gamma(b + n);
%                 Gcn = gamma(c + n);
%                 zn = zi.^n;
%                 nfact = factorial(n);
% 
%                 Ga = gamma(a);
%                 Gb = gamma(b);
%                 Gc = gamma(c);
%                 Gprod = (Gan.*Gbn.*zn)./(Gcn.*nfact);
%                 Gprod = Gprod(~isnan(Gprod));
%                 F(i,j) = (Gc/(Ga*Gb))*sum(Gprod);
%             else
%                 ai = a;
%                 bi = c - b;
%                 ci = c;
%                 zi = z(i,j)/(z(i,j) - 1);
%                 q0 = (1 - z(i,j))^(-a);
% 
%                 n = 0:1:npoints;
%                 Gan = gamma(ai + n);
%                 Gbn = gamma(bi + n);
%                 Gcn = gamma(ci + n);
%                 zn = zi.^n;
%                 nfact = factorial(n);
% 
%                 Ga = gamma(ai);
%                 Gb = gamma(bi);
%                 Gc = gamma(ci);
%                 Gprod = (Gan.*Gbn.*zn)./(Gcn.*nfact);
%                 Gprod = Gprod(~isnan(Gprod));
%                 F(i,j) = q0*(Gc/(Ga*Gb))*sum(Gprod);            
% 
%             end
        end
    end
    
elseif method == 2
    
    if ~exist('accuracy','var')
        npoints = 100;
    else
        npoints = accuracy;
    end
    
    F = zeros(1,length(z));
    for i = 1:length(z)
        zi = z(i);
        n = 0:1:npoints;
        Gan = gamma(a + n);
        Gbn = gamma(b + n);
        Gcn = gamma(c + n);
        zn = zi.^n;
        nfact = factorial(n);

        Ga = gamma(a);
        Gb = gamma(b);
        Gc = gamma(c);
        Gprod = (Gan.*Gbn.*zn)./(Gcn.*nfact);
        Gprod = Gprod(~isnan(Gprod));
        F(i) = (Gc/(Ga*Gb))*sum(Gprod);
    end
    
elseif method == 3
    integralfunc = @(t) ( t.^(b-1) ) .* ( (1 - t).^(c - b - 1) ) .* ( (1 - t.*z).^(-a) );
    integralvalue = integral(integralfunc,0,1,'ArrayValued',true,'AbsTol',1e-12,'RelTol',1e-12);
%     integralvalue = zeros(1,length(z));
%     for i = 1:length(z)
%         integralfunci = @(t) ( t.^(b-1) ) .* ( (1 - t).^(c - b - 1) ) .* ( (1 - t.*z(i)).^(-a) );
%         integralvalue(i) = integral(integralfunci,0,1,'AbsTol',1e-2,'RelTol',1e-10);
%     end

    F = (gamma(c)./gamma(b))./gamma(c-b).*integralvalue;
    
elseif method == 4
    

    if ~exist('accuracy','var')
        npoints = 100;
    else
        npoints = accuracy;
    end
    
    F = zeros(1,length(z));
    
    for i = 1:length(z)
        if abs(z(i)) == 1
            Gc = gamma(c);
            Gcab = gamma(c - a - b);
            Gca = gamma(c - a);
            Gcb = gamma(c - b);
            F(i) = Gc.*Gcab./(Gca.*Gcb);
        else
            zi = z(i);
            n = 0:1:npoints;
            qa = [1, a + n(2:end) - 1];
            qb = [1, b + n(2:end) - 1];
            qc = [1, c + n(2:end) - 1];
            
            qsuma = cumsum(log(qa));
            qsumb = cumsum(log(qb));
            qsumc = cumsum(log(qc));           
            neven = 2*floor(n/2);
            oddindex = rem(n,2);
            znvect = (neven/2).*log(zi^2);
            nvect = [0 cumsum(log(n(2:end)))];        
            allsum = qsuma + qsumb - qsumc + znvect - nvect;
            zminusvect = zi.*oddindex;
            expsum = exp(allsum).*zminusvect;
            F(i) = sum(expsum);

        end
    end        
            
end