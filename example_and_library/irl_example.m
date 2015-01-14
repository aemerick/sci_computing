function [v, alpha, beta, VH, DH, bc] = irl()

%  Testing implicitly restarted lanczos for the hydrogen atom

edet = 5;      %  Number of accurate eigenvalues to find

kmax = 10;           %  Number of eigenvalues/eigenvectors
pmax = 20;          %  Number of QR decompositions/shifts
kpmax = kmax + pmax;

if (edet > kmax )
    fprintf('edet = %d.  Should be <= kmax = %d\n', edet, kmax);
end

%  Desired accuracy for resolved eigenvalues

eps = 1e-8;

%  Number of implicit restarts to do

nrestart = 100;

%  Flag to turn on explicit orthgonalizations in Lanczos step

lanczos_ortho = 1;

%  Box size

Lx=40;
Ly=Lx;
Lz=Lx;
V=Lx*Ly*Lz;

%  Location of center of atom

rloc = [Lx/2-3.2, Ly/2-1.6, Lz/2+2.7];

%  lattice spacing in appropriate units

a = 0.5;

%  Nearest neighbors:  1=xp, 2=xm, 3=yp, 4=ym, 5=zp, 6=zm, 7=local 1/r term

nbr = zeros(V,7);

%  Boundary conditions, where eigenvectors are zero

bc = [];

%  Set bc and neighbors matrix

set_bc_nbr();

% Lanczos vectors, operated on by Av

v = zeros(V,kpmax+1);

%  Resolved eigenvalues and eigenvectors to save

eknown = 0;     %  Number of eigenvalues and eigenvectors found
evalue = zeros(1,kmax);
evector = zeros(V,kmax);

%  Lanczos parameters and matrix

alpha = zeros(1,kpmax);
beta = zeros(1,kpmax);
H = zeros(kpmax,kpmax);

%  Set random starting vector, enforce boundary conditions and normalize

v(:,1) = rand(V,1) - 0.5;
v(bc(:),1) = 0.0;
v(:,1) = v(:,1)/norm(v(:,1));

%  Initial Lanczos step

lanczos(1,kpmax)

%  Loop over the number of  restart steps

for nr=1:nrestart
    
    %  Find eigenvalues and eigenvectors of H

    for k=1:kpmax
        H(k,k) = alpha(k);
        if ( k > 1 )
            H(k-1,k) = beta(k-1);
            H(k,k-1) = beta(k-1);
        end
    end
    
    [VH, DH ] = eig(H);
  
    %  Check for convergence of eigenvalues
    %  If converged, save eigenvector and eigenvalue
    
    for k=1:kpmax
        tmp = abs(beta(kpmax) * VH(kpmax,k) / DH(k,k));
        
        if ( mod(nr,5) == 0 && k <= kmax)
            fprintf('Eigenvalue number %d,  value %e,  fractional accuracy %e\n',k,DH(k,k),tmp);
        end
        
        if ( tmp < eps && k > eknown && k <= kmax)
                        
            eknown = eknown + 1;
            evalue(1,eknown) = DH(k,k);
            for mm=1:V
                for kk=1:kpmax
                    evector(mm,eknown) = evector(mm,eknown) + v(mm,kk) * VH(kk,k);
                end
            end
            evector(:,eknown) = evector(:,eknown)/norm(evector(:,eknown));
        end
    end
    
    if ( eknown < edet )
        
        %  Do pmax shifts
        
        Q = diag(ones(1,kpmax));
        Hplus = H;
        
        for k=kmax+1:kpmax
            [Qt, R] = qr(Hplus - DH(k,k) * diag(ones(1,kpmax)));
         
            Q = Q * Qt;
            Hplus = Qt.' * Hplus * Qt;
        end
        
        %  Find v+, fpk
        
        vp = v(:,1:end-1) * Q;
        betahat = Hplus(kmax,kmax+1);
        betatilde = beta(kpmax) * Q(kpmax,kmax);
        fpk = betahat * vp(:,kmax+1) + betatilde * v(:,kpmax+1);
        betap = norm(fpk);
        fpk = fpk/betap;
        
        
        % Check orthgonality
        
        for k1=2:kpmax
            for k2=1:k1-1
                dp(k1,k2) = dot(vp(:,k1),vp(:,k2));
            end
            dp(k1,kpmax) = dot(vp(:,k1),fpk(:));
        end
        
        fprintf('Maxiumum dot product for v+ vectors is %e\n', max(max(abs(dp))))
        
        %  Now restart Lanczos.        
        
        fprintf('Implicit restart %d of Lanczos.  %d eigenvalues accurately determined\n', nr, eknown);
        eknown = 0.0;
        
        v(:,1:kmax)= vp(:,1:kmax);
        alpha(:) = 0.0;
        beta(:) = 0.0;
        
        for k=1:kmax
            alpha(k)=Hplus(k,k);
            if ( k > 1 )
                beta(k-1) = Hplus(k-1,k);
            end
        end
        
        lanczos(kmax,kpmax)       
        
    else
        break
    end
     
end

%  Check eigenvectors

for k=1:edet
    v(:,k) = evector(:,k);
    Av(k,k+1);
    tmp = dot(v(:,k),v(:,k+1));
    fprintf('Eigenvalue %d = %e, norm %e, vAv for eigenvector = %e\n', ...
        k,  evalue(k), norm(evector(:,k)), tmp);
    
    for kk=1:k-1
        dp(k,kk) = dot(evector(:,k),evector(:,kk));
    end

end
fprintf('Maxiumum dot product for eigenvectors with k ~= kk is %e\n', max(max(abs(dp))));

%  Function to do Lanczos iterations, from k1 t k2
%  beta_{i} v_{i+1} = Av_i - alpha_i v_i - beta_{i-1} v_{i-1}
%  Note the indexing on beta, which is set to agree with Sorensen, et. al.

    function lanczos(k1,k2)
        
        for i=k1:k2
            
            Av(i,i+1);
            alpha(i) = dot(v(:,i), v(:,i+1));
            if ( i == 1)
                v(:,i+1) = v(:,i+1) - alpha(i) * v(:,i);
            else
                v(:,i+1) = v(:,i+1) - alpha(i) * v(:,i) - beta(i-1) * v(:,i-1);
            end
            
            %  Explict orthogonalization
            
            if ( lanczos_ortho == 1)
                for ii=1:i
                    v(:,i+1) = v(:,i+1) - dot(v(:,ii),v(:,i+1)) * v(:,ii);
                end
            end
            
            beta(i) = norm(v(:,i+1));
            v(:,i+1) = v(:,i+1)/beta(i);
        end
 
    end

%  Function to apply matrix to vector

    function [] = Av(i,o)
        
        for n=1:V
            v(n,o) = (3-nbr(n,7))*v(n,i);
            for m=1:6
                v(n,o) = v(n,o) - 0.5 * v(nbr(n,m),i);
            end
            
        end
        
        v(bc(:),o) = 0.0;
        
    end

%  set_bc_nbr loads the neighbor array with the lexical index for the neighbors
%  of site n and determines whether site n is on the boundary or not.

    function [] = set_bc_nbr()
        
        for x=1:Lx
            for y=1:Ly
                for z=1:Lz
                    if (x == Lx); xp = 1; else xp = x+1; end
                    if (y == Ly); yp = 1; else yp = y+1; end
                    if (z == Lz); zp = 1; else zp = z+1; end
                    if (x == 1); xm = Lx; else xm = x-1; end
                    if (y == 1); ym = Ly; else ym = y-1; end
                    if (z == 1); zm = Lz; else zm = z-1; end
                    
                    n = find_n(x,y,z);
                    nbr(n,1) = find_n(xp,y,z);
                    nbr(n,2) = find_n(xm,y,z);
                    nbr(n,3) = find_n(x,yp,z);
                    nbr(n,4) = find_n(x,ym,z);
                    nbr(n,5) = find_n(x,y,zp);
                    nbr(n,6) = find_n(x,y,zm);
                    nbr(n,7) = a * ((x-rloc(1))^2 + (y-rloc(2))^2 + (z-rloc(3))^2)^(-0.5);
                    
                    if ((x == Lx) || (y == Ly)|| (z == Lz) ...
                            || (x == 1) || (y == 1)|| (z == 1))
                        
                        bc = [ bc ; n];
                        
                    end
                end
            end
        end
    max(nbr(:,7))
    min(nbr(:,7))
    
        
    end

%  find_n returns the lexical order index n for a point (x,y,z)

    function [n] = find_n(x,y,z)
        
        n = x + Lx*(y-1) + Lx * Ly * (z-1);
        
    end
    
end
        
