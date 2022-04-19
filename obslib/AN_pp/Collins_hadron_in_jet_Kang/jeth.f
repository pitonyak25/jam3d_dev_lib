      program master
      implicit none
      real*8 roots,pT,yrap,z,AUT,cross
      real*8 XL,XU,ACC,crossC,crossH,sd,chi2a,za(6),jta(8),jt
      real*8 jtint,jtdiff,width,Mcs,pi,collinscross,hadroncross
      integer NDIM,NCALL,ITMX,NPRN,NDEV,ncmax
      integer IC,i,ncross,FINI,IINI,IINIP
      COMMON / FRAGINI / FINI
      COMMON / INTINI / IINI
      COMMON / INTINIP / IINIP
      common /experiment/ roots,pT,yrap,z,IC,ncross
c----for vegas---------
      COMMON/BVEG1/XL(10),XU(10),ACC,NDIM,NCALL,ITMX,NPRN,NDEV
      SAVE  /BVEG1/
c----------------------  
      data ncmax /120000/
c-----IC=1 pi+, -1 pi-, 0 pi0      
      data FINI/0/IINI/0/IINIP/0/pi/3.1415926535d0/ncross/4/
      data width/0.12d0/Mcs/0.28d0/
      data roots /5.0d2/IC/-1/    
      data za /0.10d0,0.20d0,0.30d0,0.40d0,0.5d0,0.60d0/
      data jta /0.1d0,0.2d0,0.3d0,0.4d0,0.5d0,0.7d0,1.0d0,2.d0/
      external collinscross,hadroncross

      NDIM=1
      XL(1)=0d0                 ! ymin
      XU(1)=1d0                 ! ymax

c-----for z-dependence
c      jtint = dsqrt(pi*dexp(1d0)/2d0)*(width**0.5d0)*Mcs
c     >     /(width+Mcs)**1.5d0
c      pT=12.9d0

c      do i=1,6
c         z=za(i)

c         ncall=100         
c 45      call vegas(hadroncross,crossH,sd,chi2a)
c         if(abs(sd/crossH).gt.0.01d0.and.ncall.le.ncmax) then
c            ncall=2*ncall
c            goto 45
c         endif
c         print *,"ncall: ",ncall

c         ncall=100
c 44      call vegas(collinscross,crossC,sd,chi2a)
c         if(abs(sd/crossC).gt.0.01d0.and.ncall.le.ncmax) then
c            ncall=2*ncall
c            goto 44
c         endif
c         print *,"ncall: ",ncall
         
c         AUT=crossC/crossH*jtint
c         write( 6,110) z,crossC,crossH,AUT
c         write(66,110) z,AUT
c      enddo
      
c-----for jt-dependence
      pT=31.0d0
      z=0.37d0
      do i=1,8
         jt=jta(i)
         jtdiff = dsqrt(2d0*dexp(1d0)/Mcs)*jt*dexp(-jt**2d0/Mcs)

         ncall=100         
 45      call vegas(hadroncross,crossH,sd,chi2a)
         if(abs(sd/crossH).gt.0.01d0.and.ncall.le.ncmax) then
            ncall=2*ncall
            goto 45
         endif
         print *,"ncall: ",ncall

         ncall=100
 44      call vegas(collinscross,crossC,sd,chi2a)
         if(abs(sd/crossC).gt.0.01d0.and.ncall.le.ncmax) then
            ncall=2*ncall
            goto 44
         endif
         print *,"ncall: ",ncall
         
         AUT=crossC/crossH*jtdiff
         write( 6,110) jt,crossC,crossH,AUT
         write(66,110) jt,AUT
      enddo

 110  format(4(1pE12.4))
      
      return
      end

c----------------------------------------------------------------
c     hadron and jet cross section
c----------------------------------------------------------------
      function hadroncross(var)
      implicit none
      real*8 hadroncross,var(1),roots,pT,yrap,z,cross
      integer IC,ncross
      common /experiment/ roots,pT,yrap,z,IC,ncross
      yrap=var(1)

      call hadron(IC,roots,pT,yrap,z,ncross,cross)
      hadroncross = cross
      return
      end

c-------
      function collinscross(var)
      implicit none
      real*8 collinscross,var(1),roots,pT,yrap,z,cross
      integer IC,ncross
      common /experiment/ roots,pT,yrap,z,IC,ncross
      yrap=var(1)

      call collinshadron(IC,roots,pT,yrap,z,ncross,cross)
      collinscross = cross

      return
      end

c----------------------------------------------------------------
c     Collins cross section for hadron
c----------------------------------------------------------------

      subroutine collinshadron(IC,rts,pT,yrap,z,ncross,cross)
      implicit none
      real*8 rts1,y1,pT1,z1,jT1,cross,sigmacollins,scale,scale1
      real*8 rts,pT,yrap,z,jT,DINTEG,UU,TT,rts2,xbmin
      integer ncross,IC1,IC
      common /integral/ rts1,pT1,y1,z1,scale1,IC1
      external sigmacollins
      IC1=IC
      rts1=rts
      pT1=pT
      y1=yrap
      z1=z
      scale1=pT

      TT=-rts*pT*dexp(-yrap)
      UU=-rts*pT*dexp(yrap)
      rts2=rts**2
      xbmin=-TT/(rts2+UU)
      cross = DINTEG(sigmacollins,xbmin,1d0,ncross)      

      return
      end

c-----
      function sigmacollins(xb)
      implicit none
      real*8 sigmacollins,sig,sh,uh,th,rts1,pT1,y1,rts2
      real*8 xa,xb,z1,TT,UU,alphas,scale1
      real*8 hU1,hD1,hUB1,hDB1,hS1,hSB1
      real*8 U2,D2,UB2,DB2,S2,SB2,GL2
      real*8 Hu,Hub,Hd,Hdb,Hs,Hsb
      integer IC1
      common /integral/ rts1,pT1,y1,z1,scale1,IC1
      common / hx1/ hU1,hD1,hUB1,hDB1,hS1,hSB1
      common / px2 / U2, D2, UB2, DB2, S2, SB2, GL2
      common / Hz / Hu,Hub,Hd,Hdb,Hs,Hsb

      rts2=rts1**2
      TT=-rts1*pT1*dexp(-y1)
      UU=-rts1*pT1*dexp(y1)
      xa=-xb*UU/(xb*rts2+TT)
      
      sh=xa*xb*rts2
      uh=xb*UU
      th=xa*TT

      call TRANS(xa,scale1,hU1,hD1,hUB1,hDB1,hS1,hSB1)
      call PDF(xb,scale1,U2,D2,UB2,DB2,S2,SB2,GL2)  
      call CollFF(IC1,z1,scale1,Hu,Hub,Hd,Hdb,Hs,Hsb) 
      call COLLSIG(sig,sh,th,uh)
         
      sigmacollins=2d0*pT1/(xa*xb*(xb*rts2+TT))*sig
     >     *(alphas(scale1)**2d0)/rts2
      
      return
      end


c----------------------------------------------------------------
c     Differential cross section for hadron
c----------------------------------------------------------------

      subroutine hadron(IC,rts,pT,yrap,z,ncross,cross)
      implicit none
      real*8 rts1,y1,pT1,z1,cross,sigmahadron,scale,scale1
      real*8 rts,pT,yrap,z,DINTEG,UU,TT,rts2,xbmin
      integer ncross,IC,IC1
      common /integral/ rts1,pT1,y1,z1,scale1,IC1
      external sigmahadron
      IC1=IC
      rts1=rts
      pT1=pT
      y1=yrap
      z1=z
      scale1=pT

      TT=-rts*pT*dexp(-yrap)
      UU=-rts*pT*dexp(yrap)
      rts2=rts**2
      xbmin=-TT/(rts2+UU)
      cross = DINTEG(sigmahadron,xbmin,1d0,ncross)      

      return
      end

c-----
      function sigmahadron(xb)
      implicit none
      real*8 sigmahadron,sig,sh,uh,th,rts1,pT1,y1,rts2
      real*8 xa,xb,z1,TT,UU,alphas,scale1
      real*8 U1,D1,UB1,DB1,S1,SB1,GL1
      real*8 U2,D2,UB2,DB2,S2,SB2,GL2
      real*8 fu,fub,fd,fdb,fs,fsb,fg
      integer IC1
      common /integral/ rts1,pT1,y1,z1,scale1,IC1
      common / px1 / U1,D1,UB1,DB1,S1,SB1,GL1
      common / px2 / U2,D2,UB2,DB2,S2,SB2,GL2
      common / fz / fu,fub,fd,fdb,fs,fsb,fg

      rts2=rts1**2
      TT=-rts1*pT1*dexp(-y1)
      UU=-rts1*pT1*dexp(y1)
      xa=-xb*UU/(xb*rts2+TT)
      
      sh=xa*xb*rts2
      uh=xb*UU
      th=xa*TT

      call PDF(xa,scale1,U1,D1,UB1,DB1,S1,SB1,GL1)
      call PDF(xb,scale1,U2,D2,UB2,DB2,S2,SB2,GL2)  
      call FF(IC1,z1,scale1,fu,fub,fd,fdb,fs,fsb,fg) 
      call UUSIG(sig,sh,th,uh)
         
      sigmahadron=2d0*pT1/(xa*xb*(xb*rts2+TT))*sig
     >     *(alphas(scale1)**2d0)/rts2
      
      return
      end

c-------------------------------------------------------------------
c     unpolarized partonic cross section
c-------------------------------------------------------------------

      subroutine UUSIG(sig,s,t,u)
      implicit none
      real*8 U1,D1,UB1,DB1,S1,SB1,GL1
      real*8 U2,D2,UB2,DB2,S2,SB2,GL2
      real*8 sig,s,t,u,sig1,sig2,sig3,sig4
      real*8 WQ1,WQ2,WQ3,WQ4,WQ5,WQ6,WQ7,WQ8
      real*8 WT1,WT2,WT3,WT4,WT5,WT6,WT7,WT8
      real*8 fu,fub,fd,fdb,fs,fsb,fg
      common / px1 / U1,D1,UB1,DB1,S1,SB1,GL1
      common / px2 / U2,D2,UB2,DB2,S2,SB2,GL2
      common / fz / fu,fub,fd,fdb,fs,fsb,fg

      call Upp(s,t,u,WQ1,WQ2,WQ3,WQ4,WQ5,WQ6,WQ7,WQ8)
      call Upp(s,u,t,WT1,WT2,WT3,WT4,WT5,WT6,WT7,WT8)

      SIG1=WQ2*(U1*U2*fu+UB1*UB2*fub+D1*D2*fd+DB1*DB2*fdb
     >         +S1*S2*fs+SB1*SB2*fsb)
     >    +WQ3*(U1*UB2*(fd+fs)+D1*DB2*(fu+fs)+S1*SB2*(fu+fd)
     >         +UB1*U2*(fdb+fsb)+DB1*D2*(fub+fsb)
     >         +SB1*S2*(fub+fdb))
     >    +WT3*(UB1*U2*(fd+fs)+DB1*D2*(fu+fs)+SB1*S2*(fu+fd)
     >         +U1*UB2*(fdb+fsb)+D1*DB2*(fub+fsb)
     >         +S1*SB2*(fub+fdb))
      SIG2=WQ4*(U1*UB2*fu+D1*DB2*fd+S1*SB2*fs
     >         +UB1*U2*fub+DB1*D2*fdb+SB1*S2*fsb)
     >    +WT4*(UB1*U2*fu+DB1*D2*fd+SB1*S2*fs
     >         +U1*UB2*fub+D1*DB2*fdb+S1*SB2*fsb)
     >    +WQ1*(U1*(D2+DB2+S2+SB2)*fu
     >         +D1*(U2+UB2+S2+SB2)*fd
     >         +UB1*(D2+DB2+S2+SB2)*fub
     >         +DB1*(U2+UB2+S2+SB2)*fdb
     >         +S1*(U2+UB2+D2+DB2)*fs
     >         +SB1*(U2+UB2+D2+DB2)*fsb)
      SIG3=WT1*((D1+DB1+S1+SB1)*U2*fu
     >         +(U1+UB1+S1+SB1)*D2*fd
     >         +(D1+DB1+S1+SB1)*UB2*fub
     >         +(U1+UB1+S1+SB1)*DB2*fdb
     >         +(U1+UB1+D1+DB1)*S2*fs
     >         +(U1+UB1+D1+DB1)*SB2*fsb)
     >    +WQ5*(U1*UB2+D1*DB2+S1*SB2)*fg
     >    +WT5*(UB1*U2+DB1*D2+SB1*S2)*fg
      SIG4=WQ6*GL1*GL2*(fu+fd+fs)
     >    +WT6*GL1*GL2*(fub+fdb+fsb)
     >    +WQ7*(GL2*(U1*fu+D1*fd+S1*fs+UB1*fub+DB1*fdb+SB1*fsb)
     >         +GL1*fg*(U2+UB2+D2+DB2+S2+SB2))
     >    +WT7*(GL1*(U2*fu+D2*fd+S2*fs+UB2*fub+DB2*fdb+SB2*fsb)
     >         +GL2*fg*(U1+UB1+D1+DB1+S1+SB1))
     >    +WQ8*GL1*GL2*fg
      SIG=SIG1+SIG2+SIG3+SIG4

      return
      end
     
c-------------------------------------------------------------------

      subroutine Upp(s,t,u,WQ1,WQ2,WQ3,WQ4,WQ5,WQ6,WQ7,WQ8)
      implicit none
      real*8 s,t,u,WQ1,WQ2,WQ3,WQ4,WQ5,WQ6,WQ7,WQ8,Nc
      
      Nc=3d0

c     qq' ->qq'
      WQ1=(Nc**2-1d0)/(2d0*Nc**2)*(s*s+u*u)/(t*t)
c     qq  ->qq
      WQ2=(Nc**2-1d0)/(2d0*Nc**2)*( (s*s+u*u)/(t*t)+(s*s+t*t)/(u*u) )
     >     -(Nc**2-1d0)/(Nc**3)*(s*s)/(t*u)
c     qqb ->q'qb'
      WQ3=(Nc**2-1d0)/(2d0*Nc**2)*(t*t+u*u)/(s*s)
c     qqb ->qqb
      WQ4=(Nc**2-1d0)/(2d0*Nc**2)*( (s*s+u*u)/(t*t)+(t*t+u*u)/(s*s) )
     >     -(Nc**2-1d0)/(Nc**3)*(u*u)/(s*t)
c     qqb ->gg
      WQ5=(Nc**2-1d0)**2/(2d0*Nc**3)*(u/t+t/u)
     >     -(Nc**2-1d0)/Nc*(t*t+u*u)/(s*s)
c     gg  ->qqb
      WQ6=1d0/(2d0*Nc)*(t/u+u/t)-Nc/(Nc**2-1d0)*(t*t+u*u)/(s*s)
c     qg  ->qg
      WQ7=(Nc**2-1d0)/(2d0*Nc**2)*(-s/u-u/s)+(s*s+u*u)/(t*t)
c     gg  ->gg
      WQ8=4d0*Nc**2/(Nc**2-1d0)*(3d0-t*u/(s*s)-s*u/(t*t)-s*t/(u*u))


      return
      end

c-------------------------------------------------------------------
c     Collins partonic cross section
c-------------------------------------------------------------------
      subroutine COLLSIG(sig,s,t,u)
      implicit none
      real*8 sig,s,t,u,SIG1,SIG2,SIG3,SIG4
      real*8 hU1,hD1,hUB1,hDB1,hS1,hSB1
      real*8 U2,D2,UB2,DB2,S2,SB2,GL2
      real*8 Hu,Hub,Hd,Hdb,Hs,Hsb
      real*8 WQ1,WQ2,WQ4,WT4,WQ7
      common / hx1/ hU1,hD1,hUB1,hDB1,hS1,hSB1
      common / px2 / U2, D2, UB2, DB2, S2, SB2, GL2
      common / Hz / Hu,Hub,Hd,Hdb,Hs,Hsb

      call Collpp(s,t,u,WQ1,WQ2,WQ4,WT4,WQ7)

      SIG1=WQ2*(hU1*U2*Hu+hUB1*UB2*Hub+hD1*D2*Hd+hDB1*DB2*Hdb
     >         +hS1*S2*Hs+hSB1*SB2*Hsb)
      SIG2=WQ1*(hU1*(D2+DB2+S2+SB2)*Hu
     >         +hD1*(U2+UB2+S2+SB2)*Hd
     >         +hUB1*(D2+DB2+S2+SB2)*Hub
     >         +hDB1*(U2+UB2+S2+SB2)*Hdb
     >         +hS1*(U2+UB2+D2+DB2)*Hs
     >         +hSB1*(U2+UB2+D2+DB2)*Hsb)
      SIG3=WQ4*(hU1*UB2*Hu+hD1*DB2*Hd+hUB1*U2*Hub
     >         +hDB1*D2*Hdb+hS1*SB2*Hs+hSB1*S2*Hsb)
     >    +WT4*(hUB1*U2*Hu+hDB1*D2*Hd+hU1*UB2*Hub
     >         +hD1*DB2*Hdb+hSB1*S2*Hs+hS1*SB2*Hsb)
      SIG4=WQ7*GL2*(hU1*Hu+hD1*Hd+hUB1*Hub+hDB1*Hdb
     >         +hS1*Hs+hSB1*Hsb)

      SIG=SIG1+SIG2+SIG3+SIG4
      RETURN
      END

c-------------------------------------------------------------------    
 
      subroutine Collpp(s,t,u,WQ1,WQ2,WQ4,WT4,WQ7)
      implicit none
      real*8 s,t,u,WQ1,WQ2,WQ4,WT4,WQ7,Nc

      Nc=3d0

c     q+q'->q+q'
      WQ1=(Nc**2-1d0)/(Nc**2)*(s*u)/(-t*t)
c     q+q->q+q
      WQ2=(Nc**2-1d0)/(Nc**2)*( (s*u)/(-t*t)-1d0/Nc*s/(-t))
c     q+qb->q+qb, qb+q->qb+q
      WQ4=(Nc**2-1d0)/(Nc**2)*( (s*u)/(-t*t)+1d0/Nc*u/t)
c     qb+q->q+qb, q+qb->qb+q
      WT4 = -(Nc**2-1d0)/(Nc**3)
c     q+g->q+g
      WQ7=(Nc**2-1d0)/(Nc**2)+2d0*(s*u)/(-t*t)

c-----all other channels are equal to zero
c     q'q->qq', qqb->q'qb', qqb->gg, gg->qqb, gg->gg

      return
      end


c----------------------------------------------------------------
c     GRV98 LO PDFs
c----------------------------------------------------------------

      subroutine PDF(x,Q,U,D,UB,DB,SS,SB,GL)
      implicit none
      real*8 x,Q,Q2,U,D,UB,DB,SS,SB,GL
      real*8 UV,DV,US,DS,ST,GU
      integer ISET
      data ISET /1/ ! LO

      Q2 = Q*Q
      call GRV98PA (ISET, X, Q2, UV, DV, US, DS, ST, GU)

      U = (UV+US)/X
      D = (DV+DS)/X
      UB= US/X
      DB= DS/X
      SS= ST/X
      SB= SS
      GL= GU/X
     
      return
      end

C-------------------------------------------------------------------
C     transversity distribution: from 1510.05389
C     with GRSV2000 helicity and GRV98 PDFs (both LO)
C-------------------------------------------------------------------
      SUBROUTINE TRANS(X,Q,U,D,UB,DB,SS,SB)
      implicit none
      real*8 X,Q,U,D,UB,DB,SS,SB
      real*8 Q2,hU,hD,hUB,hDB,hST
      real*8 Nut,Ndt,al,be,tem
      data Nut,Ndt,al,be /0.61d0,-1d0,0.70d0,1.80d0/
      integer ISET
      data ISET /2/ ! LO

      Q2 = Q*Q

      call PARPOLT(ISET, X, Q2, hU, hD, hUB, hDB, hST)
      tem=(x**al)*(1d0-x)**be*(al+be)**(al+be)/(al**al)/(be**be)
      U = Nut*tem*hU/x
      D = Ndt*tem*hD/x
      UB = 0d0
      DB = 0d0
      SS = 0d0
      SB = 0d0
      
      RETURN
      END


c-------------------------------------------------------------------
c     Collins fragmentation function: the collinear part
c-------------------------------------------------------------------
      subroutine CollFF(IC,z,Q,DPU,DPUB,DPD,DPDB,DPS,DPSB)
      implicit none
      integer IC,ipion
      real*8 z,Q,DPU,DPUB,DPD,DPDB,DPS,DPSB
      real*8 fu,fub,fd,fdb,fs,fsb,fg
      real*8 FAV,UNF,tem,Nfav,Nunf,gam,del
      data Nfav,Nunf,gam,del /0.90d0,-0.37d0,2.02d0,0.00d0/
      data ipion /1/ ! use pi+ like Feng and Werner

      call FF(ipion,z,Q,fu,fub,fd,fdb,fs,fsb,fg) 
      tem=z**gam*(1d0-z)**del*(gam+del)**(gam+del)/gam**gam/del**del
      FAV=Nfav*tem*fu
      UNF=Nunf*fd

      if(IC.eq.1) then
         DPU = FAV
         DPUB = UNF
         DPD = UNF
         DPDB = FAV
      elseif(IC.eq.-1) then
         DPU = UNF
         DPUB = FAV
         DPD = FAV
         DPDB = UNF
      elseif(IC.eq.0) then
         DPU = (FAV + UNF)/2d0
         DPUB = (FAV + UNF)/2d0
         DPD = (FAV + UNF)/2d0
         DPDB = (FAV + UNF)/2d0
      else
         print *,'the hadron is not available,icharge:',IC
      endif

      DPS = 0D0
      DPSB = 0D0

      return
      end


c----------------------------------------------------------------
c     DSS FFs
c----------------------------------------------------------------

      subroutine FF(IC,z,Q,fu,fub,fd,fdb,fs,fsb,fg)
      implicit none
      real*8 z,Q,Q2,fu,fub,fd,fdb,fs,fsb,fg
      real*8 U,UB,D,DB,S,SB,C,B,GL
      integer IS,IL,IC,IO
      data IS,IL,IO /0,0,1/

      Q2=Q*Q
      call fDSSH(IS,IL,IC,IO,z,Q2,U,UB,D,DB,S,SB,C,B,GL)
      fu=U/z
      fub=UB/z
      fd=D/z
      fdb=DB/z
      fs=S/z
      fsb=SB/z
      fg=GL/z

      return
      end

c--------------------------------------------------------------
c     alphas(q) -- strong coupling constant
c--------------------------------------------------------------      
      function alphas(q)
      implicit none
      real*8 q,q2,lambda,lambda2,mb,alphas,b0,b1,pi,tt
      integer nf
      pi=4d0*datan(1d0)
      mb=4.5d0
      if (q.le.mb) then
         nf=4
         lambda=0.326d0
      else
         nf=5
         lambda=0.226d0
      endif
      b0=11d0-2d0/3d0*nf
      b1=51d0-19d0/3d0*nf
      q2=q*q
      lambda2=lambda*lambda
      tt=dlog(q2/lambda2)
      alphas=4d0*pi/(b0*tt)*(1d0-2d0*b1/(b0*b0)*log(tt)/tt)
      return
      end

c-------------------------------------------------------------------
c     \int_yi^yf dy func(y)
c     n: # of interval, we split the integration region
c-------------------------------------------------------------------

      function DINTEG(func,yi,yf,n)
      implicit none
      real*8 DINTEG,func,yi,yf,yn,value
      real*8 y1,y2,val,ym,yr,dy,y(8),w(8)
      integer n,i,j
      external func
      data y/0.0950125098376374d0, 0.2816035507792589d0,
     &     0.4580167776572274d0, 0.6178762444026438d0,
     &     0.7554044083550031d0, 0.865631202387832d0,
     &     0.944575023073233d0, 0.98940093499165d0/

      data w/0.1894506104550685d0, 0.1826034150449236d0,
     &     0.1691565193950025d0,  0.1495959888165767d0,
     &     0.1246289712555339d0, 0.0951585116824928d0,
     &     0.06225352393864789d0, 0.0271524594117541d0/

      yn=(yf-yi)/n
      value=0d0
      y2=yi
      Do 200 i=1,n
         y1=y2
         y2=y1+yn
         
         ym=0.5d0*(y2+y1)
         yr=0.5d0*(y2-y1)
         val=0d0
         Do 300 j=1,8
            dy=yr*y(j)
            val=val+w(j)*(func(ym+dy)
     >                   +func(ym-dy))
 300     continue
         value=value+yr*val
 200  continue
      DINTEG=value
      return
      end
