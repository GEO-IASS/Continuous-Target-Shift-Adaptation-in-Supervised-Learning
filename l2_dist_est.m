function [ l2_dist ] = l2_dist_est( x,U,V,ucap,delta,lamda,n,n1,b )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 H=V'*inv(U+delta*eye(n+n1))*V+lamda*eye(b);
      f=-ucap'*inv(U+delta*eye(n+n1))*V;
      
      c=0.5*ucap'*inv(U+delta*eye(n+n1))*ucap;
      
      l2_dist=0.5*x'*H*x+f*x+c;

end

