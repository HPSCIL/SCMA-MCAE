function [ elmatrix ] = elcomputer( x,y )
%ELCOMPUTER Summary of this function goes here
%   Detailed explanation goes here
 Z = imsubtract( x,y );
%  elmatrix = sum(Z.*Z,2);
 elmatrix = sqrt(sum(Z.*Z,2));

 


 
 


