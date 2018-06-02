function G=discriminantFunction(X,m,S,P)
G=((-1/2)*((X-m))*((inv(S))*(X-m)'))-0.5*log(det(S))+ log(P);
end