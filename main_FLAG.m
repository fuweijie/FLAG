% Code of Fater Learning on Anchor Graph
% data \in N x D, label \in N x 1, m is # of anchors.

% Graph construction On medium size datasets, where Z1 is the inter-layer adjacency matrix.
[~, anchor] = litekmeans(data, m, 'MaxIter', 3); 
[Z1, ~] = AnchorGraph(data', anchor', 3, 0, 10);
% For ANNS-based graph construction on large-size datasets, please download the source code
% at FLANN 'http://www.cs.ubc.ca/research/flann/' and compile the it based on your own PC.

% Main function
% e.g., sparse parameter = 4. lambda=1e-2, mu=1e-6, d=20.
d=20;lambda=1e-2;mu=1e-6;

W=Z1'*Z1;
temp=W-diag(diag(W));
val = zeros(m,4);
pos = val;
for i = 1:4
   [val(:,i),pos(:,i)] = max(temp,[],2);
   tep = (pos(:,i)-1)*m+[1:m]';
   temp(tep) = 0;
end
W=sparse([1:m,1:m,1:m,1:m]',[pos(:);],[val(:);]);
if min(size(W))<m
    W(m,m)=0;
end
W=max(W,W');
sumW = sum(W') ;
D = sparse(1:m,1:m,sumW.^(-1/2));
E = D*W*D;E=max(E,E');
L=speye(size(E))-E;
[U,~]=eigs(L, d, 'sm');
rL=U'*diag(sum(W))*U-U'*W*U;

[err] = AnchorGraphReg(Z1, U, rL, label', label_index, lambda, mu);
