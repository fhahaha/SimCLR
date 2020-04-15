# simCLR-contrastive-loss
contrastive loss for simCLR
```
out:[a1,a2,a3] \
out_aug:[b1,b2,b3] \

labels [batch_size,2*batch_size] batch_size=3 \
1 0 0 0 0 0 \
0 1 0 0 0 0 \
0 0 1 0 0 0 \
mask [batch_size,batch_size]
1 0 0 \
0 1 0 \
0 0 1 \

logits_aa [batch_size,batch_size]
a1*a1,a1*a2,a1*a3 \
a2*a1,a2*a2,a2*a3 \ 
a3*a1,a3*a2,a3*a3 \ 

logits_bb [batch_size,batch_size]
b1*b1,b1*b2,b1*b3 \
b2*b1,b2*b2,b2*b3 \
b3*b1,b3*b2,b3*b3 \

logits_aa-INF*mask
-INF,a1*a2,a1*a3 \
a2*a1,-INF,a2*a3 \
a3*a1,a3*a2,-INF \

logits_bb-INF*mask [batch_size,batch_size] # delete same samples
-INF,b1*b2,b1*b3 \
b2*b1,-INF,b2*b3 \
b3*b1,b3*b2,-INF \

logits_ab [batch_size,batch_size]
a1*b1,a1*b2,a1*b3 \
a2*b1,a2*b2,a2*b3 \
a3*b1,a3*b2,a3*b3 \

logtis_ba [batch_size,batch_size]
b1*a1,b1*a2,b1*a3 \
b2*a1,b2*a2,b2*a3 \
b3*a1,b3*a2,b3*a3 \

concat[logits_ab,logits_aa]:
a1*b1,a1*b2,a1*b3,-INF,a1*a2,a1*a3 \
a2*b1,a2*b2,a2*b3,a2*a1,-INF,a2*a3 \
a3*b1,a3*b2,a3*b3,a3*a1,a3*a2,-INF \
only a1*b1, a2*b2, a3*b3  are positives

concat [logits_ab,logits_bb]:
b1*a1,b1*a2,b1*a3,-INF,b1*b2,b1*b3 \
b2*a1,b2*a2,b2*a3,b2*b1,-INF,b2*b3 \
b3*a1,b3*a2,b3*a3,b3*b1,b3*b2,-INF \
only b1*a1, b2*a2, b3*a3  are positives, so calculate the softmax_cross_entropy with labels
```
