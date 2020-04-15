# simCLR-constractive-loss
constractive loss for simCLR
'''
out:[a1,a2,a3]
out_aug:[b1,b2,b3]

labels [batch_size,2*batch_size] batch_size=3 \
1 0 0 0 0 0 \
0 1 0 0 0 0 \
0 0 1 0 0 0 \
mask [batch_size,batch_size]
<br> 1 0 0 <br/>
<br> 0 1 0 <br/>
<br> 0 0 1 <br/>

logits_aa [batch_size,batch_size]
<br> a1*a1,a1*a2,a1*a3 <br/>
<br> a2*a1,a2*a2,a2*a3 <br/>
<br> a3*a1,a3*a2,a3*a3 <br/>

logits_bb [batch_size,batch_size]
<br> b1*b1,b1*b2,b1*b3 <br/>
<br> b2*b1,b2*b2,b2*b3 <br/>
<br> b3*b1,b3*b2,b3*b3 <br/>

logits_aa-INF*mask
<br> -INF,a1*a2,a1*a3 <br/>
<br> a2*a1,-INF,a2*a3 <br/>
<br> a3*a1,a3*a2,-INF <br/>

logits_bb-INF*mask [batch_size,batch_size] # delete same samples
<br> -INF,b1*b2,b1*b3 <br/>
<br> b2*b1,-INF,b2*b3 <br/>
<br> b3*b1,b3*b2,-INF <br/>

logits_ab [batch_size,batch_size]
<br> a1*b1,a1*b2,a1*b3 <br/>
<br> a2*b1,a2*b2,a2*b3 <br/>
<br> a3*b1,a3*b2,a3*b3 <br/>

logtis_ba [batch_size,batch_size]
<br> b1*a1,b1*a2,b1*a3 <br/>
<br> b2*a1,b2*a2,b2*a3 <br/>
<br> b3*a1,b3*a2,b3*a3 <br/>

concat[logits_ab,logits_aa]:
<br> a1*b1,a1*b2,a1*b3,-INF,a1*a2,a1*a3 <br/>
<br> a2*b1,a2*b2,a2*b3,a2*a1,-INF,a2*a3 <br/>
<br> a3*b1,a3*b2,a3*b3,a3*a1,a3*a2,-INF <br/>
<br> only a1*b1, a2*b2, a3*b3  is positives

concat [logits_ab,logits_bb]:
<br> b1*a1,b1*a2,b1*a3,-INF,b1*b2,b1*b3 <br/>
<br> b2*a1,b2*a2,b2*a3,b2*b1,-INF,b2*b3 <br/>
<br> b3*a1,b3*a2,b3*a3,b3*b1,b3*b2,-INF <br/>
only b1*a1, b2*a2, b3*a3  is positives, so calculate the softmax_cross_entropy with labels
labels [batch_size,2*batch_size] batch_size=3
<br> 1 0 0 0 0 0 <br/>
<br> 0 1 0 0 0 0 <br/>
<br> 0 0 1 0 0 0 <br/>
'''
