import numpy as np

def contrastive_loss(out,out_aug,batch_size=128,hidden_norm=False,temperature=1.0):

    if hidden_norm:
        out=tf.nn.l2_normalize(out,-1)
        out_aug=tf.nn.l2_normalize(out_aug,-1)
    inf_value = np.inf
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2) #[batch_size,2*batch_size]
    masks = tf.one_hot(tf.range(batch_size), batch_size) #[batch_size,batch_size]
    logits_aa = tf.matmul(out, out, transpose_b=True) / temperature #[batch_size,batch_size]
    logits_bb = tf.matmul(out_aug, out_aug, transpose_b=True) / temperature #[batch_size,batch_size]
    logits_aa = logits_aa - masks * inf_value # remove the same samples in out
    logits_bb = logits_bb - masks * inf_value # remove the same samples in out_aug
    logits_ab = tf.matmul(out, out_aug, transpose_b=True) / temperature
    logits_ba = tf.matmul(out_aug, out, transpose_b=True) / temperature
    loss_a = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss=loss_a+loss_b
    return loss,logits_ab
