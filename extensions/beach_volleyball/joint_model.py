"""C74 input/MLP with full-table latent gradients, independent A/B networks."""
import base_c74 as b
tf=b.tf;P=b.P
class JointFinder(b.Finder):
 def __init__(self,m,seed):
  super().__init__(m,'direct',seed)
  self.table=tf.Variable(self.u0,trainable=True,name='joint_latent_table');self.lo=b.adam([self.table]);self.ck.table=self.table;self.ck.joint_optimizer=self.lo
  self.visits=[self.joint_visit(side) for side in range(2)]
 def joint_visit(self,side):
  @tf.function(reduce_retracing=True)
  def visit(a,bb,c,y,player,keys):
   n=tf.shape(a)[0];tr=tf.TensorArray(tf.float32,size=tf.shape(keys)[0])
   for j in tf.range(tf.shape(keys)[0]):
    sl=slice(j*b.B,tf.minimum((j+1)*b.B,n))
    with tf.GradientTape() as tape:
     pred=self.nets[side](self.features(a[sl],bb[sl],c[sl],side,key=keys[j],training=True),training=True);data=tf.reduce_mean((pred-y[sl])**2);reg=b.penalty(self.nets[side]);loss=data+P['l2']*reg
    vs=[self.table]+self.nets[side].trainable_variables;gs=tape.gradient(loss,vs);tf.debugging.assert_all_finite(loss,'joint loss')
    self.lo.apply_gradients([(gs[0],self.table)]);self.opts[side].apply_gradients(zip(gs[1:],vs[1:]));tr=tr.write(j,tf.stack([loss,data,reg]))
   return tr.stack()
  return visit
