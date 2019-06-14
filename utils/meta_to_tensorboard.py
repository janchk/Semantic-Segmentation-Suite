import tensorflow as tf

g = tf.Graph()

with g.as_default() as g:
    tf.train.import_meta_graph('')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='logdir/new_unet', graph=g)
