import tensorflow as tf

FILE = 'data/testbisenet4.pb'

graph = tf.Graph()
with graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FILE, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session()
    op = sess.graph.get_operations()

    for m in op:
        print(m.values())