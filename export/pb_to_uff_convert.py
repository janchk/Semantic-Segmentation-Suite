# Import TensorFlow and TensorRT
import tensorflow as tf
import uff

"""
Optimizing frozen tensorflow graph to perform inference
on tensorRT engine. Note that uff models probably wont't be supported by
Nvidia any time longer.
"""

# Inference with TF-TRT frozen graph workflow:


def optimize_pb_graph(graph_def, output_nodes, output_name,  sess):
    """
    :param graph_def:
    :param output_nodes:
    :param output_name: name of output file with .uff extension
    :param sess:
    :return: written file
    """
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, output_nodes)
    tf.graph_util.remove_training_nodes(frozen_graph)
    name = output_name.split(".")[0]
    output_name = "%s.uff" %name
    uff.from_tensorflow(
        frozen_graph, output_nodes,
        output_filename=output_name,
        text=True,
        # list_nodes=True,
        # write_preprocessed=True,
    )


if __name__ == "__main__":
    input_filename = "frozen_model_softmax_output.pb"  # Specify proper path to .pb graph
    output_name = "exported/unet_trt_last.uff"  # Path to optimized uff model

    output_nodes = ["output_name"]
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # First deserialize your frozen graph:
            with tf.gfile.GFile(input_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            optimize_pb_graph(graph_def, output_nodes, output_name, sess)
