import os, argparse
import tensorflow as tf
from export.pb_to_uff_convert import optimize_pb_graph

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_node_names):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    # name of checkpoint
    ckpt_name = input_checkpoint.split("/")[-1].split("/")[0]
    output_graph = absolute_model_dir + "/%s_frozen.pb" % ckpt_name

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the useful nodes
        )
        for node in output_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def, ckpt_name,  sess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="Path to checkpoint model directory",
                        help="Model folder to export")
    parser.add_argument("--export_uff", type=bool, default=False,
                        help="Option to export uff model")
    parser.add_argument("--output_node_names", type=str, default="output_name",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    graph, ckpt_name, sess = freeze_graph(args.model_dir, args.output_node_names)

    if args.export_uff:
        ckpt_name = "export/exported/%s" %ckpt_name
        optimize_pb_graph(graph, [args.output_node_names], ckpt_name, sess)
