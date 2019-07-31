import os, argparse
import tensorflow as tf
from builders import model_builder

from tensorflow.python.platform import gfile


dir = os.path.dirname(os.path.realpath(__file__))

num_classes = 17


def freeze_graph(args, output_node_names):
    if not tf.gfile.Exists(args.model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % args.model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    # checkpoint = tf.train.get_checkpoint_state(model_dir)
    # input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(args.model_dir.split('/')[:-1])
    output_graph = os.path.join(absolute_model_dir, "unfrozen_model_softmax_output.pb")

    with tf.Graph().as_default() as graph:
        net_input = tf.placeholder(tf.float32, shape=[1, args.crop_height, args.crop_width, 3])  # NHWC format

        net, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input,
                                                 num_classes=num_classes, crop_width=args.crop_width,
                                                 crop_height=args.crop_height, is_training=False)

        net = tf.nn.softmax(net, name="output_name_softmax")
        net = tf.argmax(net, name="output_name_argmax", axis=3)

        graph_def = graph.as_graph_def()
        with gfile.GFile(output_graph, 'wb') as f:
            f.write(graph_def.SerializeToString())
            print('Successfull written to', output_graph)



    # return output_graph_def


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="Path to checkpoint model directory",
                        help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="output_name",
                        help="The name of the output nodes, comma separated.")
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
    parser.add_argument('--model', type=str, default="MobileUNet",
                        help='The model you are using. See model_builder.py for supported models')
    parser.add_argument('--frontend', type=str, default="MobileNetV2",
                        help='The frontend you are using. See frontend_builder.py for supported models')
    args = parser.parse_args()

    freeze_graph(args, args.output_node_names)
