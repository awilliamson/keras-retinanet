import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator


class VocPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(VocPlugin, self).__init__()

        self.dataset_type = "pascal"

    def parse_args(self, subparsers):
        pascal_parser = subparsers.add_parser(self.dataset_type)
        pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

        return pascal_parser

    def get_generator(self, args, transform_generator=None):
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            transform_generator=transform_generator,
            batch_size=args.batch_size
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            batch_size=args.batch_size
        )

        return {
            "train_generator": train_generator,
            "validation_generator": validation_generator
        }