import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

from keras_retinanet.preprocessing.csv_generator import CSVGenerator


class CSVPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(CSVPlugin, self).__init__()

        self.dataset_type = "csv"

    def parser_args(self, subparsers):
        csv_parser = subparsers.add_parser(self.dataset_type)
        csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
        csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
        csv_parser.add_argument('--val-annotations',
                                help='Path to CSV file containing annotations for validation (optional).')

        return csv_parser

    def get_generator(self, args, transform_generator=None):
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            batch_size=args.batch_size
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                batch_size=args.batch_size
            )
        else:
            validation_generator = None

        return {
            "train_generator": train_generator,
            "validation_generator": validation_generator
        }