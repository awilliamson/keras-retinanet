import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

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
