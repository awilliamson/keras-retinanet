from yapsy.IPlugin import IPlugin


class DatasetPlugin(IPlugin):
    def __init__(self):
        self.dataset_type = None
        super(DatasetPlugin, self).__init__()


    def parser_args(self, parser):
        pass

    def check_args(self, parsed_args):
        pass

    def get_generator(self):
        pass