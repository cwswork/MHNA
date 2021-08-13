import os

class Myprint:
    def __init__(self, filePath, filename):
        if not os.path.exists(filePath):
            print('output not exists' + filePath)
            os.makedirs(filePath)

        self.outfile = filePath + filename

    def print(self, print_str):
        print(print_str)
        ''' Save log file '''
        with open(self.outfile, 'a', encoding='utf-8') as fw:
            fw.write('{}\n'.format(print_str))

