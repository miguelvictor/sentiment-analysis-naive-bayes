import os
import shutil
import urllib.request

if __name__ == '__main__':
    print('INFO - Downloading dataset archive ...')
    urllib.request.urlretrieve(
        'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', 'aclImdb_v1.tar.gz')

    print('INFO - Unpacking archive ...')
    shutil.unpack_archive('aclImdb_v1.tar.gz', '.')

    print('INFO - Cleaning up ...')
    os.unlink('aclImdb_v1.tar.gz')
