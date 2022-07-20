from easyocr.easyocr import *
import  os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# GPU 설정
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


if __name__ == '__main__':

    # # Using default model
    # reader = Reader(['en'], gpu=True)

    # Using custom model
    reader = Reader(['ko', 'en'], gpu=False,
                    model_storage_directory='./workspace/user_network_dir',
                    user_network_directory='./workspace/user_network_dir',    #   ./easyOCR/workspace/user_network_dir  
                    recog_network='PGOCR_0112')   # korean_g2  # PGOCR_0112

    files, count = get_files('C:\\Users\\powegen\\Desktop\\EasyOCR\\deep-text-recognition-benchmark-master\\pictures')
    # 'C:\\Users\\powegen\\Desktop\\EasyOCR\\deep-text-recognition-benchmark-master\\pictures'
    for idx, file in enumerate(files):
        filename = os.path.basename(file)

        result = reader.readtext(file)

        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for (bbox, string, confidence) in result:
            print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))
            # print('bbox: ', bbox)
