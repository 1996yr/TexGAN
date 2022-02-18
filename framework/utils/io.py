import zipfile, os, sys
import numpy as np


def dfs_get_zip_file(src_input_path, current_input_path, result_full_path, result_relative_path):
    files = os.listdir(os.path.join(src_input_path, current_input_path))
    for file in files:
        if os.path.isdir(os.path.join(src_input_path, current_input_path, file)):
            dfs_get_zip_file(src_input_path, os.path.join(current_input_path, file), result_full_path, result_relative_path)
        else:
            result_full_path.append(os.path.join(src_input_path, current_input_path, file))
            result_relative_path.append(os.path.join(current_input_path, file))

def zip_folder(input_path, output_fn):
    assert os.path.exists(input_path)
    full_path_list = []
    relative_path_list = []
    dfs_get_zip_file(input_path, '', full_path_list, relative_path_list)
    f = zipfile.ZipFile(output_fn, 'w', zipfile.ZIP_DEFLATED)
    assert len(full_path_list) == len(relative_path_list)
    for file_id in range(len(full_path_list)):
        f.write(full_path_list[file_id], relative_path_list[file_id])
    f.close()
    return output_fn

def save_pfm(filepath, img, reverse = 1):
    color = None
    file = open(filepath, 'wb')
    if(img.dtype.name != 'float32'):
        img = img.astype(np.float32)

    color = True if (len(img.shape) == 3) else False

    if(reverse and color):
        img = img[:,:,::-1]

    img = img[::-1,...]

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (img.shape[1], img.shape[0]))

    endian = img.dtype.byteorder
    scale = 1.0
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)
    img.tofile(file)
    file.close()

def load_pfm(filepath, reverse = 1):
    file = open(filepath, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    color = (header == b'PF')

    width, height = map(int, file.readline().strip().decode('ascii').split(' '))
    scale = float(file.readline().rstrip().decode('ascii'))
    endian = '<' if(scale < 0) else '>'
    scale = abs(scale)

    rawdata = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    file.close()

    if(color):
        return rawdata.reshape(shape).astype(np.float32)[::-1,:,::-1]
    else:
        return rawdata.reshape(shape).astype(np.float32)[::-1,:]

def get_fn_from_txt(txt_fn):
    f = open(txt_fn, 'r')
    fn_list = []
    for line in f.readlines():
        if '\n' in line:
            line = line[0:-1]
        if len(line)>0:
            fn_list.append(line)
    return fn_list

if __name__ == '__main__':
    try:
        zip_folder(r'F:\ruiyu\temp', r'F:\ruiyu\temp.zip')
    except:
        print('zip_folder() api failed')
