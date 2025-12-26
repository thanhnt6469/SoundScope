import os

main_dir = './11_lfcc_01/'
dir_01 = 'train/real'
dir_02 = 'train/fake'

dir_01 = os.path.join(main_dir, dir_01)
dir_02 = os.path.join(main_dir, dir_02)

dir_03 = 'dev/real'
dir_04 = 'dev/fake'

dir_03 = os.path.join(main_dir, dir_03)
dir_04 = os.path.join(main_dir, dir_04)

print('--------Train Real:')
print(len(os.listdir(dir_01)))
print('--------Train Fake:')
print(len(os.listdir(dir_02)))
print('--------Dev Real:')
print(len(os.listdir(dir_03)))
print('--------Dev Fake:')
print(len(os.listdir(dir_04)))


