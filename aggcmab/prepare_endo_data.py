import os, sys
import os.path as osp
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

from PIL import Image
from torchvision.transforms import Resize

rsz = Resize((512,640), interpolation=Image.BILINEAR)
rsz_bin = Resize((512,640), interpolation=Image.NEAREST)

im_dir = 'data/trainData_EndoCV2021_21_Feb2021-V2/'
pos_seq_dir = 'data/trainData_EndoCV2021_21_Feb2021-V2/sequenceData/positive/'
im_path_out = 'data/images'
mask_path_out = 'data/masks'
os.makedirs(im_path_out, exist_ok=True)
os.makedirs(mask_path_out, exist_ok=True)

im_list_all = []
mask_list_all = []
center_list_all = []
for root, centers, files in os.walk(im_dir):
    for center in sorted(centers):
        if 'sequence' in center: continue
        # if int(center[-1])!=3: continue
        center_path = osp.join(im_dir, center)
        im_fold = center.replace('data', 'images')
        mask_fold = center.replace('data', 'masks')
        im_path = osp.join(center_path, im_fold)
        mask_path =  osp.join(center_path, mask_fold)
        im_list = sorted(os.listdir(im_path))
        mask_list = [n.replace('images', 'masks').replace('.jpg', '_mask.jpg') for n in im_list]
        im_list = [osp.join(im_path, n) for n in im_list]
        mask_list = [osp.join(mask_path, n) for n in mask_list]
        im_list_all.extend(im_list)
        mask_list_all.extend(mask_list)
        center_nr = int(center[-1])
        center_list_all.extend(len(im_list)*[center_nr])
        print(center, center_path, im_fold, mask_fold)
    break

# subsample positive sequence data by 7, ignore negative sequence data
for seq in os.listdir(pos_seq_dir):
    if osp.isdir(osp.join(pos_seq_dir, seq)):
        im_list = sorted(os.listdir(osp.join(pos_seq_dir, seq, 'images')))
        im_list = [osp.join(pos_seq_dir, seq, 'images', n) for n in im_list]
        mask_list = [n.replace('images', 'masks').replace('.jpg', '_mask.jpg') for n in im_list]

        im_list = im_list[::7]
        mask_list = mask_list[::7]

        seq_nr = int(seq.replace('seq', ''))
        im_list_all.extend(im_list)
        mask_list_all.extend(mask_list)
        center_list_all.extend(len(im_list)*[6])

im_list_all_final, mask_list_all_final = [], []
for i in tqdm(range(len(im_list_all))):
    im_name_in = im_list_all[i]
    mask_name_in = mask_list_all[i]

    im_in = Image.open(im_name_in)
    mask_in = Image.open(mask_name_in)

    im_name_out = osp.join(im_path_out, im_name_in.split('/')[-1])
    mask_name_out = osp.join(mask_path_out, mask_name_in.split('/')[-1])

    # resize
    rsz(im_in).save(im_name_out)
    rsz_bin(mask_in).save(mask_name_out)


    # im_in.resize((512,512), Image.BILINEAR).save(im_name_out)
    # mask_in.resize((512, 512), Image.NEAREST).save(mask_name_out)

    im_list_all_final.append(im_name_out)
    mask_list_all_final.append(mask_name_out)


data_tuples = list(zip(im_list_all_final,mask_list_all_final,center_list_all))
df_all = pd.DataFrame(data_tuples, columns=['image_path','mask_path','center'])

num_ims = len(im_list_all_final)
meh, df_val1 = train_test_split(df_all, test_size=num_ims//4, random_state=0, stratify=df_all.center)
meh, df_val2 = train_test_split(meh,    test_size=num_ims//4, random_state=0, stratify=meh.center)
df_val4, df_val3 = train_test_split(meh,    test_size=num_ims//4, random_state=0, stratify=meh.center)

df_train1 = pd.concat([df_val2,df_val3,df_val4], axis=0)
df_train2 = pd.concat([df_val1,df_val3,df_val4], axis=0)
df_train3 = pd.concat([df_val1,df_val2,df_val4], axis=0)
df_train4 = pd.concat([df_val1,df_val2,df_val3], axis=0)

df_train1.to_csv('data/train_1.csv', index=None)
df_val1.to_csv('data/val_1.csv', index=None)

df_train2.to_csv('data/train_2.csv', index=None)
df_val2.to_csv('data/val_2.csv', index=None)

df_train3.to_csv('data/train_3.csv', index=None)
df_val3.to_csv('data/val_3.csv', index=None)

df_train4.to_csv('data/train_4.csv', index=None)
df_val4.to_csv('data/val_4.csv', index=None)

df_train, df_val = train_test_split(df_all, test_size=num_ims//5, random_state=0, stratify=df_all.center)
df_train.to_csv('data/train.csv', index=None)
df_val.to_csv('data/val.csv', index=None)


# os.makedirs('data/local_val_data', exist_ok=True)
#
# in_dir = osp.join(im_dir, 'data_C1', 'images_C1/')
# out_dir = osp.join('data/local_val_data', 'data_C1')
# os.makedirs(out_dir, exist_ok=True)
# val_center1 = list(df_val[df_val.center==1].image_path)
#
# for n in val_center1:
#     src = osp.join(in_dir, n.split('/')[-1])
#     dst = osp.join(out_dir, n.split('/')[-1])
#     shutil.copyfile(src, dst)
#
#
# in_dir = osp.join(im_dir, 'data_C2', 'images_C2/')
# out_dir = osp.join('data/local_val_data', 'data_C2')
# os.makedirs(out_dir, exist_ok=True)
# val_center2 = list(df_val[df_val.center==2].image_path)
#
# for n in val_center2:
#     src = osp.join(in_dir, n.split('/')[-1])
#     dst = osp.join(out_dir, n.split('/')[-1])
#     shutil.copyfile(src, dst)
#
#
# in_dir = osp.join(im_dir, 'data_C3', 'images_C3/')
# out_dir = osp.join('data/local_val_data', 'data_C3')
# os.makedirs(out_dir, exist_ok=True)
# val_center3 = list(df_val[df_val.center==3].image_path)
#
# for n in val_center3:
#     src = osp.join(in_dir, n.split('/')[-1])
#     dst = osp.join(out_dir, n.split('/')[-1])
#     shutil.copyfile(src, dst)
#
#
# in_dir = osp.join(im_dir, 'data_C4', 'images_C4/')
# out_dir = osp.join('data/local_val_data', 'data_C4')
# os.makedirs(out_dir, exist_ok=True)
# val_center4 = list(df_val[df_val.center==4].image_path)
#
# for n in val_center4:
#     src = osp.join(in_dir, n.split('/')[-1])
#     dst = osp.join(out_dir, n.split('/')[-1])
#     shutil.copyfile(src, dst)
#
# in_dir = osp.join(im_dir, 'data_C5', 'images_C5/')
# out_dir = osp.join('data/local_val_data', 'data_C5')
# os.makedirs(out_dir, exist_ok=True)
# val_center5 = list(df_val[df_val.center==5].image_path)
#
# for n in val_center5:
#     src = osp.join(in_dir, n.split('/')[-1])
#     dst = osp.join(out_dir, n.split('/')[-1])
#     shutil.copyfile(src, dst)