import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# DATASET_FOLDER = "/Users/tolgaozgun/gazi_brains_2020/data/GAZI_BRAINS_2020/sourcedata"
DATASET_FOLDER = "/tf/shared-datas/TurkBeyinProjesi/GaziBrains_BIDS/GAZI_BRAINS_2020/sourcedata/"

GAZI_CYCLEGAN_OUTPUT = "./gazi_cyclegan"
GAZI_PIX2PIX_OUTPUT = "./gazi_pix2pix"
CREATE_SUB_FOLDERS = True

VALIDATION_PERCENTAGE = 0.2

cyclegan_trainA_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "trainA")
cyclegan_trainB_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "trainB")
cyclegan_valA_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "valA")
cyclegan_valB_folder = os.path.join(GAZI_CYCLEGAN_OUTPUT, "valB")
pix2pix_trainA_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "A", "train")
pix2pix_trainB_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "B", "train")
pix2pix_valA_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "A", "val")
pix2pix_valB_folder = os.path.join(GAZI_PIX2PIX_OUTPUT, "B", "val")

# Create trainA and trainB directories if they don't exist
os.makedirs(cyclegan_trainA_folder, exist_ok=True)
os.makedirs(cyclegan_trainB_folder, exist_ok=True)
os.makedirs(cyclegan_valA_folder, exist_ok=True)
os.makedirs(cyclegan_valB_folder, exist_ok=True)
os.makedirs(pix2pix_trainA_folder, exist_ok=True)
os.makedirs(pix2pix_trainB_folder, exist_ok=True)
os.makedirs(pix2pix_valA_folder, exist_ok=True)
os.makedirs(pix2pix_valB_folder, exist_ok=True)

def load_data_from_folder(folder_path):

    base_name = os.path.basename(folder_path)
    folder_path = os.path.join(folder_path, "anat")
    
    # Load mandatory files
    t1w_path = os.path.join(folder_path, f"{base_name}_T1w.nii.gz")
    flair_path = os.path.join(folder_path, f"{base_name}_FLAIR.nii.gz")
    t2w_path = os.path.join(folder_path, f"{base_name}_T2w.nii.gz")
    t1w_img = nib.load(t1w_path).get_fdata()
    flair_img = nib.load(flair_path).get_fdata()
    t2w_img = nib.load(t2w_path).get_fdata()

    # Load optional files if available
    gadolinium_t1w_path = os.path.join(folder_path, f"{base_name}_ce-GADOLINIUM_T1w.nii.gz")
    gadolinium_t1w_img = nib.load(gadolinium_t1w_path).get_fdata()


    return flair_img, t1w_img, t2w_img, gadolinium_t1w_img
    
parse_counter = 0

def parse_images(flair_imgs, t1w_imgs, t2w_imgs, gadolinium_t1w_imgs, sub_no):
    assert(t1w_imgs.shape[2] == t2w_imgs.shape[2] == flair_imgs.shape[2] == gadolinium_t1w_imgs.shape[2])
    no_of_sequences = t1w_imgs.shape[2]

    sequences = range(0, no_of_sequences)
    sequences_train, sequences_val = train_test_split(sequences, test_size=0.2, random_state=42)

    if CREATE_SUB_FOLDERS: 
        cyclegan_train_sub_path = os.path.join(cyclegan_trainA_folder, sub_no)
        cyclegan_val_sub_path = os.path.join(cyclegan_valA_folder, sub_no)
        os.makedirs(cyclegan_train_sub_path, exist_ok=True)
        os.makedirs(cyclegan_val_sub_path, exist_ok=True)

        pix2pix_train_sub_path = os.path.join(pix2pix_trainA_folder, sub_no)
        pix2pix_val_sub_path = os.path.join(pix2pix_valA_folder, sub_no)
        os.makedirs(pix2pix_train_sub_path, exist_ok=True)
        os.makedirs(pix2pix_val_sub_path, exist_ok=True)

    for i in range(0, no_of_sequences):
        t1w_img = t1w_imgs[..., i]
        t2w_img = t2w_imgs[..., i]
        flair_img = flair_imgs[..., i]
        gadolinium_t1w_img = gadolinium_t1w_imgs[..., i]

        assert(t1w_img.shape == t2w_img.shape == flair_img.shape == gadolinium_t1w_img.shape)

        concatenated_img = np.concatenate([t1w_img[..., np.newaxis], t2w_img[..., np.newaxis], flair_img[..., np.newaxis]], axis=-1)

        concatenated_img = rescale_image(concatenated_img)
        gadolinium_t1w_img = rescale_image(gadolinium_t1w_img)

        img_A_data = Image.fromarray(concatenated_img)
        img_B_data = Image.fromarray(gadolinium_t1w_img)

        if i in sequences_val:

            # CycleGAN validation dataset

            if CREATE_SUB_FOLDERS: 
                t1w_img = rescale_image(t1w_img)
                t2w_img = rescale_image(t2w_img)
                flair_img = rescale_image(flair_img)
                img_t1w_data = Image.fromarray(t1w_img)
                img_t2w_data = Image.fromarray(t2w_img)
                img_flair_data = Image.fromarray(flair_img)
                img_t1w_data.save(os.path.join(cyclegan_val_sub_path, f"{sub_no}_seq{i}_t1w.png"))
                img_t2w_data.save(os.path.join(cyclegan_val_sub_path, f"{sub_no}_seq{i}_t2w.png"))
                img_flair_data.save(os.path.join(cyclegan_val_sub_path, f"{sub_no}_seq{i}_flair.png"))

            img_A_data.save(os.path.join(cyclegan_valA_folder, f"{sub_no}_seq{i}.png"))
            img_B_data.save(os.path.join(cyclegan_valB_folder, f"{sub_no}_seq{i}.png"))


            # Pix2pix validation dataset

            if CREATE_SUB_FOLDERS: 
                t1w_img = rescale_image(t1w_img)
                t2w_img = rescale_image(t2w_img)
                flair_img = rescale_image(flair_img)
                img_t1w_data = Image.fromarray(t1w_img)
                img_t2w_data = Image.fromarray(t2w_img)
                img_flair_data = Image.fromarray(flair_img)
                img_t1w_data.save(os.path.join(pix2pix_val_sub_path, f"{sub_no}_seq{i}_t1w.png"))
                img_t2w_data.save(os.path.join(pix2pix_val_sub_path, f"{sub_no}_seq{i}_t2w.png"))
                img_flair_data.save(os.path.join(pix2pix_val_sub_path, f"{sub_no}_seq{i}_flair.png"))

            img_A_data.save(os.path.join(pix2pix_valA_folder, f"{sub_no}_seq{i}.png"))
            img_B_data.save(os.path.join(pix2pix_valB_folder, f"{sub_no}_seq{i}.png"))

        else:

            # Cyclegan training dataset

            if CREATE_SUB_FOLDERS: 
                t1w_img = rescale_image(t1w_img)
                t2w_img = rescale_image(t2w_img)
                flair_img = rescale_image(flair_img)
                img_t1w_data = Image.fromarray(t1w_img)
                img_t2w_data = Image.fromarray(t2w_img)
                img_flair_data = Image.fromarray(flair_img)
                img_t1w_data.save(os.path.join(cyclegan_train_sub_path, f"{sub_no}_seq{i}_t1w.png"))
                img_t2w_data.save(os.path.join(cyclegan_train_sub_path, f"{sub_no}_seq{i}_t2w.png"))
                img_flair_data.save(os.path.join(cyclegan_train_sub_path, f"{sub_no}_seq{i}_flair.png"))

            img_A_data.save(os.path.join(cyclegan_trainA_folder, f"{sub_no}_seq{i}.png"))
            img_B_data.save(os.path.join(cyclegan_trainB_folder, f"{sub_no}_seq{i}.png"))

            # Pix2pix training dataset

            if CREATE_SUB_FOLDERS: 
                t1w_img = rescale_image(t1w_img)
                t2w_img = rescale_image(t2w_img)
                flair_img = rescale_image(flair_img)
                img_t1w_data = Image.fromarray(t1w_img)
                img_t2w_data = Image.fromarray(t2w_img)
                img_flair_data = Image.fromarray(flair_img)
                img_t1w_data.save(os.path.join(pix2pix_train_sub_path, f"{sub_no}_seq{i}_t1w.png"))
                img_t2w_data.save(os.path.join(pix2pix_train_sub_path, f"{sub_no}_seq{i}_t2w.png"))
                img_flair_data.save(os.path.join(pix2pix_train_sub_path, f"{sub_no}_seq{i}_flair.png"))

            img_A_data.save(os.path.join(pix2pix_trainA_folder, f"{sub_no}_seq{i}.png"))
            img_B_data.save(os.path.join(pix2pix_trainB_folder, f"{sub_no}_seq{i}.png"))

    
    global parse_counter
    parse_counter += 1
    print(f"{sub_no} completed. Total count: {parse_counter}")

def rescale_image(image):
    # Find the minimum and maximum values in the image

    min_val = np.min(image)
    max_val = np.max(image)

    # Scale the image to the range of 0 to 255
    scaled_image = (image - min_val) * (255.0 / (max_val - min_val))

    scaled_image = scaled_image.astype(np.uint8)
    
    return scaled_image

def get_valid_data_folders(root_dir):
    data_folders = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        anat_path = os.path.join(folder_path, "anat")
        if os.path.isdir(folder_path) and os.path.isdir(anat_path) and has_optional_files(folder_path, anat_path):
            flair_imgs, t1w_imgs, t2w_imgs, gadolinium_t1w_imgs = load_data_from_folder(folder_path)
            base_name = os.path.basename(folder_path)
            parse_images(flair_imgs, t1w_imgs, t2w_imgs, gadolinium_t1w_imgs, base_name)

    return data_folders

def has_optional_files(folder_path, anat_path):
    base_name = os.path.basename(folder_path)
    optional_files = [f'{base_name}_ce-GADOLINIUM_T1w.json', f'{base_name}_ce-GADOLINIUM_T1w.nii.gz']
    for file_name in optional_files:
        file_path = os.path.join(anat_path, file_name)
        if not os.path.exists(file_path):
            return False
    return True


def main():
    get_valid_data_folders(DATASET_FOLDER)


if __name__ == "__main__":
    main()

