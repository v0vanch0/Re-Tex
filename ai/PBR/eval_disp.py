import glob
import os

import torch
import torchvision.utils
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
PATH_CHK = "checkpoints/disp/disp_net_last.pth"
CROP = 1024

# %%
transform = transforms.Compose([
    transforms.Resize(CROP),
    transforms.CenterCrop(CROP),
    transforms.ToTensor()
    # outputs range from -1 to 1
])

transformDoNotResize = transforms.Compose([
    transforms.ToTensor()
    # outputs range from -1 to 1
])

class TestDataset(Dataset):
    def __init__(self, img_dir, single=False):
        if (single):
            self.file_list = glob.glob(img_dir)
            self.names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_list]
            return

        self.file_list = glob.glob(img_dir + "/*.png")
        self.names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.file_list[i]).convert('RGB')

        h, w = img.size

        if w < 256 or h < 256 or w - 300 > h or h - 300 > w or w > 1024 or h > 1024:
            img = transform(img)
        else:
            img = transformDoNotResize(img)
        return img, self.names[i]


# %% test
def generateDisp(net, DIR_FROM, DIR_EVAL):
    output_normal = DIR_EVAL
    if not os.path.exists(output_normal):
        os.makedirs(output_normal)

    data_test = TestDataset(DIR_FROM)
    testloader = DataLoader(data_test, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=12,
                            persistent_workers=True)

    print("\nProcessing displacement files...")

    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            img_in = data[0].cuda()
            # Разделение тензора на куски по 256 элементов
            split_size = 256
            splits = torch.split(img_in, split_size, dim=0)

            # Прогон каждого куска через модель
            processed_splits = []
            for split in splits:
                img_out = net(split)
                processed_splits.append(img_out)

            # Соединение результатов обратно в один тензор
            img_out_combined = torch.cat(processed_splits, dim=0)

            name = f"{data[1][0]}_disp"
            img_out_filename = os.path.join(output_normal, f"{name}.png")
            torchvision.utils.save_image(img_out_combined, img_out_filename, value_range=(-1, 1), normalize=True)

            im = Image.open(img_out_filename).convert("L")
            im.save(img_out_filename)

    print("Done!")


def generateDispSingle(net, DIR_FROM, DIR_EVAL):
    output_normal = DIR_EVAL
    if not os.path.exists(output_normal):
        os.makedirs(output_normal)

    data_test = TestDataset(DIR_FROM, True)
    # print(batch_size)
    testloader = DataLoader(data_test, batch_size=1, shuffle=False,
                            pin_memory=True)

    print("\nOutput disp files...")

    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            img_in = data[0].cuda()
            img_out = net(img_in)
            name = f"{data[1][0]}_disp"

            img_out_filename = os.path.join(output_normal, f"{name}.png")
            torchvision.utils.save_image(img_out, img_out_filename, value_range=(-1, 1), normalize=True)

            im = Image.open(img_out_filename).convert("L")
            im.save(img_out_filename)

    print("Done!")


if __name__ == "__main__":
    from model import span

    model = span()
    model.load_state_dict(torch.load("./checkpoints/Displacement/last.pth"), strict=False)
    model.cuda().share_memory()
    # Set the model to evaluation mode (e.g., for batch normalization and dropout)
    model.eval()

    generateDisp(model, "./textures", "./out")
