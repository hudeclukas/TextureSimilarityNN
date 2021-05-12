import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os


class SimilarityDataset(Dataset):
    def __init__(self, img_dir, max_samples=30, image_size=150, transform=None, max_images=-1):
        super(SimilarityDataset,self).__init__()
        np.random.seed(42)
        self.img_dir = img_dir
        self.transform = transform
        self.all_image_shapes = []
        self.max_samples = max_samples
        self.image_size = image_size
        self.max_images = max_images
        self.__load_image_names(max_images)
        self.__sample_images()
        self.indices = np.arange(len(self.all_image_names), dtype=np.int32)
        self.all_image_names = np.array(self.all_image_names)
        self.all_image_shapes = np.array(self.all_image_shapes)

    def __load_image_names(self, max_images):
        self.all_image_names = os.listdir(self.img_dir)
        join_path = lambda x: os.path.join(self.img_dir, x)
        is_folder = lambda x: not os.path.isdir(x)
        self.all_image_names = map(join_path, self.all_image_names)
        self.all_image_names = list(filter(is_folder, self.all_image_names))
        shape = lambda x: self._read_image(x).shape
        self.all_image_shapes = list(map(shape, tqdm(self.all_image_names, desc='Data')))  # tqdm for nicer progress tracking
        if max_images > 0:
            self.__max_images_constraint(max_images)
        else:
            self.image_names = np.array(self.all_image_names)
            self.max_images = len(self.all_image_names)                                     # set max_images when -1
        if len(self.image_names) == 0:
            print("Path empty: ", self.img_dir)
            raise IOError

    def __shuffle_all_images(self):
        np.random.shuffle(self.indices)
        self.all_image_names = self.all_image_names[self.indices]
        self.all_image_shapes = self.all_image_shapes[self.indices]

    def __max_images_constraint(self, max_images):
        self.image_names = self.all_image_names[:max_images]
        self.image_shapes = self.all_image_shapes[:max_images]

    def __sample_images(self):
        self.image_sample_indices = []
        for im_idx in range(len(self.all_image_shapes)):
            sample_idxs = np.random.random((self.max_samples,2))
            sample_idxs[:,0] *= self.all_image_shapes[im_idx][1] - self.image_size # it is just a shape, not whole image
            sample_idxs[:,1] *= self.all_image_shapes[im_idx][2] - self.image_size # it is just a shape, not whole image
            self.image_sample_indices.append(np.array(sample_idxs,dtype=int))

    def _read_image(self, x):
        gray_image = np.asarray(cv2.imread(x, cv2.IMREAD_GRAYSCALE) / 255, dtype=np.float32)
        if len(gray_image.shape) == 2:
            gray_image = np.expand_dims(gray_image, axis=2)
        return np.transpose(gray_image, axes=[2, 0, 1])

    def get_samples(self, item):
        image = self._read_image(self.image_names[item])
        output = [image[:,x:x+self.image_size,y:y+self.image_size] for x,y in self.image_sample_indices[item]]
        return torch.from_numpy(np.array(output))

    def reset(self, other_images=False):
        if other_images:
            self.__shuffle_all_images()
            self.image_names = self.all_image_names[:self.max_images]
            self.__max_images_constraint(self.max_images)
        self.__sample_images()

    def __len__(self):
        return self.max_images * self.max_samples

    def __getitem__(self, index):
        image_idx = index // self.max_samples
        image_sample_idx = index % self.max_samples
        label = None
        image1 = None
        image2 = None
        if image_sample_idx % 2 == 1:
            label = 1.0
            idx1 = np.random.choice(len(self.image_sample_indices[image_idx]))
            idx1 = self.image_sample_indices[image_idx][idx1]
            image1 = self._read_image(self.image_names[image_idx])[:,idx1[0]:idx1[0]+self.image_size, idx1[1]:idx1[1]+self.image_size]
            idx2 = np.random.choice(len(self.image_sample_indices[image_idx]))
            idx2 = self.image_sample_indices[image_idx][idx2]
            image2 = self._read_image(self.image_names[image_idx])[:,idx2[0]:idx2[0]+self.image_size, idx1[1]:idx1[1]+self.image_size]
        if image_sample_idx % 2 == 0:
            label = 0.0
            idx1 = np.random.choice(len(self.image_sample_indices[image_idx]))
            idx1 = self.image_sample_indices[image_idx][idx1]
            image1 = self._read_image(self.image_names[image_idx])[:,idx1[0]:idx1[0] + self.image_size, idx1[1]:idx1[1] + self.image_size]
            image_idx2 = np.random.randint(0, self.max_images)
            if image_idx == image_idx2:
                image_idx2 = np.random.randint(0, self.max_images)
            idx2 = np.random.choice(len(self.image_sample_indices[image_idx2]))
            idx2 = self.image_sample_indices[image_idx2][idx2]
            image2 = self._read_image(self.image_names[image_idx2])[:,idx2[0]:idx2[0] + self.image_size, idx2[1]:idx2[1] + self.image_size]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return torch.from_numpy(image1), torch.from_numpy(image2), torch.from_numpy(np.array(label,dtype=np.float32))


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    import matplotlib.pyplot as plt

    data = SimilarityDataset('data/train', max_samples=50, max_images=4)
    data.reset(other_images=True)
    # all_samples = data.get_samples(0)
    batch_size = 32
    train_dataloader = DataLoader(data, batch_size, shuffle=True, num_workers=8)
    epoch_start = time.time()
    start = time.time()
    for batch, (x1, x2, y) in tqdm(enumerate(train_dataloader)):
        print(f"x1.shape: {x1.shape}, x2.shape: {x2.shape}, y.shape: {y.shape}, Simillars: {torch.sum(y).item()}")
        end = time.time()
        print(f"batch time: {end-start}")
        start = end

        _, axs = plt.subplots(1, batch_size, sharey=True, sharex=True, figsize=[19.6, 1.7])
        for ii, (i1, i2, t) in enumerate(zip(x1, x2, y)):
            axs[ii].imshow(np.concatenate((i1, i2), axis=1).transpose([1, 2, 0]), cmap='gray')
            axs[ii].set_title("Y: {:.1f}".format(t.item()))
        plt.show()
        plt.close()
    epoch_end = time.time()
    print(epoch_end-epoch_start)
