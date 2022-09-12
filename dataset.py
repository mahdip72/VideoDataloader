import torch
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, csv_dir, max_frame, frames_per_segment, num_segments, target_size):

        self.pairs = self.prepare_samples(csv_dir=csv_dir)
        self.num_samples = len(self.pairs)
        self.target_size = target_size
        self.frames_per_segment = frames_per_segment
        self.num_segments = num_segments
        self.max_frame = max_frame

        print('number of frames:', self.frames_per_segment * self.num_segments)

    @staticmethod
    def prepare_samples(csv_dir):
        df = pd.read_csv(csv_dir)
        samples = []
        for index, video_dir, label in df.itertuples():
            samples.append((video_dir, label))
        return samples

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def normalise_image(img):
        """Normalises image data to be a float between 0 and 1
        """
        img = img.astype('float32') / 255
        return img

    def load_video(self, video_dir):
        video = []
        cap = cv2.VideoCapture(video_dir)

        while cap.isOpened() and len(video) < self.max_frame:
            ret, frame = cap.read()
            if ret:
                video.append(cv2.resize(self.normalise_image(frame[..., ::-1]), self.target_size))
            else:
                break

        cap.release()
        return np.array(video)

    def compute_indices(self, num_frames):
        max_valid_start_index = (num_frames - self.frames_per_segment + 1) // self.num_segments

        return np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
               np.random.randint(max_valid_start_index, size=self.num_segments)

    def __getitem__(self, idx):
        input_dir, label = self.pairs[idx]
        video = self.load_video(input_dir)

        num_frames = len(video)
        indices = self.compute_indices(num_frames)
        video = video[indices]
        video = torch.tensor(video)
        return video, label


def plot_video(frames):
    for frame in frames:
        cv2.imshow('Frame', frame[..., ::-1])
        cv2.waitKey(30)


if __name__ == '__main__':
    data_path = 'samples.csv'

    batch = 2
    training_data = CustomDataset(
        csv_dir=data_path,
        target_size=(320, 200),
        max_frame=80,
        frames_per_segment=1,
        num_segments=80
        )
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=False)

    for i, j in train_dataloader:
        i = i.cpu().numpy()
        for sample_number in range(i.shape[0]):
            vid = i[sample_number]
            lab = j[sample_number]

            print('label:', lab)
            plot_video(vid)
