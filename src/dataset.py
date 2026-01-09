from torch.utils.data import Dataset as TorchDataset
import json
import os
import torch
import numpy as np
from pathlib import Path

class RawTokenDataset(TorchDataset):
    """ Loads raw uint32 tokens as memmap-backed array """
    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of training frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata["s"], self.metadata["s"])
        video_tokens_path, segment_ids_path, action_tokens_path = [data_dir / f"{name}.bin"
                                                                   for name in ["video", "segment_ids", "actions"]]
        token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))
        self.data = np.memmap(video_tokens_path, dtype=token_dtype, mode="r", shape=shape)
        # self.actions = np.memmap(action_tokens_path, dtype=np.uint16, mode="r", shape=(self.metadata["num_images"],))

        if os.path.isfile(segment_ids_path):
            self.segment_ids = np.memmap(
                segment_ids_path,
                dtype=np.int32,
                mode="r",
                shape=(self.metadata["num_images"],)
            )
        else:
            self.segment_ids = None
            if filter_interrupts:
                raise NotImplementedError("Cannot filter interrupted sequences without segment ids.")

        self.window_size, self.stride = window_size, stride
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride

        self.valid_start_inds = []
        for start_ind in range(len(self.data) - self.video_len):
            # Assuming `segment_ids` is monotonically increasing, a sequence is interrupted
            # if the first and last frames have different segment ids.
            if not (filter_interrupts and self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        x = torch.from_numpy((self.data[start_ind : start_ind + self.video_len + 1 : self.stride]).astype(np.int64))
        x = x.flatten()

        attention_mask = torch.ones_like(x)
        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
        }