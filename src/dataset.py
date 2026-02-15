import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class TrainDataset(TorchDataset):
    """
    - video shards をグローバル連結
    - window = 6 token (past3 + future3)
    - stride = 3 token（= 3トークン単位でスライド）
    - segment_idx (raw) を用いて、境界混入する3トークンブロックを除外
      さらに pastブロックと futureブロックが同一clipのものだけ残す
    - 追加: robot_states/states_{i}.bin から raw 34 (=17+17) フレーム分の状態(25次元)も返す
    """

    def __init__(
        self,
        root: str | Path,
        s: int = 32,
        past_frames: int = 3,
        future_frames: int = 3,
        stride_tokens: int = 3,                 # 0-5,3-8,... を作る
        token_dtype: np.dtype = np.int32,       # video_*.bin の dtype
        seg_dtype: np.dtype = np.int32,         # segment_idx_*.bin の dtype
        state_dtype: np.dtype = np.float32,     # states_*.bin の dtype
        state_dim: int = 25,                    # states の次元
        output_format: str = "seq2seq",         # "seq2seq" or "causal_lm"
        device: str | None = None,
        cache_path: str | Path | None = None,   # valid_starts を保存/ロード
        use_fixed_17to3_mapping: bool = True,   # raw=floor(tok*17/3)
        chunk_tokens: int = 1_000_000,          # 前計算のチャンク
    ):
        self.root = Path(root)
        self.s = int(s)
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.block_frames = self.past_frames + self.future_frames  # 6
        self.stride_tokens = int(stride_tokens)                    # 3
        assert self.block_frames == 6, "この実装は past3+future3(=6) 前提"
        assert self.stride_tokens == 3, "この実装は stride=3 前提"

        self.token_dtype = np.dtype(token_dtype)
        self.seg_dtype = np.dtype(seg_dtype)
        self.state_dtype = np.dtype(state_dtype)
        self.state_dim = int(state_dim)

        assert output_format in ("seq2seq", "causal_lm")
        self.output_format = output_format
        self.device = device
        self.use_fixed = use_fixed_17to3_mapping
        self.chunk_tokens = int(chunk_tokens)

        # ---- root metadata ----
        meta_root = json.loads((self.root / "metadata.json").read_text())
        self.num_shards = int(meta_root["num_shards"])
        self.hz = int(meta_root.get("hz", 30))

        # ---- shard paths & lengths (token) ----
        self.video_paths: List[Path] = []
        self.n_tok_by_shard: List[int] = []

        bytes_per_token_frame = self.s * self.s * self.token_dtype.itemsize

        for k in range(self.num_shards):
            m = json.loads((self.root / "metadata" / f"metadata_{k}.json").read_text())
            shard_ind = int(m["shard_ind"])
            vp = self.root / "segment_indices" / "videos" / f"video_{shard_ind}.bin"
            if not vp.exists():
                raise FileNotFoundError(vp)
            self.video_paths.append(vp)

            vb = vp.stat().st_size
            if vb % bytes_per_token_frame != 0:
                raise ValueError(f"{vp} size not divisible by {bytes_per_token_frame}: {vb}")
            self.n_tok_by_shard.append(int(vb // bytes_per_token_frame))

        self.tok_cum = np.concatenate([[0], np.cumsum(self.n_tok_by_shard, dtype=np.int64)])
        self.N_tok_total = int(self.tok_cum[-1])

        # ---- shard paths & lengths (raw segment_idx & robot states) ----
        self.seg_paths: List[Path] = []
        self.state_paths: List[Path] = []
        self.n_raw_by_shard: List[int] = []

        bytes_per_state_frame = self.state_dim * self.state_dtype.itemsize  # 25*4=100 bytes/frame

        for k in range(self.num_shards):
            m = json.loads((self.root / "metadata" / f"metadata_{k}.json").read_text())
            shard_ind = int(m["shard_ind"])
            n_raw = int(m["shard_num_frames"])
            self.n_raw_by_shard.append(n_raw)

            sp = self.root / "segment_indices" / f"segment_idx_{shard_ind}.bin"
            if not sp.exists():
                raise FileNotFoundError(sp)
            self.seg_paths.append(sp)

            stp = self.root / "robot_states" / f"states_{shard_ind}.bin"
            if not stp.exists():
                raise FileNotFoundError(stp)

            # サイズチェック（任意だけど安全）
            sb = stp.stat().st_size
            need = n_raw * bytes_per_state_frame
            if sb != need:
                raise ValueError(f"{stp} size mismatch: on_disk={sb}, need={need} (n_raw={n_raw}, state_dim={self.state_dim})")

            self.state_paths.append(stp)

        self.raw_cum = np.concatenate([[0], np.cumsum(self.n_raw_by_shard, dtype=np.int64)])
        self.N_raw_total = int(self.raw_cum[-1])

        # ---- memmap caches ----
        self._video_mmaps: Dict[int, np.memmap] = {}
        self._seg_mmaps: Dict[int, np.memmap] = {}
        self._state_mmaps: Dict[int, np.memmap] = {}

        # ---- build valid_starts (token indices) ----
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.valid_starts = self._build_or_load_valid_starts()

    def __len__(self) -> int:
        return int(len(self.valid_starts))

    # ---------- memmap helpers ----------
    def _get_video_mmap(self, shard_k: int) -> np.memmap:
        if shard_k not in self._video_mmaps:
            vp = self.video_paths[shard_k]
            n_tok = self.n_tok_by_shard[shard_k]
            self._video_mmaps[shard_k] = np.memmap(
                vp, dtype=self.token_dtype, mode="r", shape=(n_tok, self.s, self.s)
            )
        return self._video_mmaps[shard_k]

    def _get_seg_mmap(self, shard_k: int) -> np.memmap:
        if shard_k not in self._seg_mmaps:
            sp = self.seg_paths[shard_k]
            n_raw = self.n_raw_by_shard[shard_k]
            self._seg_mmaps[shard_k] = np.memmap(
                sp, dtype=self.seg_dtype, mode="r", shape=(n_raw,)
            )
        return self._seg_mmaps[shard_k]

    def _get_state_mmap(self, shard_k: int) -> np.memmap:
        if shard_k not in self._state_mmaps:
            stp = self.state_paths[shard_k]
            n_raw = self.n_raw_by_shard[shard_k]
            self._state_mmaps[shard_k] = np.memmap(
                stp, dtype=self.state_dtype, mode="r", shape=(n_raw, self.state_dim)
            )
        return self._state_mmaps[shard_k]

    def _locate_token(self, g: int) -> Tuple[int, int]:
        shard_k = int(np.searchsorted(self.tok_cum, g, side="right") - 1)
        local = int(g - self.tok_cum[shard_k])
        return shard_k, local

    def _locate_raw(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shard = np.searchsorted(self.raw_cum, r, side="right") - 1
        local = r - self.raw_cum[shard]
        return shard.astype(np.int64), local.astype(np.int64)

    def _read_token_range(self, g0: int, length: int) -> np.ndarray:
        out = np.empty((length, self.s, self.s), dtype=self.token_dtype)
        pos = 0
        g = g0
        while pos < length:
            shard_k, local = self._locate_token(g)
            mm = self._get_video_mmap(shard_k)
            remain = self.n_tok_by_shard[shard_k] - local
            take = min(length - pos, remain)
            out[pos:pos+take] = mm[local:local+take]
            pos += take
            g += take
        return out

    def _read_state_range_raw(self, r0: int, length: int) -> np.ndarray:
        out = np.empty((length, self.state_dim), dtype=self.state_dtype)
        pos = 0
        r = int(r0)

        # ★絶対に末尾を踏まない
        if r < 0:
            r = 0
        if r + length > self.N_raw_total:
            raise IndexError(f"raw range out of bounds: r0={r0}, length={length}, N_raw_total={self.N_raw_total}")

        while pos < length:
            shard_k = int(np.searchsorted(self.raw_cum, r, side="right") - 1)
            local = int(r - self.raw_cum[shard_k])

            mm = self._get_state_mmap(shard_k)
            remain = self.n_raw_by_shard[shard_k] - local
            take = min(length - pos, remain)

            out[pos:pos+take] = mm[local:local+take]
            pos += take
            r += take

        return out

    # ---------- build valid starts ----------
    def _build_or_load_valid_starts(self) -> np.ndarray:
        if self.cache_path is not None and self.cache_path.exists():
            return np.load(self.cache_path)

        def token_to_raw(tok_idx: np.ndarray) -> np.ndarray:
            if self.use_fixed:
                r = np.floor(tok_idx * (17.0 / 3.0)).astype(np.int64)
            else:
                r = ((tok_idx * self.N_raw_total) // self.N_tok_total).astype(np.int64)
            return np.clip(r, 0, self.N_raw_total - 1)

        N = self.N_tok_total
        n_groups = N // 3                # 3-token group 数（端数は捨てる）
        TokN = 3 * n_groups              # 判定に使う token 数

        seg_id = np.empty(TokN, dtype=np.int32)

        t0 = 0
        while t0 < TokN:
            t1 = min(t0 + self.chunk_tokens, TokN)
            t = np.arange(t0, t1, dtype=np.int64)
            r = token_to_raw(t)
            shard, local = self._locate_raw(r)

            order = np.argsort(shard)
            shard_s = shard[order]
            local_s = local[order]
            out = np.empty(t1 - t0, dtype=np.int32)

            p = 0
            while p < len(order):
                sh = int(shard_s[p])
                q = p
                while q < len(order) and int(shard_s[q]) == sh:
                    q += 1
                mm = self._get_seg_mmap(sh)
                out[p:q] = np.array(mm[local_s[p:q]], dtype=np.int32)
                p = q

            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            seg_id[t0:t1] = out[inv]
            t0 = t1

        seg3 = seg_id.reshape(n_groups, 3)
        group_valid = (seg3[:, 0] == seg3[:, 1]) & (seg3[:, 1] == seg3[:, 2])
        group_seg = seg3[:, 0]

        # 6-token sample (= 2 groups) が clip を跨がない条件
        ok = group_valid[:-1] & group_valid[1:] & (group_seg[:-1] == group_seg[1:])
        valid_group_starts = np.nonzero(ok)[0]

        valid_starts = (valid_group_starts.astype(np.int64) * 3)  # token start index

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, valid_starts)

        return valid_starts

    # ---------- dataset output ----------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g0 = int(self.valid_starts[int(idx)])  # token start index, multiple of 3
        frames = self._read_token_range(g0, 6).astype(np.int64, copy=False)  # (6,32,32)

        # ★ raw 開始は 3-token group index k = g0//3 に対して raw_start = 17*k
        k = g0 // 3
        raw_start = int(17 * k)
        # ★末尾ガード：34フレーム読めないなら最後の読める位置に寄せる（またはraise）
        max_start = self.N_raw_total - 34
        if raw_start > max_start:
            raw_start = max_start   # ここが効く（IndexError消える）
        # past17 + future17 = 34
        states_34 = self._read_state_range_raw(raw_start, 34).astype(np.float32, copy=False)  # (34,25)
        states_past = states_34[:17]
        states_future = states_34[17:]

        flat = torch.from_numpy(frames.reshape(-1))  # (6144,)

        if self.output_format == "seq2seq":
            past = flat[: 3 * self.s * self.s]
            fut = flat[3 * self.s * self.s :]
            attn = torch.ones_like(past)

            batch = {
                "input_ids": past,
                "labels": fut,
                "attention_mask": attn,

                # token側（デバッグ用）
                "past_frames": frames[:3],      # (3,32,32) np int64
                "future_frames": frames[3:],    # (3,32,32) np int64
                "token_start": g0,

                # ★ 追加：robot states
                "robot_states": states_34,          # (34,25) np float32
                "robot_states_past": states_past,   # (17,25)
                "robot_states_future": states_future,# (17,25)
                "raw_start": raw_start,             # デバッグ用
            }
        else:
            labels = flat.clone()
            labels[: 3 * self.s * self.s] = -100
            attn = torch.ones_like(flat)
            batch = {
                "input_ids": flat,
                "labels": labels,
                "attention_mask": attn,
                "token_start": g0,

                "robot_states": states_34,
                "robot_states_past": states_past,
                "robot_states_future": states_future,
                "raw_start": raw_start,
            }

        if self.device is not None:
            for k2, v2 in list(batch.items()):
                if torch.is_tensor(v2):
                    batch[k2] = v2.to(self.device)

        return batch
