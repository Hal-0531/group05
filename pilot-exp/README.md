実装手順

1.動画準備
・download_movi.pyを実行　→　./tensorflow_datasets/movi_a/128x128/1.0.0/の中にファイルが生成されるはず。
・extract_movi.pyを実行 　→　./datasets/movi_a/videos/の中に動画ファイルが生成される。

2.slot-attentionの学習
・train_slot_pixels.pyを実行
ピクセルベースの動画をもとに学習する。

・visualize.pyを実行
slotの中身と再構成画像の生成ができる。
ある程度分離できていたらよい。

3.dynamicsmodelの学習
・train_dynamics_pixels_v2.pyを実行　
・eval_pixel_dynamics.pyを実行

・train_slot_dynamics_pixels.pyを実行
・eval_slot_dynamics_pixels.pyを実行

どちらもTransformerベースのダイナミクスモデル。

フォルダ名が一部相対パスではなく絶対パスとなっているかもしれません。各自調整してください

