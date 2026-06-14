# Crater identification simulation using LiDAR on Lunar rover の再現

このフォルダにある論文
`Crater identification simulation using LiDAR on Lunar rover.pdf`
を元に、月面クレータ地形、LiDAR走査、DBSCANによるクレータ検出、RANSACによる円補完、DP/CR/terrain frequency評価を再現するスクリプトを追加した。

## 実行方法

```bash
python3 reproduce_crater_lidar.py
```

出力先は `output/reproduction/`。

## インタラクティブシミュレータ

ブラウザで次のファイルを開く。

```bash
interactive_simulator.html
```

サーバは不要。クレーター直径、クレーター深さ、センサ設置高さ、センサとクレーター前方リムの距離、水平視野角、垂直視野角、LiDAR角度分解能をスライダーで変更できる。角度分解能は隣接する光線どうしの角度として扱い、FOVが分解能で割り切れない場合は余りを両端に均等配分して光線列を中央配置する。クレーター中心は地形座標の原点に固定し、距離変更時はセンサ位置だけが動く。点群は角度分解能で決まる水平・垂直角度グリッドの各ビームにつき最大1点だけ生成し、地面に到達しないビームからは点を生成しない。上面地形ビュー、横断面ビュー、疑似3D地形ビューは軸とスケールを表示し、スクロールで拡大縮小、ドラッグで平行移動できる。軸目盛りとスケールバーは表示範囲に合わせて更新される。検出点数、DP、CR、補完直径、リム高さ、DDRが即時更新される。単一クレーターのため、DDRに関わらずリム高さは0としている。補完円は負標高点だけをDBSCANでクラスタ化し、最大クラスタに対してRANSAC円フィットを行って推定している。推定半径にリム幅を足す補正は行わない。

主な出力:

- `terrain_frequency_vs_volume.png`: 論文Fig. 7相当。クレータ体積とterrain frequencyの関係。
- `dp_vs_point_density.png`: 論文Fig. 10相当。点群密度と検出確率DP。
- `dp_vs_errors.png`: 論文Fig. 11/12相当。測距誤差・垂直角誤差によるDP低下。
- `cr_vs_ddr.png`: 論文Fig. 13相当。DDRと補完率CR。
- `crater_completion_demo.png`: LiDARが取得した部分点群、真のクレータ円、RANSAC補完円の可視化。
- `*.csv`: 各図の数値。

## 現行検出プロセスでのDDR vs CR

インタラクティブシミュレータと同じ単一クレーター、リムなし、負標高点のみ、DBSCAN、RANSACの検出プロセスでDDR vs CRを描くには次を実行する。

```bash
python3 plot_ddr_vs_cr_current.py
```

出力先は `output/ddr_vs_cr_current/`。

- `ddr_vs_cr_current.png`: 現行検出プロセスでのDDR vs CR。
- `ddr_vs_cr_current.csv`: DDR、DP、CR、負標高点数の数値。

## 実装した論文要素

- クレータは論文式(5)(6)の回転放物面モデルで生成。
- 岩は論文式(9)の放物面モデルで生成。
- 複数クレーター地形の再現では、クレーターリム高さはDDR 0.11未満では0、DDR 0.11以上では論文Table 1の形態別範囲からmature、young、freshの中央値をDDRで補間して生成。単一クレーター実験とインタラクティブシミュレータではリム高さを0に固定。リム幅はモデル化せず、地形生成とCR評価の実効半径は入力半径と同一とする。
- terrain frequencyは論文式(10)(11)の2D FFTベース指標で算出。
- LiDARはFOV、画素数、測距誤差、水平角誤差、垂直角誤差を持つレイサンプリングとして実装。
- 検出確率DPは論文式(12)、補完率CRは論文式(13)の円重なり面積で算出。

## 再現上の制約

論文ではPANGU/TINベースの地形交差と詳細な縁点抽出を使っているが、PDFだけでは実装詳細と乱数条件が完全には分からない。そのため、この再現は公開本文の数式から構成した近似実装であり、絶対値を完全一致させるものではない。

特に `cr_vs_ddr.png` は、負標高点群クラスタから直接RANSACする実装のため、論文Fig. 13ほど明確な単調増加にはならない。検出・補完パイプライン自体と、誤差によるDP/CR低下、terrain frequencyとクレータ量の相関は確認できる。
