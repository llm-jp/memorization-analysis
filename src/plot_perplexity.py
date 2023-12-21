import argparse
import datetime

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=1000)
args = parser.parse_args()

# ファイルからiterationとpplを読み込む
# x軸にiteration、y軸にpplをとる

date = datetime.datetime.today()


# CSVファイルをUTF-8形式で読み込む
path_csv = "/home/kajiwara/llm-jp-corpus-wg-search-ppl/result/v1_100.csv"  # ファイル名
df_csv = pd.read_csv(path_csv, encoding="UTF8")


df_x = df_csv[df_csv.columns[0]]
df_y = df_csv[df_csv.columns[3]]

df_xy = pd.merge(df_x, df_y, how="outer", left_index=True, right_index=True)

df_xy.columns = ["Training Steps", "Perplexity"]

# IterationごとにPerplexityの平均を計算し、新たなデータフレームを作成
average_perplexity_df = (
    df_xy.groupby("Training Steps")["Perplexity"].mean().reset_index()
)

# 新たなデータフレームの表示
average_perplexity_df = average_perplexity_df[
    average_perplexity_df["Training Steps"] % args.step == 0
]
print(average_perplexity_df)


fig, ax = plt.subplots()
# 点をプロットして散布図を作成する
ax.set_xlim(0, 96656)
# ax.set_ylim(40, 80)
ax.scatter(
    average_perplexity_df["Training Steps"], average_perplexity_df["Perplexity"], s=5
)

# x軸とy軸のラベルを設定する
plt.xlabel("Training Steps")
plt.ylabel("Perplexity")

# グラフをファイルに保存し、表示する
plt.savefig(f"{args.step}_{date}_scatter_plot.png")
plt.show()
print("Done!")
