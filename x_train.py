import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import seaborn as sns

all_data = pd.read_csv("./data/feature_data_530.csv")
# all_data post_location和post_text_language为NaN的转为str Nan
all_data["post_location"] = all_data["post_location"].fillna("NaN").astype(str)
all_data["post_text_language"] = all_data["post_text_language"].fillna("NaN").astype(str)

# 对一些数据取 log 处理
all_data["user_likes_count"] = np.log1p(all_data["user_likes_count"])
all_data["user_heart_count"] = np.log1p(all_data["user_heart_count"])
all_data["user_follower_count"] = np.log1p(all_data["user_follower_count"])
all_data["user_following_count"] = np.log1p(all_data["user_following_count"])
all_data["user_digg_count"] = np.log1p(all_data["user_digg_count"])

# # 加载 smp_video_embeddings_pca.npy
# video_embeddings = np.load("smp_video_embeddings_pca.npy")
# # 将视频嵌入添加到数据集中
# for i in range(video_embeddings.shape[1]):
#     all_data[f"video_embedding_{i}"] = video_embeddings[:, i]

# 加载伪标签
pred_data = pd.read_csv("0.17366836223740170(alpha_1.05).csv")
pred_data = pred_data.rename(columns={"popularity_score": "label"})
# update all_data 中的 label 列
all_data.set_index("pid", inplace=True)
all_data.update(pred_data.set_index("pid"))
all_data.reset_index(inplace=True)

train_all_data = all_data[all_data["train_type"] != -1]
# train_all_data = all_data

submit_all_data = all_data[all_data["train_type"] == -1]

# 训练数据根据 IQR 方法去除异常值
Q1 = train_all_data["label"].quantile(0.25)
Q3 = train_all_data["label"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
train_all_data = train_all_data[
    (train_all_data["label"] >= lower_bound) & (train_all_data["label"] <= upper_bound)
]
# train_all_data = train_all_data[(train_all_data["label"] <= upper_bound)]


train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)

feature_columns = ["pid", "train_type", "label", "mean_label"]

train_label_df = train_all_data[["pid", "label"]]
train_feature_df = train_all_data.drop(feature_columns, axis=1)

submit_label_df = submit_all_data[["pid", "label"]]
submit_feature_df = submit_all_data.drop(feature_columns, axis=1)

print(len(train_feature_df), len(submit_feature_df))
print(len(train_label_df), len(submit_label_df))
print(train_feature_df.columns.to_list())

cb_params = {
    # "objective": "MAPE",
    # "eval_metric": "MAPE",
    # "objective": "RMSE",
    "objective": "Huber:delta=1",
    "eval_metric": "MAPE",
    "learning_rate": 0.01,
    "l2_leaf_reg": 3,
    "max_ctr_complexity": 1,
    "depth": 10,
    "leaf_estimation_method": "Gradient",
    "use_best_model": True,
    # "iterations": 100000,
    # "early_stopping_rounds": 5000,
    "iterations": 100000,
    "early_stopping_rounds": 5000,
    "verbose": 500,
    "task_type": "GPU",
    "devices": "0",
}
# pid,train_type,uid,uid_count,mean_label,post_content_len,post_content_number,post_suggested_words_len,video_height,video_width,video_duration,video_ratio,video_format,music_title,music_duration,hour,day,weekday,week_hour,year_weekday,user_following_count,user_follower_count,user_likes_count,user_video_count,user_digg_count,user_heart_count,user_friend_count,label

cate_cols = ["uid", "video_ratio", "video_format", "music_title", "hour", "day", "weekday", "week_hour", "year_weekday", "post_location", "post_text_language"]
submit_data = Pool(data=submit_feature_df, label=submit_label_df["label"], cat_features=cate_cols)
full_train_data = Pool(data=train_feature_df, label=train_label_df["label"], cat_features=cate_cols)


valid_ans = []
train_proba = []
submit_proba = []
# kfold = KFold(n_splits=5, shuffle=True, random_state=2025)
kfold = KFold(n_splits=5, shuffle=True)
k = 0

for train_idx, valid_idx in kfold.split(train_feature_df, train_label_df):
    fold_train_x, fold_train_y = (
        train_feature_df.loc[train_idx],
        train_label_df["label"].loc[train_idx],
    )
    fold_valid_x, fold_valid_y = (
        train_feature_df.loc[valid_idx],
        train_label_df["label"].loc[valid_idx],
    )

    train_data = Pool(data=fold_train_x, label=fold_train_y, cat_features=cate_cols)
    valid_data = Pool(data=fold_valid_x, label=fold_valid_y, cat_features=cate_cols)

    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(train_data, eval_set=valid_data)
    # 输出重要性特征
    importance = cb_model.get_feature_importance(train_data, type="FeatureImportance")
    feature_importance = pd.DataFrame({"feature": train_feature_df.columns, "importance": importance}).sort_values(by="importance", ascending=False)
    with open(f"feature_importance_{k}.json", "w") as f:
        json.dump(feature_importance.to_dict(orient="records"), f, indent=4)

    valid_pred = cb_model.predict(valid_data)
    valid_mse = mean_squared_error(fold_valid_y, valid_pred)
    valid_mae = mean_absolute_error(fold_valid_y, valid_pred)
    valid_mape = mean_absolute_percentage_error(fold_valid_y, valid_pred)

    print("MSE: %.4f, MAE: %.4f, MAPE: %.4f" % (valid_mse, valid_mae, valid_mape))
    valid_ans.append([valid_mse, valid_mae, valid_mape])

    submit_pred = cb_model.predict(submit_data)
    submit_proba.append(submit_pred)
    
    train_pred = cb_model.predict(full_train_data)
    train_proba.append(train_pred)
    

    # cb_model.save_model("./save_model/KFold_catboost_" + str(k) + ".pkl")
    k += 1

valid_ans = np.mean(valid_ans, axis=0)
print("valid: MSE: %.4f, MAE: %.4f, MAPE: %.4f" % (valid_ans[0], valid_ans[1], valid_ans[2]))

# save result json
submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result["pid"] = submit_label_df["pid"]
result["popularity_score"] = submit_ans.round(decimals=4)
# csv
result.to_csv(
    "submit.csv",
    index=False,
    header=True,
)
# save train result json
train_ans = np.mean(train_proba, axis=0)
train_result = pd.DataFrame()
train_result["pid"] = train_label_df["pid"]
train_result["popularity_score"] = train_ans.round(decimals=4)
# csv
train_result.to_csv(
    "train_result.csv",
    index=False,
    header=True,
)

train_pred = cb_model.predict(full_train_data)
# 可视化预测值与真实值的分布差异
plt.figure(figsize=(10, 6))
sns.kdeplot(train_label_df["label"], label="True Labels", linewidth=2)
sns.kdeplot(train_pred, label="Predicted Labels", linewidth=2)
plt.title("Distribution Comparison of True vs Predicted Labels (Training Set)")
plt.xlabel("Label Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_distribution_comparison.png")
plt.show()

# 计算残差
residuals = train_pred - train_label_df["label"]

plt.figure(figsize=(10, 6))
plt.scatter(train_label_df["label"], residuals, alpha=0.4, edgecolor='k')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title("Residuals vs True Labels")
plt.xlabel("True Label")
plt.ylabel("Residual (Prediction - True)")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_vs_true_labels.png")
plt.show()
