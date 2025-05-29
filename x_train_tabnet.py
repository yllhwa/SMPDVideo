import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder
import torch

# 读取数据
all_data = pd.read_csv("./data/feature_data_530.csv")
all_data["post_location"] = all_data["post_location"].fillna("NaN").astype(str)
all_data["post_text_language"] = all_data["post_text_language"].fillna("NaN").astype(str)

train_all_data = all_data[all_data["train_type"] != -1].reset_index(drop=True)
submit_all_data = all_data[all_data["train_type"] == -1].reset_index(drop=True)

feature_columns = ["pid", "train_type", "label", "mean_label"]
train_label_df = train_all_data[["pid", "label"]]
train_feature_df = train_all_data.drop(feature_columns, axis=1)

submit_label_df = submit_all_data[["pid", "label"]]
submit_feature_df = submit_all_data.drop(feature_columns, axis=1)

print(len(train_feature_df), len(submit_feature_df))
print(train_feature_df.columns.to_list())

# 类别特征
cate_cols = [
    "uid",
    "video_ratio",
    "video_format",
    "music_title",
    "hour",
    "day",
    "weekday",
    "week_hour",
    "year_weekday",
    "post_location",
    "post_text_language"
]

# Label Encoding
for col in cate_cols:
    le = LabelEncoder()
    all_vals = pd.concat([train_feature_df[col], submit_feature_df[col]]).astype(str)
    le.fit(all_vals)
    train_feature_df[col] = le.transform(train_feature_df[col].astype(str))
    submit_feature_df[col] = le.transform(submit_feature_df[col].astype(str))

cat_idxs = [train_feature_df.columns.get_loc(col) for col in cate_cols]
all_df = pd.concat([train_feature_df, submit_feature_df], axis=0, sort=False)
cat_dims = [all_df[col].nunique() for col in cate_cols]

# TabNet 参数
tabnet_params = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "n_independent": 2,
    "n_shared": 2,
    "momentum": 0.3,
    "mask_type": "entmax",
    "verbose": 1,
    "device_name": "cuda" if torch.cuda.is_available() else "cpu"
}

valid_ans = []
submit_proba = []
kfold = KFold(n_splits=5, shuffle=True, random_state=2025)

X_submit = submit_feature_df.values
k = 0

for train_idx, valid_idx in kfold.split(train_feature_df):
    X_train = train_feature_df.iloc[train_idx].values
    y_train = train_label_df["label"].iloc[train_idx].values.reshape(-1, 1)

    X_valid = train_feature_df.iloc[valid_idx].values
    y_valid = train_label_df["label"].iloc[valid_idx].values.reshape(-1, 1)

    model = TabNetRegressor(cat_idxs=cat_idxs, cat_dims=cat_dims, **tabnet_params)

    model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["mae"],
        max_epochs=200,
        patience=30,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0
    )

    valid_pred = model.predict(X_valid).squeeze()
    valid_mse = mean_squared_error(y_valid, valid_pred)
    valid_mae = mean_absolute_error(y_valid, valid_pred)
    valid_mape = mean_absolute_percentage_error(y_valid, valid_pred)

    print("Fold %d - MSE: %.4f, MAE: %.4f, MAPE: %.4f" % (k, valid_mse, valid_mae, valid_mape))
    valid_ans.append([valid_mse, valid_mae, valid_mape])

    # 推理提交
    submit_pred = model.predict(X_submit).squeeze()
    submit_proba.append(submit_pred)

    # 保存模型（可选）
    # model.save_model(f"./save_model/tabnet_fold_{k}")
    k += 1

valid_ans = np.mean(valid_ans, axis=0)
print("Overall - MSE: %.4f, MAE: %.4f, MAPE: %.4f" % tuple(valid_ans))

# 保存提交
submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result["pid"] = submit_label_df["pid"]
result["popularity_score"] = submit_ans.round(4)
result.to_csv("submit_tabnet.csv", index=False)
