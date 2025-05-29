import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb

# 读取数据
all_data = pd.read_csv("./data/feature_data_530.csv")
all_data["post_location"] = all_data["post_location"].fillna("NaN").astype(str)
all_data["post_text_language"] = all_data["post_text_language"].fillna("NaN").astype(str)

# 分离训练和提交数据
train_all_data = all_data[all_data["train_type"] != -1]
submit_all_data = all_data[all_data["train_type"] == -1]

# 去除异常值
Q1 = train_all_data["label"].quantile(0.25)
Q3 = train_all_data["label"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
train_all_data = train_all_data[(train_all_data["label"] >= lower_bound) & (train_all_data["label"] <= upper_bound)]

train_all_data = train_all_data.reset_index(drop=True)
submit_all_data = submit_all_data.reset_index(drop=True)

feature_columns = ["pid", "train_type", "label", "mean_label"]
train_label_df = train_all_data[["pid", "label"]]
train_feature_df = train_all_data.drop(columns=feature_columns)
submit_label_df = submit_all_data[["pid", "label"]]
submit_feature_df = submit_all_data.drop(columns=feature_columns)

print(len(train_feature_df), len(submit_feature_df))
print(train_feature_df.columns.to_list())

# 类别特征
cate_cols = [
    "uid", "video_ratio", "video_format", "music_title",
    "hour", "day", "weekday", "week_hour", "year_weekday",
    "post_location", "post_text_language"
]

# LightGBM 参数
lgb_params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mape",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 3,
    "verbosity": -1,
    "device": "gpu",  # 使用 GPU
    "gpu_platform_id": 0,
    "gpu_device_id": 0
}

# 转换类别列为 category 类型
for col in cate_cols:
    train_feature_df[col] = train_feature_df[col].astype("category")
    submit_feature_df[col] = submit_feature_df[col].astype("category")

# 交叉验证训练
valid_ans = []
submit_proba = []
kfold = KFold(n_splits=5, shuffle=True, random_state=2025)
k = 0

for train_idx, valid_idx in kfold.split(train_feature_df, train_label_df):
    fold_train_x = train_feature_df.iloc[train_idx]
    fold_train_y = train_label_df.iloc[train_idx]["label"]
    fold_valid_x = train_feature_df.iloc[valid_idx]
    fold_valid_y = train_label_df.iloc[valid_idx]["label"]

    train_dataset = lgb.Dataset(fold_train_x, label=fold_train_y, categorical_feature=cate_cols)
    valid_dataset = lgb.Dataset(fold_valid_x, label=fold_valid_y, categorical_feature=cate_cols)

    lgb_model = lgb.train(
        lgb_params,
        train_set=train_dataset,
        valid_sets=[train_dataset, valid_dataset],
        num_boost_round=100000,
        early_stopping_rounds=1000,
        verbose_eval=500
    )

    # 特征重要性
    importance = lgb_model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        "feature": train_feature_df.columns,
        "importance": importance
    }).sort_values(by="importance", ascending=False)
    with open(f"feature_importance_{k}.json", "w") as f:
        json.dump(feature_importance.to_dict(orient="records"), f, indent=4)

    valid_pred = lgb_model.predict(fold_valid_x)
    valid_mse = mean_squared_error(fold_valid_y, valid_pred)
    valid_mae = mean_absolute_error(fold_valid_y, valid_pred)
    valid_mape = mean_absolute_percentage_error(fold_valid_y, valid_pred)

    print("MSE: %.4f, MAE: %.4f, MAPE: %.4f" % (valid_mse, valid_mae, valid_mape))
    valid_ans.append([valid_mse, valid_mae, valid_mape])

    submit_pred = lgb_model.predict(submit_feature_df)
    submit_proba.append(submit_pred)

    k += 1

# 汇总结果
valid_ans = np.mean(valid_ans, axis=0)
print("valid: MSE: %.4f, MAE: %.4f, MAPE: %.4f" % (valid_ans[0], valid_ans[1], valid_ans[2]))

submit_ans = np.mean(submit_proba, axis=0)
result = pd.DataFrame()
result["pid"] = submit_label_df["pid"]
result["popularity_score"] = submit_ans.round(4)
result.to_csv("submit_lightgbm.csv", index=False, header=True)
