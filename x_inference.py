import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

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

train_all_data = all_data[all_data["train_type"] != -1]
submit_all_data = all_data[all_data["train_type"] == -1]

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

cate_cols = ["uid", "video_ratio", "video_format", "music_title", "hour", "day", "weekday", "week_hour", "year_weekday", "post_location", "post_text_language"]
submit_data = Pool(data=submit_feature_df, label=submit_label_df["label"], cat_features=cate_cols)
full_train_data = Pool(data=train_feature_df, label=train_label_df["label"], cat_features=cate_cols)


submit_preds = []
for i in range(5):
    # save_model/KFold_catboost_{i}.pkl
    model_path = f"save_model/KFold_catboost_{i}.pkl"
    cb_model = CatBoostRegressor()
    cb_model.load_model(model_path)
    submit_pred = cb_model.predict(submit_data)
    submit_preds.append(submit_pred)
    
submit_preds = np.mean(submit_preds, axis=0)
submit_preds = submit_preds * 1.05
submit_result_pd = pd.DataFrame({
    "pid": submit_all_data["pid"],
    "popularity_score": submit_preds
})
submit_result_pd.to_csv("submit_final.csv", index=False, header=True)