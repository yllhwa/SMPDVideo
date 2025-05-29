# In[1]:
from datasets import load_dataset
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import random
import os

from gensim.models import KeyedVectors
from PIL import Image


random_seed = 2025
random.seed(random_seed)
np.random.seed(random_seed)

if not os.path.exists("./data/combine_data_530.csv"):
    # Load Train dataset
    ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']
    ds_train_users = load_dataset("smpchallenge/SMP-Video", 'users')['train']
    ds_train_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['train']
    ds_train_labels = load_dataset("smpchallenge/SMP-Video", 'labels')['train']
    pd_train_posts = pd.DataFrame(ds_train_posts)
    pd_train_users = pd.DataFrame(ds_train_users)
    pd_train_videos = pd.DataFrame(ds_train_videos)
    pd_train_labels = pd.DataFrame(ds_train_labels)
    train_data = pd_train_posts.merge(pd_train_videos, on=['uid', 'pid'], how='left')
    train_data = train_data.merge(pd_train_labels, on=['uid', 'pid'], how='left')
    train_data = train_data.merge(pd_train_users, on='uid', how='left')

    ds_test_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['test']
    ds_test_users = load_dataset("smpchallenge/SMP-Video", 'users')['test']
    ds_test_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['test']
    pd_test_posts = pd.DataFrame(ds_test_posts)
    pd_test_users = pd.DataFrame(ds_test_users)
    # 用户表去重
    pd_test_users = pd_test_users.drop_duplicates(subset=['uid'])
    pd_test_videos = pd.DataFrame(ds_test_videos)
    test_data = pd_test_posts.merge(pd_test_videos, on=['uid', 'pid'], how='left')
    test_data = test_data.merge(pd_test_users, on='uid', how='left')
    print(len(train_data), len(test_data))

    # In[4]:


    def pandas_split_valid_test_dataset(
        pandas_dataset, valid_ratio=0.1, test_ratio=0.1, shuffle=True
    ):
        index = list(range(len(pandas_dataset)))
        #     if shuffle:
        #         random.shuffle(index)

        length = len(pandas_dataset)
        len_valid = int(length * valid_ratio + 0.6)
        len_test = int(length * test_ratio + 0.6)

        train_data = pandas_dataset.loc[index[: -len_test - len_valid]]
        valid_data = pandas_dataset.loc[index[-len_test - len_valid : -len_test]]
        test_data = pandas_dataset.loc[index[-len_test:]]
        return train_data, valid_data, test_data


    train_df, valid_df, test_df = pandas_split_valid_test_dataset(
        train_data, valid_ratio=0.1, test_ratio=0.1, shuffle=True
    )
    print(len(train_df), len(valid_df), len(test_df))


    train_df["train_type"] = 0
    valid_df["train_type"] = 1
    test_df["train_type"] = 2
    test_data["train_type"] = -1

    all_data = pd.concat([train_df, valid_df, test_df, test_data], axis=0, sort=False)
    all_data = all_data.reset_index(drop=True)
    print(f"all_data length: {len(all_data)}")
    # all_data = all_data.fillna('0')

    all_data.to_csv("./data/combine_data_530.csv", header=True)


# In[ ]:


# In[5]:


all_data = pd.read_csv("./data/combine_data_530.csv", low_memory=False)
# post_content 填充空字符串
all_data["post_content"] = all_data["post_content"].apply(
    lambda x: x if isinstance(x, str) else ""
)


# In[6]:

GLOVE_FILE = "./data/glove.42B.300d/glove.42B.300d.txt"
GLOVE_FILE_BIN = GLOVE_FILE + ".bin"

if not os.path.exists(GLOVE_FILE_BIN):
    wv_model = KeyedVectors.load_word2vec_format(GLOVE_FILE, binary=False, no_header=True)
    wv_model.save(GLOVE_FILE_BIN)
else:
    wv_model = KeyedVectors.load(GLOVE_FILE_BIN, mmap='r')
print("Loaded the word2vec model")

# In[8]:

post_content_split = all_data["post_content"].apply(lambda x: x.lower().split(","))

post_content_ans = []
for sentence in post_content_split:
    v = [wv_model[w] for w in sentence if w in wv_model]
    if len(v) == 0:
        post_content_ans.append(np.zeros(300))
    else:
        post_content_ans.append(np.mean(v, 0))

post_content_feature = np.array(post_content_ans)

pd_post_content_feature = pd.DataFrame(post_content_feature, dtype="float")
pd_post_content_feature.columns = ["post_content_fe_{}".format(i) for i in range(300)]
pd_post_content_feature.to_csv("./data/post_content_feature.csv", header=True, index=None)

print("post_content over!")


# In[11]:


def get_img_data(img_file):
    if os.path.exists(img_file) is True:
        return img_file
    else:
        return "./data/none_picture.jpg"


def get_feature(data_df):
    feature_data = pd.DataFrame()
    feature_data["pid"] = data_df["pid"]
    feature_data["train_type"] = data_df["train_type"]

    Uid_set = set(data_df["uid"])
    Uid_map = dict(zip(Uid_set, list(range(len(Uid_set)))))
    feature_data["uid"] = data_df["uid"].map(Uid_map)

    feature_data["uid_count"] = data_df["uid"].map(
        dict(data_df.groupby("uid")["pid"].count())
    )
    feature_data["mean_label"] = data_df["uid"].map(
        dict(data_df.groupby("uid")["popularity"].mean())
    )

    # post_content base
    feature_data["post_content_len"] = data_df["post_content"].apply(lambda x: len(x))
    feature_data["post_content_number"] = data_df["post_content"].apply(
        lambda x: len(x.lower().split(","))
    )
    feature_data["post_suggested_words_len"] = data_df["post_suggested_words"].apply(
        lambda x: len(x)
    )

    # video base
    # data_df["video_path"] = data_df["video_path"].apply(
    #     lambda x: get_img_data(x)
    # )
    # img_mode_map = {"P": 0, "L": 1, "RGB": 2, "CMYK": 3}
    # img_length, img_width, img_pixel, img_model = [], [], [], []
    # for file in data_df["img_file"]:
    #     pm = Image.open(file)
    #     img_length.append(pm.size[0])
    #     img_width.append(pm.size[1])
    #     img_pixel.append(pm.size[0] * pm.size[1])
    #     img_model.append(img_mode_map[pm.mode])
    # feature_data["img_length"] = img_length
    # feature_data["img_width"] = img_width
    # feature_data["pixel"] = img_pixel
    # feature_data["img_model"] = img_model
    feature_data["video_height"] = data_df["video_height"]
    feature_data["video_width"] = data_df["video_width"]
    feature_data["video_duration"] = data_df["video_duration"]
    feature_data["video_ratio"] = data_df["video_ratio"]
    feature_data["video_format"] = data_df["video_format"]
    feature_data["music_title"] = data_df["music_title"]
    feature_data["music_duration"] = data_df["music_duration"]

    # title svd
    tf_idf_enc_t = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_vec_t = tf_idf_enc_t.fit_transform(data_df["post_content"])
    svd_enc_t = TruncatedSVD(n_components=20, n_iter=100, random_state=2020)
    mode_svd_t = svd_enc_t.fit_transform(tf_idf_vec_t)
    mode_svd_t = pd.DataFrame(mode_svd_t)
    mode_svd_t.columns = ["svd_mode_t_{}".format(i) for i in range(20)]
    feature_data = pd.concat([feature_data, mode_svd_t], axis=1)

    # Tags svd
    tf_idf_enc = TfidfVectorizer(ngram_range=(1, 2))
    data_df["post_suggested_words"] = data_df["post_suggested_words"].apply(
        lambda x: x[1:-1]
    )
    tf_idf_vec = tf_idf_enc.fit_transform(data_df["post_suggested_words"])
    svd_enc = TruncatedSVD(n_components=20, n_iter=100, random_state=2020)
    mode_svd = svd_enc.fit_transform(tf_idf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ["svd_mode_{}".format(i) for i in range(20)]
    feature_data = pd.concat([feature_data, mode_svd], axis=1)

    # Temporal-spatial
    data_df["datetime"] = data_df["post_time"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )
    feature_data["hour"] = data_df["datetime"].apply(lambda x: x.hour)
    feature_data["day"] = data_df["datetime"].apply(lambda x: x.day)
    feature_data["weekday"] = data_df["datetime"].apply(lambda x: x.weekday())
    feature_data["week_hour"] = data_df["datetime"].apply(
        lambda x: x.weekday() * 7 + x.hour
    )
    feature_data["year_weekday"] = data_df["datetime"].apply(
        lambda x: x.isocalendar()[1]
    )

    # User data
    feature_data["user_following_count"] = pd.DataFrame(data_df["user_following_count"], dtype="int")
    feature_data["user_follower_count"] = pd.DataFrame(data_df["user_follower_count"], dtype="int")
    feature_data["user_likes_count"] = pd.DataFrame(data_df["user_likes_count"], dtype="int")
    feature_data["user_video_count"] = pd.DataFrame(data_df["user_video_count"], dtype="int")
    feature_data["user_digg_count"] = pd.DataFrame(data_df["user_digg_count"], dtype="int")
    feature_data["user_heart_count"] = pd.DataFrame(data_df["user_heart_count"], dtype="int")
    feature_data["user_friend_count"] = pd.DataFrame(data_df["user_friend_count"], dtype="int")
    
    # post_location
    feature_data["post_location"] = pd.DataFrame(data_df["post_location"], dtype="str")
    feature_data["post_text_language"] = pd.DataFrame(data_df["post_text_language"], dtype="str")
    
    # 标签流行度统计
    tag_df = data_df[["post_suggested_words", "post_time"]].copy()
    tag_df = tag_df.dropna()
    tag_df = tag_df[tag_df["post_suggested_words"].str.strip() != ""]
    tag_df["date"] = tag_df["post_time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
    tag_df["tag_list"] = tag_df["post_suggested_words"].str.replace(r"[\[\]'\"\s]", "", regex=True).str.split(",")

    tag_df = tag_df.explode("tag_list").rename(columns={"tag_list": "tag"})
    tag_popularity = tag_df.groupby("tag")["date"].nunique().to_dict()

    def compute_avg_tag_popularity(tag_str):
        if not isinstance(tag_str, str):
            return 0
        tags = tag_str.replace("[", "").replace("]", "").replace("'", "").replace("\"", "").split(",")
        tags = [t.strip() for t in tags if t.strip()]
        if not tags:
            return 0
        return np.mean([tag_popularity.get(t, 1) for t in tags])

    feature_data["avg_tag_popularity"] = data_df["post_suggested_words"].apply(compute_avg_tag_popularity)

    # label
    feature_data["label"] = pd.DataFrame(data_df["popularity"], dtype="float")
    return feature_data


save_feature_df = get_feature(all_data)
save_feature_df.to_csv("./data/feature_data_530.csv", header=True, index=None)
print("feature save!")