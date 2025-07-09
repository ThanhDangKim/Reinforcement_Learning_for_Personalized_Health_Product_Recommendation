#%% 
# Import các thư viện cần thiết
import json
import pandas as pd
import spacy
import numpy as np
import gym
import tensorflow as tf
import copy
import random
import pylab
import os
import gzip
from urllib.request import urlopen
from collections import deque
from keras import layers, models
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import random
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

# %% 
# Tải dữ liệu JSON cho danh mục Health_and_Personal_Care
product_data = []
with gzip.open('../OriginalDatasets/Health_and_Personal_Care.json.gz', 'rt', encoding='utf-8') as file:
    for line in file:
        product_data.append(json.loads(line.strip()))

# Đọc dữ liệu từ file meta_product_data
meta_product_data = []
with gzip.open('../OriginalDatasets/meta_Health_and_Personal_Care.jsonl.gz', 'rt', encoding='utf-8') as file:
    for line in file:
        meta_product_data.append(json.loads(line.strip()))

# Tạo từ điển từ meta_product_data để tra cứu nhanh
def get_first_valid_image(images):
    """Hàm lấy hình ảnh đầu tiên từ 'large' hoặc 'hi_res'."""
    for img in images:
        if "large" in img and img["large"]:
            return img["large"]
        elif "hi_res" in img and img["hi_res"]:
            return img["hi_res"]
    return None

meta_lookup = {
    item["parent_asin"]: {
        "title": item.get("title"),
        "image": get_first_valid_image(item.get("images", []))
    }
    for item in meta_product_data if "parent_asin" in item
}

# Kết hợp dữ liệu
for product in product_data:
    asin = product.get("asin")
    if asin in meta_lookup:
        # Bổ sung title và image vào product_data nếu khớp với parent_asin
        product["title"] = meta_lookup[asin]["title"]
        product["image"] = meta_lookup[asin]["image"]

# Kiểm tra kết quả (in 5 dòng đầu tiên)
print(f"Tổng số sản phẩm: {len(product_data)}")
print(f"Dữ liệu mẫu sau khi kết hợp (5 sản phẩm đầu tiên):")
for item in product_data[:5]:
    print(item)

# (Tùy chọn) Lưu dữ liệu kết hợp vào file JSON
with open('../ProcessedDatasets/combined_product_data.json', 'w', encoding='utf-8') as output_file:
    json.dump(product_data, output_file, ensure_ascii=False, indent=4)

print("Dữ liệu đã được lưu vào file 'combined_product_data.json'")

# %%
with open('../ProcessedDatasets/combined_product_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Xử lý dữ liệu để giữ lại các trường cần thiết
filtered_data = []
for item in data:
    filtered_item = {
        "asin": item.get("asin"),
        "style": item.get("style"),
        "reviewText": item.get("reviewText"),
        "summary": item.get("summary"),
        "title": item.get("title"),  # Nếu có trường 'title'
        "image": item.get("image")   # Nếu có trường 'image'
    }
    filtered_data.append(filtered_item)

# Lưu dữ liệu đã xử lý vào file mới
with open('../ProcessedDatasets/filtered_reviews.json', 'w', encoding='utf-8') as output_file:
    json.dump(filtered_data, output_file, ensure_ascii=False, indent=4)
    print("Dữ liệu đã xử lý và lưu vào file filtered_reviews.json")

# %%
import json
import os

# Đường dẫn file JSON gốc
input_file = "../ProcessedDatasets/filtered_reviews.json"

# Đường dẫn thư mục để lưu các file chia nhỏ
output_dir = "../ProcessedDatasets/split_files"

# Số lượng phần muốn chia
num_parts = 20

# Đảm bảo thư mục lưu file tồn tại
os.makedirs(output_dir, exist_ok=True)

# Đọc file JSON
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Tổng số phần tử trong file
total_items = len(data)

# Số phần tử mỗi phần
chunk_size = total_items // num_parts
remainder = total_items % num_parts  # Số phần tử dư

# Tách file thành các phần
start = 0
for i in range(num_parts):
    end = start + chunk_size + (1 if i < remainder else 0)  # Phân phối dư
    chunk = data[start:end]
    
    # Ghi phần tách ra thành file JSON mới
    output_file = os.path.join(output_dir, f"part_{i+1}.json")
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(chunk, out_file, ensure_ascii=False, indent=4)
    
    print(f"Đã tạo file: {output_file} với {len(chunk)} phần tử")
    start = end

print("Hoàn tất tách file JSON.")

# %%
import json

# Files uploaded by the user
files = [
    '../ProcessedDatasets/filtered_reviews.json'
]

processed_data = []

# Default image link if the image is missing
default_image_link = "https://drive.google.com/file/d/1YYS_6vN9fdJstPjR-ft6W7Xplu22drEh/view"

# Function to generate a short title (max 4 words)
def generate_short_title(title, review_text):
    if title and len(title.split()) <= 4:
        return title
    if review_text:
        return " ".join(review_text.split()[:4])
    return "No Title"

# Process each file
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for record in data:
            title = record.get("title")
            image = record.get("image", default_image_link)
            review_text = record.get("reviewText")
            
            # Generate title if missing or too long
            short_title = generate_short_title(title, review_text)
            
            # Append the processed record
            processed_data.append({
                "asin": record.get("asin"),
                "title": short_title,
                "image": image
            })

# Save the processed data to a new JSON file
output_file_path = '../ProcessedDatasets/processed_product_data.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(processed_data, output_file, ensure_ascii=False, indent=4)

output_file_path

# %%
# Tải dữ liệu JSON cho danh mục Health_and_Personal_Care
product_data = []
with gzip.open('../OriginalDatasets/Health_and_Personal_Care.json.gz', 'rt', encoding='utf-8') as file:
    for line in file:
        product_data.append(json.loads(line.strip()))

# %% 
# Khởi tạo Spacy cho xử lý ngôn ngữ tự nhiên (NLP) và trích xuất danh từ
nlp_model = spacy.load('en_core_web_sm')

# Tạo DataFrame để dễ dàng thao tác dữ liệu
data_df = pd.DataFrame(product_data)

# Lựa chọn các cột cần thiết
# data_df = data_df[['overall', 'verified', 'reviewerID', 'asin', 'style', 'reviewerName', 'reviewText', 'summary', 'reviewTime', 'title', 'image']]
data_df = data_df[['overall', 'verified', 'reviewerID', 'asin', 'style', 'reviewerName', 'reviewText', 'summary', 'reviewTime']]

# Lọc các đánh giá được xác minh và có điểm đánh giá không rỗng
filtered_data_df = data_df[(data_df['verified'] == True) & (~data_df['overall'].isnull())]

print(filtered_data_df.head(10))

# Định nghĩa lớp sản phẩm cho danh mục Health and Personal Care
class HealthProduct: pass
class Reviewer: pass

# Hàm trích xuất danh từ từ văn bản đánh giá
def extract_nouns(doc):
    try:
        return " ".join([token.text for token in doc if token.pos_ in {"NOUN", "PROPN"}])
    except Exception as e:
        print(f"Lỗi khi trích xuất danh từ: {e}")
        return ""

# %% 
# Nhóm đánh giá theo người dùng
reviewer_dict = {}
reviewer_group_df = filtered_data_df.groupby('reviewerID')

for reviewer_id, group in reviewer_group_df:
    product_list = group[group['asin'].notna()]['asin'].unique()
    if len(product_list) > 5:
        reviewer = Reviewer()
        reviewer.reviewerId = reviewer_id
        reviewer.products = product_list
        reviewer.product_count = len(product_list)
        reviewer_dict[reviewer_id] = reviewer

print(len(reviewer_dict))

# %% 
# Lọc dữ liệu chỉ cho những người dùng có đủ đánh giá
reviewer_values = list(reviewer_dict.values())
filtered_data_df = filtered_data_df[(filtered_data_df['reviewerID'].isin([reviewer.reviewerId for reviewer in reviewer_values]))]
print(len(filtered_data_df))

# %% 
# Nhóm các đánh giá theo mã sản phẩm (ASIN)
filtered_data_df['reviewTime'] = pd.to_datetime(filtered_data_df['reviewTime'])
filtered_data_df.sort_values('reviewTime')
asin_reviewer_group = filtered_data_df.groupby(['asin', 'reviewerID', 'reviewTime'], sort=False)
print(len(asin_reviewer_group))

print("=== DETAILED CONTENT OF asin_reviewer_group ===")

# %% In danh sách các khóa (keys) trong nhóm
print("\n--- Keys in asin_reviewer_group (first 10 keys): ---")
keys = list(asin_reviewer_group.groups.keys())  # Lấy danh sách các khóa
for i, key in enumerate(keys[:10]):  # Giới hạn 10 khóa đầu tiên
    print(f"{i + 1}: {key}")

# In chi tiết các nhóm (giới hạn 5 nhóm đầu)
print("\n--- Detailed Data for First 5 Groups: ---")
for i, key in enumerate(keys[:5]):
    print(f"\n--- Group {i + 1} - Key: {key} ---")
    group_data = asin_reviewer_group.get_group(key)  # Lấy dữ liệu của nhóm
    print(group_data)

# Khởi tạo từ điển để lưu trữ đặc tính sản phẩm như trạng thái
product_states = {}
product_dict = {}

# Tạo từ điển metadata lookup từ meta_product_data
title_image_lookup = {product["asin"]: {"title": product.get("title"), "image": product.get("image")} for product in product_data}

# Duyệt qua từng sản phẩm trong nhóm
for (asin, reviewer_id, review_time), group in asin_reviewer_group:
    if (asin, reviewer_id) in product_states: 
        continue

    product = HealthProduct()
    product.asin = asin
    product.reviewerId = reviewer_id
    product.time = review_time

    if asin in title_image_lookup:
        product.title = title_image_lookup[asin]["title"]
        product.image = title_image_lookup[asin]["image"]

    if asin not in product_dict:
        product_dict[asin] = product
        current_product = product_dict[asin]
        current_product.reviewers = set()
        current_product.formats = set()
        current_product.reviews = set()
        current_product.rating = []

    product.reviewers = product_dict[asin].reviewers

    # Trích xuất dữ liệu "Format" từ cột style
    style_data = group[group['style'].notna()]['style']
    format_list = style_data.apply(lambda x: x.get("Format:", "")).unique().tolist()
    product_dict[asin].formats.update(format_list)

    # Trích xuất các danh từ từ văn bản đánh giá
    review_texts = group[group['reviewText'].notna()]['reviewText']
    extracted_nouns = " ".join(review_texts.apply(lambda x: " ".join([extract_nouns(chunk) for chunk in nlp_model(x).noun_chunks]).strip()).unique())
    product_dict[asin].reviews.update(extracted_nouns)

    # Trích xuất các danh từ từ văn bản đánh giá
    # review_texts = group[group['reviewText'].notna()]['reviewText']
    # if not review_texts.empty:
    #     extracted_nouns = " ".join(
    #         review_texts.apply(lambda x: " ".join([extract_nouns(chunk) for chunk in nlp_model(x).noun_chunks]).strip())
    #     )
    # else:
    #     extracted_nouns = ""
    # product_dict[asin].reviews.update(extracted_nouns)

    # Tính điểm đánh giá dùng phương pháp RMS
    ratings_list = group[group['overall'] > 0]['overall'].tolist()
    product_dict[asin].rating.extend(ratings_list)
    product.ratings = np.sqrt(np.mean([r**2 for r in product_dict[asin].rating]))

    # # Kết hợp thông tin metadata
    formats_text = " ".join(product_dict[asin].formats)
    reviews_text = " ".join(product_dict[asin].reviews)
    product.metadata = " ".join((reviews_text + " " + formats_text).split())

    # Thêm thông tin sản phẩm của người đánh giá vào metadata
    for recent_reviewer in list(product.reviewers)[-2:]:
        previous_state = product_states[(asin, recent_reviewer)]
        product.metadata += " " + previous_state.metadata

    # Kết hợp thông tin metadata
    # formats_text = " ".join(product_dict[asin].formats)
    # reviews_text = " ".join(product_dict[asin].reviews)
    # product.metadata = " ".join(set((reviews_text + " " + formats_text).split()))

    # # Thêm thông tin từ các đánh giá trước
    # for recent_reviewer in list(product.reviewers)[-2:]:
    #     if (asin, recent_reviewer) in product_states:
    #         previous_state = product_states[(asin, recent_reviewer)]
    #         product.metadata += " " + previous_state.metadata

    # # Loại bỏ trùng lặp và ký tự thừa
    # product.metadata = " ".join(set(product.metadata.split()))

    # Cập nhật danh sách người đánh giá
    product_dict[asin].reviewers.add(reviewer_id)

    product_states[(asin, reviewer_id)] = product

product_state_list = list(product_states.values())

# %% 
# Sắp xếp danh sách trạng thái theo thời gian đánh giá
product_state_list.sort(key=lambda x: x.time)
print(len(product_state_list))
user_data = {}

# %% 
# Xây dựng dữ liệu cho từng người dùng và gán thông tin
for product_state in product_state_list:
    if product_state.reviewerId not in user_data:
        user_data[product_state.reviewerId] = Reviewer()
        user_data[product_state.reviewerId].products = set()
    for previous_product in list(user_data[product_state.reviewerId].products)[-2:]:
        previous_state = product_states[(previous_product, product_state.reviewerId)]
        product_state.metadata += previous_state.metadata
    user_data[product_state.reviewerId].products.add(product_state.asin)

# %% 
# Loại bỏ trạng thái sản phẩm không có metadata
product_state_list = [state for state in product_state_list if state.metadata.strip() != '']
print(len(product_state_list))

# %%
print("=== DETAILED ATTRIBUTES OF product_state_list ===")

# Lặp qua từng phần tử trong product_state_list
for i, product_state in enumerate(product_state_list):
    print(f"\n--- Product State {i + 1} ---")
    if hasattr(product_state, '__dict__'):
        attributes = vars(product_state)  # Lấy các thuộc tính nếu là đối tượng
        for key, value in attributes.items():
            print(f"{key}: {value}")
    else:
        print(product_state)  # Nếu không phải đối tượng, in giá trị trực tiếp

    # Giới hạn số phần tử để không in toàn bộ nếu danh sách quá lớn
    if i == 9:  # Dừng lại sau 10 phần tử
        print("\n(Danh sách quá dài. Đã dừng sau 10 phần tử.)")
        break

# %% 
# Tạo môi trường cho hệ thống gợi ý sản phẩm dựa trên dữ liệu mới
class ProductRecommendationEnv(gym.Env):
    def __init__(self, product_states, state_dict, iteration_limit=10):
        self.product_states = product_states
        self.state = self.product_states[0]
        self.state_dict = state_dict
        self.iteration_limit = iteration_limit
        self.index = 0
        self.state.action = 0

    def step(self, actions):
        reward = 0
        done = False
        reviewer_id = self.state.reviewerId
        future_asins = [p for p in reviewer_dict[reviewer_id].products if self.state_dict[(p, reviewer_id)].time > self.state.time]
        matched_recommendation = False
        
        print(future_asins)
        for i in actions:
            if self.product_states[i].asin in future_asins:
                self.action = i
                matched_recommendation = True
                break

        if matched_recommendation:
            reward = 1  # Phần thưởng cao hơn nếu sản phẩm được mua trong tương lai
        else:
            self.action = actions[0]

        self.index += 1

        self.state = self.product_states[self.index]
        print(f"Iteration :{self.index}")
        if (self.iteration_limit == self.index): done = True

        return self.state, reward, done, {}


    def reset(self, iterations=10):
        self.state = self.product_states[0]
        self.iteration_limit = iterations
        self.index = 0
        return self.state

# Tạo môi trường gợi ý
env = ProductRecommendationEnv(product_state_list, product_states, 50)

# %%
# Kiểm tra các giá trị bên ngoài lớp
print("=== External Check ===")
print(f"Product States (first 5): {env.product_states[:5]}")  # In 5 trạng thái đầu tiên
print(f"Initial State: {env.state}")
print(f"State Dictionary (keys): {list(env.state_dict.keys())[:5]}")  # In 5 khóa đầu tiên
print(f"Iteration Limit: {env.iteration_limit}")
print(f"Index: {env.index}")
print(f"Initial State Action: {env.state.action}")

# %%
# In toàn bộ thông tin từ biến env
print("=== ENVIRONMENT DETAILS ===")

# In tổng quan các thuộc tính chính của env
print(f"Iteration Limit: {env.iteration_limit}")
print(f"Current Index: {env.index}")

# In trạng thái ban đầu
print("\n=== Initial State Details ===")
state_attributes = vars(env.state) if hasattr(env.state, '__dict__') else env.state.__dict__
for key, value in state_attributes.items():
    print(f"{key}: {value}")

# In thông tin toàn bộ product_states (giới hạn 5 trạng thái đầu để dễ quan sát)
print("\n=== Product States (First 5) ===")
for i, state in enumerate(env.product_states[:5]):
    print(f"State {i + 1}:")
    state_details = vars(state) if hasattr(state, '__dict__') else state.__dict__
    for key, value in state_details.items():
        print(f"  {key}: {value}")
    print("-" * 50)

# In toàn bộ dictionary state_dict (giới hạn 5 phần tử đầu)
print("\n=== State Dictionary (First 5 Entries) ===")
for i, (key, value) in enumerate(list(env.state_dict.items())[:5]):
    print(f"Key: {key}")
    state_details = vars(value) if hasattr(value, '__dict__') else value.__dict__
    for sub_key, sub_value in state_details.items():
        print(f"  {sub_key}: {sub_value}")
    print("-" * 50)

# %% 
# Cài đặt thuật toán DQN (Deep Q-Learning)
class DQNAgent:
    def __init__(self, state_dim, action_dim, product_states):
        # Khởi tạo các thông số cho DQN Agent
        self.product_states = product_states
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.memory_capacity = 2000
        self.train_start_threshold = 1000
        self.replay_memory = deque(maxlen=self.memory_capacity)
        
        # Xây dựng mô hình và mô hình mục tiêu
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    # Hàm xây dựng mô hình DQN sử dụng metadata và rating
    def build_model(self):
        # Tạo bộ mã hóa văn bản và thích nghi với dữ liệu metadata
        text_encoder = tf.keras.layers.TextVectorization(max_tokens=10000)
        metadata_texts = [product.metadata for product in self.product_states]
        text_encoder.adapt(metadata_texts)

        # Xây dựng mô hình mạng nơron
        model = tf.keras.Sequential([
            text_encoder,
            layers.Embedding(len(text_encoder.get_vocabulary()), 64, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(tf.keras.layers.LSTM(32)),
            layers.Dense(64, activation='relu')
        ])
        
        # Nhập điểm đánh giá và kết hợp các lớp
        rating_input = tf.keras.layers.Input(shape=(1,), name='ratings_input')
        combined_input = layers.concatenate([model.output, rating_input])
        dense_layer = layers.Dense(64, activation='relu')(combined_input)
        output_layer = layers.Dense(len(self.product_states), activation='linear')(dense_layer)

        # Hoàn thành mô hình và biên dịch với loss 'mse' (Mean Squared Error)
        model = tf.keras.Model(inputs=[model.input, rating_input], outputs=output_layer)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Cập nhật trọng số cho mô hình mục tiêu
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Chọn hành động theo chính sách epsilon-greedy
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_dim), 10)
        else:
            q_values = self.model.predict([np.array([state.metadata]), np.array([state.ratings])])
            return np.argpartition(q_values[0], -10)[-10:]

    # Thêm mẫu <s, a, r, s'> vào bộ nhớ replay
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Huấn luyện mô hình dựa trên các mẫu từ bộ nhớ
    def train_model(self):
        if len(self.replay_memory) < self.train_start_threshold:
            return
        batch_size = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, min(self.batch_size, batch_size))

        current_metadata, current_ratings = [], []
        next_metadata, next_ratings = [], []
        actions, rewards, dones = [], [], []

        for exp in range(self.batch_size):
            current_metadata.append(np.array(mini_batch[exp][0].metadata))
            current_ratings.append(np.array(mini_batch[exp][0].ratings))
            actions.append(mini_batch[exp][1])
            rewards.append(mini_batch[exp][2])
            next_metadata.append(np.array(mini_batch[exp][3].metadata))
            next_ratings.append(np.array(mini_batch[exp][3].ratings))
            dones.append(mini_batch[exp][4])

        target_q_values = self.model.predict([np.transpose(current_metadata), np.transpose(current_ratings)])
        next_q_values = self.target_model.predict([np.transpose(next_metadata), np.transpose(next_ratings)])

        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(next_q_values[i])

        self.model.fit([np.transpose(current_metadata), np.transpose(current_ratings)], target_q_values, batch_size=self.batch_size, epochs=1, verbose=1)

#%%
# Khởi tạo DQN Agent
state_dim = len(env.product_states)
action_dim = state_dim
dqn_agent = DQNAgent(state_dim, action_dim, env.product_states)

#%% Huấn luyện DQN Agent
EPISODES = 40
dqn_reward_scores, episode_numbers = [], []
state_cache, done_cache, action_cache = {}, {}, {}

for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset(100)

    while not done:
        if (state.asin, state.reviewerId) in state_cache:
            next_state = state_cache[(state.asin, state.reviewerId)]
            reward = 1
            done = done_cache[(state.asin, state.reviewerId)]
            action = action_cache[(state.asin, state.reviewerId)]
            env.index += 1
        else:
            actions = dqn_agent.choose_action(state)
            next_state, reward, done, _ = env.step(actions)
            action = env.action
            if reward == 1:
                state_cache[(state.asin, state.reviewerId)] = next_state
                done_cache[(state.asin, state.reviewerId)] = done
                action_cache[(state.asin, state.reviewerId)] = env.action

        dqn_agent.store_experience(state, action, reward, next_state, done)
        dqn_agent.train_model()
        score += reward
        state = next_state

        if done:
            dqn_agent.update_target_model()
            dqn_reward_scores.append(score)
            episode_numbers.append(e)
            pylab.plot(episode_numbers, dqn_reward_scores, 'b')
            print(f"DQN - Episode: {e}, Score: {score}, Memory: {len(dqn_agent.replay_memory)}, Epsilon: {dqn_agent.epsilon}")
    
    if score > 65:
        break

# Lưu mô hình đã huấn luyện
dqn_agent.model.save('../agents/Health_and_Personal_Care_DQN', save_format='tf')

# %%
# Cài đặt lớp REINFORCEAgent
class DEEPREINFORCEAgent:
    def __init__(self, state_dim, action_dim, product_states):
        # Khởi tạo các tham số cho REINFORCE Agent
        self.product_states = product_states
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.states, self.actions, self.rewards = [], [], []

    # Xây dựng mô hình REINFORCE
    def build_model(self):
        text_encoder = tf.keras.layers.TextVectorization(max_tokens=10000)
        metadata_texts = [product.metadata for product in self.product_states]
        text_encoder.adapt(metadata_texts)

        model = tf.keras.Sequential([
            text_encoder,
            layers.Embedding(len(text_encoder.get_vocabulary()), 64, mask_zero=True),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation='relu')
        ])
        
        rating_input = tf.keras.layers.Input(shape=(1,), name='ratings_input')
        combined_input = layers.concatenate([model.output, rating_input])
        dense_layer = layers.Dense(64, activation='relu')(combined_input)
        output_layer = layers.Dense(self.action_dim, activation='softmax')(dense_layer)

        model = tf.keras.Model(inputs=[model.input, rating_input], outputs=output_layer)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Lấy hành động dựa trên xác suất của chính sách hiện tại
    def choose_action(self, state):
        policy = self.model.predict([np.array([state.metadata]), np.array([state.ratings])], batch_size=1).flatten()
        return np.random.choice(self.action_dim, 10, p=policy)

    # Tính toán phần thưởng giảm dần
    def compute_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=float)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * self.discount_factor + rewards[t]
            discounted_rewards[t] = cumulative
        return discounted_rewards

    # Thêm trạng thái, hành động và phần thưởng vào bộ nhớ
    def store_experience(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # Huấn luyện mô hình với các mẫu kinh nghiệm
    def train_model(self):
        episode_len = len(self.states)
        discounted_rewards = self.compute_discounted_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        current_metadata, current_ratings = [], []
        advantage_matrix = np.zeros((episode_len, self.action_dim))

        for i in range(episode_len):
            current_metadata.append(self.states[i].metadata)
            current_ratings.append(self.states[i].ratings)
            advantage_matrix[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit([np.array(current_metadata), np.array(current_ratings)], advantage_matrix, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

# %% 
# Khởi tạo môi trường REINFORCE và thiết lập tham số huấn luyện
state_dim = len(env.product_states)
action_dim = state_dim
deep_reinforce_agent = DEEPREINFORCEAgent(state_dim, action_dim, env.product_states)

# %% 
# Vòng lặp huấn luyện cho REINFORCE Agent
EPISODES = 40
dr_reward_scores, episode_numbers = [], []
state_cache, done_cache, action_cache = {}, {}, {}

for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset(100)

    while not done:
        if (state.asin, state.reviewerId) in state_cache:
            next_state = state_cache[(state.asin, state.reviewerId)]
            reward = 1
            done = done_cache[(state.asin, state.reviewerId)]
            action = action_cache[(state.asin, state.reviewerId)]
            env.index += 1
        else:
            actions = deep_reinforce_agent.choose_action(state)
            next_state, reward, done, _ = env.step(actions)
            action = env.action
            if reward == 1:
                state_cache[(state.asin, state.reviewerId)] = next_state
                done_cache[(state.asin, state.reviewerId)] = done
                action_cache[(state.asin, state.reviewerId)] = env.action

        deep_reinforce_agent.store_experience(state, action, reward)
        score += reward
        state = next_state

        if done:
            deep_reinforce_agent.train_model()
            dr_reward_scores.append(score)
            episode_numbers.append(e)
            pylab.plot(episode_numbers, dr_reward_scores, 'b')
            print(f"Deep Reinforce - Episode: {e}, Score: {score}")

    # if score > 65:
    #     break

# Lưu mô hình đã huấn luyện
deep_reinforce_agent.model.save('../agents/Health_and_Personal_Care_DR', save_format='tf')

# %%
# Cài đặt thuật toán Q-Learning
class QLearningAgent:
    def __init__(self, state_dim, action_dim):
        # Khởi tạo các thông số cho Q-Learning Agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        # Khởi tạo Q-Table với tất cả giá trị bằng 0
        self.q_table = np.zeros((self.state_dim, self.action_dim))

    # Chọn hành động theo chính sách epsilon-greedy
    def choose_action(self, state_index):
        if np.random.rand() <= self.epsilon:
            # Chọn ngẫu nhiên một hành động
            return np.random.choice(self.action_dim)
        else:
            # Chọn hành động có giá trị Q cao nhất tại trạng thái đó
            return np.argmax(self.q_table[state_index])

    # Cập nhật Q-Table dựa trên phương trình Q-Learning
    def update_q_table(self, state_index, action, reward, next_state_index, done):
        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[next_state_index])
        
        # Cập nhật giá trị Q cho trạng thái hiện tại và hành động đã chọn
        self.q_table[state_index, action] += self.learning_rate * (target - self.q_table[state_index, action])

        # Giảm giá trị epsilon để khám phá ít hơn theo thời gian
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# %% Khởi tạo Q-Learning Agent
state_dim = len(env.product_states)
action_dim = state_dim
q_agent = QLearningAgent(state_dim, action_dim)

# %% Huấn luyện Q-Learning Agent
EPISODES = 200
ql_reward_scores, episode_numbers = [], []

for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset(50)
    state_index = env.product_states.index(state)  # Chuyển đổi trạng thái sang dạng chỉ số cho Q-Table
    step_count = 0

    while not done:
        step_count += 1 
        # Chọn hành động dựa trên chính sách epsilon-greedy
        action = q_agent.choose_action(state_index)

        # Thực hiện hành động và lấy trạng thái tiếp theo
        next_state, reward, done, _ = env.step([action])
        next_state_index = env.product_states.index(next_state)

        # Cập nhật Q-Table
        q_agent.update_q_table(state_index, action, reward, next_state_index, done)

        score += reward
        state_index = next_state_index  # Cập nhật trạng thái hiện tại thành trạng thái tiếp theo

        if step_count == 12:
            print(f"=== Episode: {e}, Step: {step_count} ===")
            print(f"State Index: {state_index - 1}")
            print(f"Action: {action}")
            print(f"Reward: {reward}")
            print(f"Next State Index: {next_state_index}")
            print(f"Epsilon: {q_agent.epsilon}")
            print(f"Q-Table for State {state_index}: {q_agent.q_table[state_index]}")

        if done:
            ql_reward_scores.append(score)
            episode_numbers.append(e)
            pylab.plot(episode_numbers, ql_reward_scores, 'b')
            print(f"Q-Learning - Episode: {e}, Score: {score}, Epsilon: {q_agent.epsilon}")

    if score > 65:
        break

# Lưu Q-Table để sử dụng cho các tập thử nghiệm sau
np.save('../agents/q_table.npy', q_agent.q_table)

# %%
# Cài đặt thuật toán Monte Carlo Agent
class MonteCarloAgent:
    def __init__(self, state_dim, action_dim):
        # Khởi tạo các thông số cho Monte Carlo Agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        # Khởi tạo bảng giá trị hành động (MC Table) và bộ đếm số lần ghé thăm cho mỗi (state, action)
        self.mc_table = np.zeros((self.state_dim, self.action_dim))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

        # Danh sách lưu trữ trải nghiệm cho mỗi tập
        self.episode = []

    # Hàm chọn hành động epsilon-greedy
    def choose_action(self, state_index):
        if np.random.rand() <= self.epsilon:
            # Chọn ngẫu nhiên một hành động
            return np.random.choice(self.action_dim)
        else:
            # Chọn hành động có giá trị MC cao nhất tại trạng thái đó
            return np.argmax(self.mc_table[state_index])

    # Thêm một trải nghiệm vào bộ nhớ
    def store_experience(self, state_index, action, reward):
        self.episode.append((state_index, action, reward))

    # Cập nhật MC Table sau mỗi tập (episode) sử dụng phương pháp Monte Carlo
    def update_policy(self):
        G = 0  # Tổng phần thưởng từ cuối tập về đầu
        visited_state_actions = set()  # Lưu trữ các (state, action) đã ghé thăm trong tập

        # Duyệt ngược từ cuối tập về đầu để tính toán giá trị cho mỗi (state, action)
        for state_index, action, reward in reversed(self.episode):
            G = self.discount_factor * G + reward

            # Chỉ cập nhật (state, action) lần đầu tiên nó được ghé thăm trong tập
            if (state_index, action) not in visited_state_actions:
                self.returns_sum[(state_index, action)] += G
                self.returns_count[(state_index, action)] += 1
                self.mc_table[state_index, action] = self.returns_sum[(state_index, action)] / self.returns_count[(state_index, action)]
                visited_state_actions.add((state_index, action))

        # Làm trống tập cho tập tiếp theo
        self.episode = []

        # Giảm epsilon để giảm dần việc khám phá
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#%% Khởi tạo Monte Carlo Agent
state_dim = len(env.product_states)
action_dim = state_dim
mc_agent = MonteCarloAgent(state_dim, action_dim)

#%% Huấn luyện Monte Carlo Agent
EPISODES = 200
mc_reward_scores, episode_numbers = [], []

for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset(50)
    state_index = env.product_states.index(state)  # Chuyển đổi trạng thái sang dạng chỉ số

    while not done:
        # Chọn hành động theo chính sách epsilon-greedy
        action = mc_agent.choose_action(state_index)

        # Thực hiện hành động và lấy trạng thái tiếp theo
        next_state, reward, done, _ = env.step([action])
        next_state_index = env.product_states.index(next_state)

        # Lưu trải nghiệm vào bộ nhớ
        mc_agent.store_experience(state_index, action, reward)

        score += reward
        state_index = next_state_index  # Cập nhật trạng thái hiện tại thành trạng thái tiếp theo

        if done:
            # Cập nhật MC Table sau khi hoàn thành tập
            mc_agent.update_policy()
            mc_reward_scores.append(score)
            episode_numbers.append(e)
            pylab.plot(episode_numbers, mc_reward_scores, 'b')
            print(f"Monte Carlo - Episode: {e}, Score: {score}, Epsilon: {mc_agent.epsilon}")
    
    if score > 65:
        break

# Lưu MC Table để sử dụng cho các tập thử nghiệm sau
np.save('../agents/mc_table.npy', mc_agent.mc_table)

# %%
def validate_user_states(user_id, env):
    user_states = [state for state in env.product_states if state.reviewerId == user_id]
    if not user_states:
        raise ValueError(f"Không có dữ liệu liên quan đến người dùng với ID: {user_id}")
    return user_states

def recommend_products_dqn(user_id, state, model, N=5):
    policy = model.predict([np.array([state.metadata]), np.array([state.ratings])], batch_size=1).flatten()
    recommended_indices = np.argpartition(policy, -N)[-N:]
    return [env.product_states[idx].asin for idx in recommended_indices]

def recommend_products_dr(user_id, state, model, N=5):
    policy = model.predict([np.array([state.metadata]), np.array([state.ratings])], batch_size=1).flatten()
    recommended_indices = np.argpartition(policy, -N)[-N:]
    return [env.product_states[idx].asin for idx in recommended_indices]

def recommend_products_qlearning(user_id, state, q_table, N=5):
    state_index = env.product_states.index(state)  # Get index of the current state
    recommended_indices = np.argpartition(q_table[state_index], -N)[-N:]
    return [env.product_states[idx].asin for idx in recommended_indices]

def recommend_products_mc(user_id, state, mc_table, N=5):
    state_index = env.product_states.index(state)  # Get index of the current state
    recommended_indices = np.argpartition(mc_table[state_index], -N)[-N:]
    return [env.product_states[idx].asin for idx in recommended_indices]

# %%
user_id = 'A38YATKTY2TNPJ'
# Load Deep Reinforce model
dr_model = load_model('Health_and_Personal_Care_DR')

try:
    user_states = validate_user_states(user_id, env)
    print(f"Đã tìm thấy {len(user_states)} trạng thái liên quan đến người dùng {user_id}")
    current_state = user_states[-1]
    dr_recommendations = recommend_products_dr(user_id, current_state, dr_model)
    print("Deep Reinforce Recommendations:", dr_recommendations)
except ValueError as e:
    print(e)

# %%
user_id = 'A38YATKTY2TNPJ'
# Load Deep Q learning model
dqn_model = load_model('Health_and_Personal_Care_DQN')

try:
    user_states = validate_user_states(user_id, env)
    print(f"Đã tìm thấy {len(user_states)} trạng thái liên quan đến người dùng {user_id}")
    current_state = user_states[-1]
    dr_recommendations = recommend_products_dqn(user_id, current_state, dr_model)
    print("Deep Q - learning Recommendations:", dr_recommendations)
except ValueError as e:
    print(e)

# %%
user_id = 'A38YATKTY2TNPJ'
# Load Q-Table
q_table = np.load('q_table.npy')

try:
    user_states = validate_user_states(user_id, env)
    print(f"Đã tìm thấy {len(user_states)} trạng thái liên quan đến người dùng {user_id}")
    current_state = user_states[-1]
    dr_recommendations = recommend_products_qlearning(user_id, current_state, dr_model)
    print("Q - learning Recommendations:", dr_recommendations)
except ValueError as e:
    print(e)

# %%
user_id = 'A38YATKTY2TNPJ'
# Load Monte Carlo table
mc_table = np.load('mc_table.npy')

try:
    user_states = validate_user_states(user_id, env)
    print(f"Đã tìm thấy {len(user_states)} trạng thái liên quan đến người dùng {user_id}")
    current_state = user_states[-1]
    dr_recommendations = recommend_products_mc(user_id, current_state, dr_model)
    print("Monte Carlo control Recommendations:", dr_recommendations)
except ValueError as e:
    print(e)

# %%
# Hàm tính Expected Return
def calculate_expected_return(rewards):
    return np.mean(rewards)

# Vẽ biểu đồ Return Line Graph
def plot_return_line_graph(rewards, title):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Return per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.show()

# Vẽ biểu đồ Histogram cho Return
def plot_return_histogram(rewards, title):
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.title(f"{title} - Return Histogram")
    plt.show()

# Vẽ biểu đồ Distribution cho Return
def plot_return_distribution(rewards, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(rewards)
    plt.ylabel("Return")
    plt.title(f"{title} - Return Distribution")
    plt.show()

# Vẽ biểu đồ Regret Line Graph
def plot_regret_line_graph(rewards, max_reward, title):
    regret = [max_reward - reward for reward in rewards]
    cumulative_regret = np.cumsum(regret)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_regret, label="Cumulative Regret")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.title(f"{title} - Regret Line Graph")
    plt.legend()
    plt.show()

# Đếm số lần Termination đạt ngưỡng reward nhất định
def count_terminations(rewards, threshold):
    terminations = sum(1 for reward in rewards if reward >= threshold)
    return terminations

# %% Vẽ biểu đồ cho Q-Learning
max_reward = 48  # Đặt mức reward tối đa đạt được
termination_threshold = 0.85 * max_reward

expected_return_q_learning = calculate_expected_return(ql_reward_scores)  # reward_scores từ Q-Learning
print("Expected Return (Q-Learning):", expected_return_q_learning)

plot_return_line_graph(ql_reward_scores, "Q-Learning Return over Episodes")
plot_return_histogram(ql_reward_scores, "Q-Learning")
plot_return_distribution(ql_reward_scores, "Q-Learning")
plot_regret_line_graph(ql_reward_scores, max_reward, "Q-Learning")

terminations_q_learning = count_terminations(ql_reward_scores, termination_threshold)
print("Number of Terminations (Q-Learning):", terminations_q_learning)

# %% Vẽ biểu đồ cho Monte Carlo
expected_return_mc = calculate_expected_return(mc_reward_scores)  # mc_rewards từ Monte Carlo
print("Expected Return (Monte Carlo):", expected_return_mc)

plot_return_line_graph(mc_reward_scores, "Monte Carlo Return over Episodes")
plot_return_histogram(mc_reward_scores, "Monte Carlo")
plot_return_distribution(mc_reward_scores, "Monte Carlo")
plot_regret_line_graph(mc_reward_scores, max_reward, "Monte Carlo")

terminations_mc = count_terminations(mc_reward_scores, termination_threshold)
print("Number of Terminations (Monte Carlo):", terminations_mc)

# %% Vẽ biểu đồ cho DQN
expected_return_dqn = calculate_expected_return(dqn_reward_scores)  # dqn_rewards từ DQN
print("Expected Return (DQN):", expected_return_dqn)

plot_return_line_graph(dqn_reward_scores, "DQN Return over Episodes")
plot_return_histogram(dqn_reward_scores, "DQN")
plot_return_distribution(dqn_reward_scores, "DQN")
plot_regret_line_graph(dqn_reward_scores, max_reward, "DQN")

terminations_dqn = count_terminations(dqn_reward_scores, termination_threshold)
print("Number of Terminations (DQN):", terminations_dqn)

# %% Vẽ biểu đồ cho Deep Reinforce
expected_return_dr = calculate_expected_return(dr_reward_scores)  # dr_rewards từ DR
print("Expected Return (DR):", expected_return_dr)

plot_return_line_graph(dr_reward_scores, "DR Return over Episodes")
plot_return_histogram(dr_reward_scores, "DR")
plot_return_distribution(dr_reward_scores, "DR")
plot_regret_line_graph(dr_reward_scores, max_reward, "DR")

terminations_dr = count_terminations(dr_reward_scores, termination_threshold)
print("Number of Terminations (DR):", terminations_dr)

# %%
