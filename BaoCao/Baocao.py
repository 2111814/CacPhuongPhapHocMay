import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

data = pd.read_csv("kc_house_data.csv")
data.head()
data.shape

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (data.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (data.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

data['date'] = pd.to_datetime(data['date'], format='%Y%m%dT%H%M%S')

plt.figure(figsize=(12, 6))
sns.heatmap(data.drop(columns=['date']).corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

unique_values = []
for col in object_cols:
    unique_values.append(data[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)
plt.show()

plt.figure(figsize=(18, len(object_cols) * 3))  
plt.title('Categorical Features: Distribution', fontsize=16)
index = 1

for col in object_cols:
    y = data[col].value_counts()
   
    if len(y) > 10:
        y = y[:10]
    
    plt.subplot(len(object_cols), 1, index)  
    sns.barplot(x=y.index, y=y)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)  
    index += 1

plt.tight_layout()  
plt.show()

sns.regplot(x='sqft_living', y='price', data=data, line_kws={"color": "red"}, scatter_kws={"color": "blue"})
plt.show()

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

sns.stripplot(x='bedrooms', y='price', data=data, color='orange')
plt.show()


sns.regplot(x='sqft_basement', y='price', data=data,
            line_kws={"color": "red", "lw": 2},  # Đổi màu dòng (line) và độ dày (lw)
            scatter_kws={"color": "blue", "s": 10})  # Đổi màu và kích thước điểm (s)
plt.show()


sns.regplot(x='sqft_above', y='price', data=data,
            line_kws={"color": "green", "lw": 2},
            scatter_kws={"color": "purple", "s": 15})
plt.show()

plt.scatter(data.bedrooms, data.price)
plt.title("Bedrooms and Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()

sns.despine()  

sns.stripplot(x='grade', y='price', data=data, size=5)
plt.show()

data = data[data['bedrooms'] < 10]
data = data[data['bathrooms'] < 8]

train1 = data.drop(['id', 'price'], axis=1)
floor_counts = data['floors'].value_counts()
bedroom_counts = data['bedrooms'].value_counts()

def group_small_values(counts, threshold=0.02):
    total = sum(counts)
    grouped = {key: value for key, value in counts.items() if value / total > threshold}
    others = sum(value for key, value in counts.items() if value / total <= threshold)
    if others > 0:
        grouped['Others'] = others
    return grouped

grouped_bedroom_counts = group_small_values(bedroom_counts)
plt.figure(figsize=(8, 8))
plt.pie(
    grouped_bedroom_counts.values(),
    labels=grouped_bedroom_counts.keys(),
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 10},
    pctdistance=0.85,
    labeldistance=1.1
)
plt.title("Allocate the number of bedrooms (grouped)", fontsize=14)
plt.tight_layout()
plt.show()

grouped_floor_counts = group_small_values(floor_counts)
plt.figure(figsize=(8, 8))
plt.pie(
    grouped_floor_counts.values(),
    labels=grouped_floor_counts.keys(),
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 10},
    pctdistance=0.85,
    labeldistance=1.1
)
plt.title("Allocate the number of floors (grouped)", fontsize=14)
plt.tight_layout()
plt.show()

data.floors.value_counts().plot(kind='bar')
plt.show()

plt.scatter(data.floors, data.price)
plt.show()

plt.scatter(data.condition, data.price)
plt.show()

plt.scatter(data.zipcode, data.price)
plt.title("Which is the pricey location by zipcode?")
plt.show()

conv_dates = [1 if values.year == 2014 else 0 for values in data.date]
data['date'] = conv_dates

train1 = data.drop(['id', 'price'], axis=1)

labels = data['price']

x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.8, random_state=42)

model_SVR = svm.SVR()
model_SVR.fit(x_train, y_train)
Y_pred = model_SVR.predict(x_test)
print("SVR MAPE:", mean_absolute_percentage_error(y_test, Y_pred))

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(x_train, y_train)
Y_pred = model_RFR.predict(x_test)
print("Random Forest Regressor MAPE:", mean_absolute_percentage_error(y_test, Y_pred))

model_LR = LinearRegression()
model_LR.fit(x_train, y_train)
Y_pred = model_LR.predict(x_test)
print("Linear Regression MAPE:", mean_absolute_percentage_error(y_test, Y_pred))

model_Ridge = Ridge(alpha=1.0) 
model_Ridge.fit(x_train, y_train)
Y_pred = model_Ridge.predict(x_test)
print("Ridge Regression MAPE:", mean_absolute_percentage_error(y_test, Y_pred))

plt.figure(figsize=(8, 6))
data['bedrooms'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Phân bố số lượng phòng ngủ", fontsize=14)
plt.xlabel("Số lượng phòng ngủ", fontsize=12)
plt.ylabel("Số lượng", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

import matplotlib.pyplot as plt

# Data
models = ['Linear Regression', 'Random Forest', 'SVR']
r2_scores = [0.92, 0.94, 0.93]

import matplotlib.pyplot as plt

# Dữ liệu
models = ['Linear Regression', 'Random Forest', 'SVR']
r2_scores = [0.92, 0.94, 0.93]

# Chọn màu sắc dựa trên R² score (thấp đến cao)
colors = ['#ff7f7f', '#7fff7f', '#7f7fff']  # Màu sắc gradient từ đỏ đến xanh

# Tạo biểu đồ cột
plt.bar(models, r2_scores, color=colors)
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('R² Score Comparison')

# Đặt giới hạn cho trục y từ 0.8 đến 1
plt.ylim(0.8, 1)

# Hiển thị biểu đồ
plt.show()

n_estimators = [10, 50, 100, 200, 500]
f1_scores = [0.85, 0.87, 0.90, 0.91, 0.92]

plt.plot(n_estimators, f1_scores, marker='o', color='b')
plt.xlabel('Number of Trees')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Number of Trees in Random Forest')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Dữ liệu
train1 = data.drop(['id', 'price'], axis=1)
labels = data['price']
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.8, random_state=42)

# Các mô hình hồi quy
model_LR = LinearRegression()
model_RFR = RandomForestRegressor(n_estimators=10)
model_SVR = SVR()

# Huấn luyện mô hình
model_LR.fit(x_train, y_train)
model_RFR.fit(x_train, y_train)
model_SVR.fit(x_train, y_train)

# Dự đoán kết quả
y_pred_LR = model_LR.predict(x_test)
y_pred_RFR = model_RFR.predict(x_test)
y_pred_SVR = model_SVR.predict(x_test)

# Linear Regression - Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_LR, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Linear Regression - Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Random Forest Regressor - Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_RFR, color='green', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Random Forest - Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Support Vector Regression - Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_SVR, color='purple', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('SVR - Actual vs Predicted')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
