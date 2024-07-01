import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
import matplotlib.pyplot as plt


# Bước 1: Chuẩn bị dữ liệu (đọc ảnh từ thư mục và gán nhãn)
def load_images_from_folders(folder_paths, labels, target_size=(128, 128)):
    images = []
    image_labels = []
    for folder_path, label in zip(folder_paths, labels):
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)  # Thay đổi kích thước ảnh về cùng kích thước
                images.append(img)
                image_labels.append(label)
    return images, image_labels


# Bước 2: Tính đặc trưng HoG của miếng vá ảnh
def extract_hog_features(images, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)


# Bước 3: Huấn luyện mô hình SVM và đánh giá
def train_and_evaluate_svm(hog_features, labels):
    X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.3, random_state=42)
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    # Vẽ ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Human', 'Human'],
                yticklabels=['No Human', 'Human'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return svm, X_test, y_test, y_pred


# Bước 4: Dự đoán face/human từ ảnh đầu vào và trực quan hóa kết quả
def detect_faces_humans(image, svm, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    step_size = 64
    window_size = 128
    for y in range(0, h - window_size, step_size):
        for x in range(0, w - window_size, step_size):
            window = gray[y:y + window_size, x:x + window_size]
            features = hog(window, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                           block_norm='L2-Hys')
            features = features.reshape(1, -1)
            prediction = svm.predict(features)
            if prediction == 1:  # Giả sử nhãn 1 là mặt/người
                cv2.rectangle(image, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
    return image


# Bước 5: Trực quan hóa các ảnh mẫu dự đoán đúng và sai
def visualize_predictions(images, labels, predictions):
    correct_indices = [i for i, (label, pred) in enumerate(zip(labels, predictions)) if label == pred]
    incorrect_indices = [i for i, (label, pred) in enumerate(zip(labels, predictions)) if label != pred]

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(correct_indices[:5]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"True: {labels[idx]}, Pred: {predictions[idx]}")
        plt.axis('off')

    for i, idx in enumerate(incorrect_indices[:5]):
        plt.subplot(2, 5, i + 6)
        plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        plt.title(f"True: {labels[idx]}, Pred: {predictions[idx]}")
        plt.axis('off')

    plt.suptitle('Sample Predictions')
    plt.show()


# Đường dẫn đến các thư mục
folder_0_path = 'no-human'  # Thay thế bằng đường dẫn thực tế
folder_1_path = 'human'  # Thay thế bằng đường dẫn thực tế

# Load dữ liệu từ các thư mục
images, labels = load_images_from_folders([folder_0_path, folder_1_path], [0, 1])

# Tính toán các đặc trưng HoG
hog_features = extract_hog_features(images)

# Huấn luyện mô hình SVM và đánh giá
svm, X_test, y_test, y_pred = train_and_evaluate_svm(hog_features, labels)

# Dự đoán trên một ảnh mới
test_image_path = 'test-1.jpg'  # Thay thế bằng đường dẫn đến ảnh kiểm thử
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, (128, 128))  # Thay đổi kích thước ảnh kiểm thử về cùng kích thước
detected_image = detect_faces_humans(test_image, svm)

cv2.imshow('Detected Faces/Humans', detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Trực quan hóa các ảnh mẫu dự đoán đúng và sai
visualize_predictions(images, labels, y_pred)
