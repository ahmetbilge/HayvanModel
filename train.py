"""
Görüntü Sınıflandırma Modeli Eğitim Modülü

Bu modül, bir görüntü veri seti kullanarak CNN tabanlı sınıflandırma modeli eğitir.
Eğitilen model ve etiket haritası kaydedilir.

Kullanım:
    python train.py --data_dir "veri_klasörü" --epochs 15 --batch_size 32
"""

import os
import numpy as np
import json
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score


def parse_arguments():
    """
    Komut satırı argümanlarını ayrıştırır.

    Returns:
        argparse.Namespace: Ayrıştırılmış argümanlar
    """
    parser = argparse.ArgumentParser(description='Görüntü sınıflandırma modeli eğitimi')
    parser.add_argument('--data_dir', type=str, default='data_dir', help='Veri seti klasörü')
    parser.add_argument('--image_size', type=int, default=100, help='Görüntü boyutu')
    parser.add_argument('--epochs', type=int, default=15, help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch boyutu')
    parser.add_argument('--output_dir', type=str, default='model', help='Model çıktı klasörü')
    return parser.parse_args()


def load_images(folder_path, image_size=(100, 100)):
    """
    Belirtilen klasördeki görüntüleri yükler ve ön işleme uygular.

    Args:
        folder_path (str): Görüntülerin bulunduğu klasör yolu
        image_size (tuple): Görüntülerin yeniden boyutlandırılacağı boyut

    Returns:
        tuple: (görüntüler, etiketler, etiket_haritası)
    """
    print(f"Görüntüler {folder_path} konumundan yükleniyor...")
    start_time = time.time()

    images, labels = [], []
    label_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    label_map = {name: i for i, name in enumerate(label_names)}

    for label in label_names:
        folder = os.path.join(folder_path, label)
        print(f"İşleniyor: {label} - {len(os.listdir(folder))} görüntü")

        for file in os.listdir(folder):
            try:
                path = os.path.join(folder, file)
                img = Image.open(path).resize(image_size).convert("RGB")
                images.append(np.array(img) / 255.0)  # Normalizasyon
                labels.append(label_map[label])
            except Exception as e:
                print(f"Hata: {file} yüklenirken sorun oluştu - {str(e)}")
                continue

    elapsed_time = time.time() - start_time
    print(f"Yükleme tamamlandı. {len(images)} görüntü, {len(label_names)} sınıf. Süre: {elapsed_time:.2f} saniye.")

    return np.array(images), np.array(labels), label_map


def create_model(input_shape, num_classes):
    """
    CNN modeli oluşturur.

    Args:
        input_shape (tuple): Giriş görüntü boyutu
        num_classes (int): Sınıf sayısı

    Returns:
        tensorflow.keras.models.Sequential: Oluşturulan model
    """
    model = Sequential([
        # İlk konvolüsyon bloğu
        Conv2D(32, (3, 3), activation="relu", padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # İkinci konvolüsyon bloğu
        Conv2D(64, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Üçüncü konvolüsyon bloğu
        Conv2D(128, (3, 3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Tam bağlantılı katmanlar
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_model(model, X_test, y_test, label_map):
    """
    Modeli değerlendirir ve performans metriklerini hesaplar.

    Args:
        model: Eğitilmiş model
        X_test: Test görüntüleri
        y_test: Test etiketleri (one-hot encoded)
        label_map: Etiket haritası

    Returns:
        dict: Performans metrikleri
    """
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nModel Performansı:")
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(f"Kesinlik (Precision): {precision:.4f}")
    print(f"Geri Çağırma (Recall): {recall:.4f}")
    print(f"F1 Skoru: {f1:.4f}")

    # Sınıf isimlerini kullanarak detaylı rapor
    reverse_map = {v: k for k, v in label_map.items()}
    target_names = [reverse_map[i] for i in range(len(label_map))]

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_training_history(history, output_dir):
    """
    Eğitim geçmişini görselleştirir ve kaydeder.

    Args:
        history: Model eğitim geçmişi
        output_dir: Çıktı dizini
    """
    plt.figure(figsize=(12, 5))

    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()

    # Grafikleri kaydet
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


def plot_confusion_matrix(model, X_test, y_test, label_map, output_dir):
    """
    Karışıklık matrisini oluşturur ve görselleştirir.

    Args:
        model: Eğitilmiş model
        X_test: Test görüntüleri
        y_test: Test etiketleri (one-hot encoded)
        label_map: Etiket haritası
        output_dir: Çıktı dizini
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    reverse_map = {v: k for k, v in label_map.items()}
    class_names = [reverse_map[i] for i in range(len(label_map))]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Karışıklık Matrisi')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Değerleri matris üzerine yaz
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')

    # Grafiği kaydet
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def main():
    """
    Ana işlev: Komut satırı argümanlarını alır, modeli eğitir ve değerlendirir.
    """
    args = parse_arguments()

    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)

    # Veri setini yükle
    X, y, label_map = load_images(args.data_dir, image_size=(args.image_size, args.image_size))
    y_cat = to_categorical(y)

    # Eğitim ve test verilerini ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    print(f"Eğitim veri boyutu: {X_train.shape}")
    print(f"Test veri boyutu: {X_test.shape}")
    print(f"Sınıf sayısı: {len(label_map)}")

    # Modeli oluştur
    model = create_model(X.shape[1:], len(label_map))
    model.summary()

    # Veri artırma (data augmentation)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )

    # Erken durdurma ve model kaydetme callback'leri
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(args.output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Modeli eğit
    print("Model eğitimi başlıyor...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # Eğitim geçmişini görselleştir
    plot_training_history(history, args.output_dir)

    # Karışıklık matrisini oluştur
    plot_confusion_matrix(model, X_test, y_test, label_map, args.output_dir)

    # Modeli değerlendir
    metrics = evaluate_model(model, X_test, y_test, label_map)

    # Model ve metrikleri kaydet
    model.save(os.path.join(args.output_dir, "model.h5"))

    # Etiket haritasını kaydet
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)

    # Metrikleri kaydet
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"\nModel ve etiket haritası {args.output_dir} klasörüne kaydedildi.")
    print("Eğitim tamamlandı!")


if __name__ == "__main__":
    main()
