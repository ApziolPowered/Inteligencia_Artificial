import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo CNN para clasificacion de deportes")
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas de entrenamiento (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamano del lote (batch size, default: 64)")
    parser.add_argument("--subset", type=int, default=None, help="Limitar a N imagenes por clase para pruebas rapidas")
    args = parser.parse_args()

    # 1. Localizar el directorio del dataset
    dataset_candidates = [
        os.path.join(os.getcwd(), 'sportimages', 'sportimages'),
        os.path.join(os.getcwd(), 'sportimages'),
        r'C:\Users\axels\Downloads\sportimages\sportimages',
        os.path.join(os.path.expanduser('~'), 'Downloads', 'sportimages', 'sportimages'),
        '/content/sportimages/sportimages',
        '/content/sportimages'
    ]

    DATASET_DIR = None
    for path in dataset_candidates:
        if os.path.exists(path):
            # Comprobar que realmente contenga subcarpetas como 'basket' o 'futbol'
            subitems = os.listdir(path)
            if any(s.lower() in ['basket', 'futbol', 'boxeo', 'americano'] for s in subitems):
                DATASET_DIR = path
                break

    if DATASET_DIR is None:
        raise FileNotFoundError("No se encontro el directorio sportimages con las carpetas de deportes.")

    print(f"[1/7] Dataset ubicado en: {DATASET_DIR}")

    # 2. Cargar las imagenes y registrar categorias
    images = []
    labels = []
    deportes = []

    # Obtener unicamente las carpetas de deportes validas
    subdirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith('.')])
    deportes = subdirs
    print(f"Clases detectadas ({len(deportes)}): {deportes}")

    total_loaded = 0
    for label_idx, sport_name in enumerate(deportes):
        sport_dir = os.path.join(DATASET_DIR, sport_name)
        file_list = [f for f in os.listdir(sport_dir) if re.search(r'\.(jpg|jpeg|png|bmp|tiff)$', f, re.IGNORECASE)]
        if args.subset:
            file_list = file_list[:args.subset]

        for fname in file_list:
            fpath = os.path.join(sport_dir, fname)
            img = plt.imread(fpath)
            if len(img.shape) == 3:
                # Descartar canal alfa si es RGBA
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                images.append(img)
                labels.append(label_idx)
                total_loaded += 1
                if total_loaded % 2000 == 0:
                    print(f"Cargadas {total_loaded} imagenes...", end='\r')

    print(f"\n[2/7] Total de imagenes cargadas: {len(images)}")
    
    # 3. Conversion a arrays de NumPy y verificacion
    X = np.array(images, dtype=np.uint8)
    y = np.array(labels)
    classes = np.unique(y)
    nClasses = len(classes)
    print(f"Salidas unicas ({nClasses} clases): {classes}")

    # 4. Division Train / Test
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[3/7] Train shape: {train_X.shape}, Test shape: {test_X.shape}")

    # 5. Normalizacion y One-Hot Encoding
    train_X = train_X.astype('float32') / 255.0
    test_X = test_X.astype('float32') / 255.0

    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU

    train_Y_one_hot = to_categorical(train_Y, num_classes=nClasses)
    test_Y_one_hot = to_categorical(test_Y, num_classes=nClasses)

    # Division Train / Validacion
    train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    print(f"Entrenamiento final: {train_X.shape}, Validacion: {valid_X.shape}")

    # 6. Definicion del modelo CNN
    print("[4/7] Construyendo arquitectura CNN...")
    input_shape = train_X.shape[1:]  # (21, 28, 3)
    sport_model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.5),

        Flatten(),
        Dense(32, activation='linear'),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(nClasses, activation='softmax')
    ])

    sport_model.summary()

    try:
        opt = tf.keras.optimizers.SGD(learning_rate=1e-3)
    except Exception:
        opt = 'sgd'

    sport_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # 7. Entrenamiento
    print(f"\n[5/7] Iniciando entrenamiento ({args.epochs} epocas, batch={args.batch_size})...")
    history = sport_model.fit(
        train_X, train_label,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(valid_X, valid_label)
    )

    # 8. Guardar modelo
    model_path = os.path.join(os.getcwd(), 'sport.h5')
    sport_model.save(model_path)
    print(f"\n[6/7] Modelo guardado exitosamente en: {model_path}")

    # 9. Evaluacion en Test
    print("\n[7/7] Evaluando modelo en datos de prueba...")
    test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    print(f"Test Loss: {test_eval[0]:.4f}")
    print(f"Test Accuracy: {test_eval[1]:.4f}")

    # Reporte de clasificacion
    preds = sport_model.predict(test_X)
    pred_classes = np.argmax(preds, axis=1)
    print("\nReporte de Clasificacion:")
    print(classification_report(test_Y, pred_classes, target_names=deportes))

    # Guardar grafica de entrenamiento
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validacion')
    plt.title('Precision (Accuracy)')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validacion')
    plt.title('Perdida (Loss)')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()

    plot_path = os.path.join(os.getcwd(), 'curvas_entrenamiento.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafica guardada en: {plot_path}")
    print("\n¡Entrenamiento finalizado con exito!")

if __name__ == '__main__':
    main()
